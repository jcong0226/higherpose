import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from lib.models.layers.layers import convbnrelu
from arch_manager import ArchManager
from lib.models.layers.super_layers import SuperInvBottleneck, SuperSepConv2d, SuperConv2d, SuperBatchNorm2d, \
    SuperConvTranspose2d
from lib.models.layers.transformer import TransformerEncoder, TransformerEncoderLayer


def rand(c):
    return random.randint(0, c - 1)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SuperLitePose(nn.Module):
    def __init__(self, cfg, width_mult=1.0, round_nearest=8):
        super(SuperLitePose, self).__init__()
        input_channel = 24
        inverted_residual_setting = [
            # t, c, n, s
            [6, 32, 6, 2],
            [6, 64, 8, 2],
            [6, 96, 10, 2],
            [6, 160, 10, 1],
        ]
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.arch_manager = ArchManager(cfg)
        self.cfg_arch = self.arch_manager.fixed_sample()
        self.first = nn.Sequential(
            convbnrelu(3, 32, ker=3, stride=2),
            convbnrelu(32, 32, ker=3, stride=1, groups=32),
            SuperConv2d(32, input_channel, 1, 1, 0, bias=False),
            SuperBatchNorm2d(input_channel)
        )
        self.channel = [input_channel]
        # building inverted residual blocks
        self.stage = []
        for cnt in range(len(inverted_residual_setting)):
            t, c, n, s = inverted_residual_setting[cnt]
            layer = []
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                layer.append(SuperInvBottleneck(input_channel, output_channel, stride))
                input_channel = output_channel
            layer = nn.Sequential(*layer)
            self.stage.append(layer)
            self.channel.append(output_channel)
        self.stage = nn.ModuleList(self.stage)
        extra = cfg.MODEL.EXTRA
        self.inplanes = self.channel[-1]
        self.deconv_refined, self.deconv_raw, self.deconv_bnrelu = self._make_deconv_layers(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )
        self.num_deconv_layers = extra.NUM_DECONV_LAYERS
        self.loss_config = cfg.LOSS

        self.head_final_refined, self.head_final_raw, self.head_final_channel = self._make_final_layers(
            cfg, extra.NUM_DECONV_FILTERS, cfg.MODEL.NUM_JOINTS_HEAD)
        self.hand_final_refined, self.hand_final_raw, self.hand_final_channel = self._make_final_layers(
            cfg, extra.NUM_DECONV_FILTERS, cfg.MODEL.NUM_JOINTS_HAND)
        self.foot_final_refined, self.foot_final_raw, self.foot_final_channel = self._make_final_layers(
            cfg, extra.NUM_DECONV_FILTERS, cfg.MODEL.NUM_JOINTS_FOOT)
        self.refined_fuse, self.raw_fuse, self.fuse_channel = self._make_fuse_layers(cfg)

        # transformer
        self.pos_embedding_type = cfg.MODEL.POS_EMBEDDING_TYPE  # sine
        self.d_model = self.cfg_arch['backbone_setting'][-1]['channel']
        w = h = self.cfg_arch['img_size'] // 4
        self.encoder_layers_num = cfg.MODEL.ENCODER_LAYERS  # 3
        self.dim_feedforward = self.d_model * 2
        self._make_position_embedding(w, h, self.d_model, self.pos_embedding_type)

        global_encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=self.dim_feedforward,
            activation='relu',
            return_atten_map=False
        )
        self.global_encoder = TransformerEncoder(
            global_encoder_layer,
            self.encoder_layers_num,
            return_atten_map=False
        )

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 4
                self.pe_w = w // 4
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        # logger.info(">> NOTE: this is for testing on unseen input resolutions")
        # # NOTE generalization test with interploation
        # self.pe_h, self.pe_w = 256 // 8 , 192 // 8 #self.pe_h, self.pe_w
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(2, 0, 1)
        return pos  # [h*w, 1, d_model]

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_fuse_layers(self, cfg):
        dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1
        extra = cfg.MODEL.EXTRA
        refined_fuse_layers = []
        raw_fuse_layers = []
        fuse_channel = []
        for i in range(1, extra.NUM_DECONV_LAYERS):
            oup_joint = cfg.MODEL.NUM_JOINTS if cfg.LOSS.WITH_HEATMAPS_LOSS[i - 1] else 0
            oup_tag = dim_tag if cfg.LOSS.WITH_AE_LOSS[i - 1] else 0
            refined_fuse_layers.append(SuperSepConv2d(oup_joint + oup_tag, oup_joint + oup_tag, ker=1))
            raw_fuse_layers.append(SuperSepConv2d(oup_joint + oup_tag, oup_joint + oup_tag, ker=1))
            fuse_channel.append(oup_joint + oup_tag)
        return nn.ModuleList(refined_fuse_layers), nn.ModuleList(raw_fuse_layers), fuse_channel

    def _make_final_layers(self, cfg, num_filters, num_joints):
        dim_tag = num_joints if cfg.MODEL.TAG_PER_JOINT else 1
        extra = cfg.MODEL.EXTRA
        final_raw = []
        final_refined = []
        final_channel = []
        for i in range(1, extra.NUM_DECONV_LAYERS):
            # input_channels = num_filters[i] + self.channel[-i-3]
            oup_joint = num_joints if cfg.LOSS.WITH_HEATMAPS_LOSS[i - 1] else 0
            oup_tag = dim_tag if cfg.LOSS.WITH_AE_LOSS[i - 1] else 0
            final_refined.append(SuperSepConv2d(num_filters[i], oup_joint + oup_tag, ker=5))
            final_raw.append(SuperSepConv2d(self.channel[-i - 3], oup_joint + oup_tag, ker=5))
            final_channel.append(oup_joint + oup_tag)

        return nn.ModuleList(final_refined), nn.ModuleList(final_raw), final_channel

    def _make_deconv_layers(self, num_layers, num_filters, num_kernels):
        deconv_refined = []
        deconv_raw = []
        deconv_bnrelu = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            # inplanes = self.inplanes + self.channel[-i-2]
            layers = []
            deconv_refined.append(
                SuperConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            deconv_raw.append(
                SuperConvTranspose2d(
                    in_channels=self.channel[-i - 2],
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(SuperBatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
            deconv_bnrelu.append(nn.Sequential(*layers))

        return nn.ModuleList(deconv_refined), nn.ModuleList(deconv_raw), nn.ModuleList(deconv_bnrelu)

    def forward(self, x):  # torch.Size([4, 3, 256, 256])
        x = self.first[0](x)  # torch.Size([4, 32, 128, 128])
        x = self.first[1](x)  # torch.Size([4, 32, 128, 128])
        x = self.first[2](x, self.cfg_arch['input_channel'])  # torch.Size([4, 16, 128, 128])
        x = self.first[3](x)  # torch.Size([4, 16, 128, 128])
        x_list = [x]
        backbone_setting = self.cfg_arch['backbone_setting']
        for id_stage in range(len(self.stage)):  # 4
            tmp = x_list[-1]
            n = backbone_setting[id_stage]['num_blocks']  # 4, 6, 8, 8
            s = backbone_setting[id_stage]['stride']  # 2, 2, 2, 1
            c = backbone_setting[id_stage]['channel']  # 16, 32, 48, 80
            block_setting = backbone_setting[id_stage]['block_setting']  # [[6, 7]...]
            for id_block in range(n):
                t, k = block_setting[id_block]
                tmp = self.stage[id_stage][id_block](tmp, c, k, t)
            x_list.append(
                tmp)  # torch.Size([4, 16, 64, 64]) torch.Size([4, 32, 32, 32]) torch.Size([4, 48, 16, 16]) torch.Size([4, 80, 16, 16])

        # 加入transformer,3个块,合并
        # transformer
        input_refined = x_list[-1]  # torch.Size([4, 80, 16, 16])
        b, c, h, w = x_list[-1].shape
        input_refined = input_refined.flatten(2).permute(2, 0, 1)  # torch.Size([256, 4, 80])
        input_refined = self.global_encoder(input_refined, pos=self.pos_embedding)  # torch.Size([256, 4, 80])
        input_refined = input_refined.permute(1, 2, 0).contiguous().view(b, c, h, w)  # torch.Size([4, 80, 16, 16])

        final_outputs = []

        input_raw = x_list[-2]  # torch.Size([4, 48, 16, 16])
        filters = self.cfg_arch['deconv_setting']  # [32, 24, 16]

        for i in range(self.num_deconv_layers):  # 3
            next_input_refined = self.deconv_refined[i](input_refined, filters[i])  # ([4, 32, 32, 32]) ([4, 24, 64, 64]) ([4, 16, 128, 128])
            next_input_raw = self.deconv_raw[i](input_raw, filters[i])  # ([4, 32, 32, 32]) ([4, 24, 64, 64]) ([4, 16, 128, 128])
            input_refined = self.deconv_bnrelu[i](next_input_refined + next_input_raw)  # ([4, 32, 32, 32]) ([4, 24, 64, 64]) ([4, 16, 128, 128])
            input_raw = x_list[-i - 3]  # ([4, 32, 32, 32]) ([4, 16, 64, 64]) ([4, 16, 128, 128])
            if i > 0:
                head_final_refined = self.head_final_refined[i - 1](input_refined, self.head_final_channel[i - 1])  # torch.Size([4, 10, 64, 64]) torch.Size([4, 5, 128, 128])
                head_final_raw = self.head_final_raw[i - 1](input_raw, self.head_final_channel[i - 1])  # torch.Size([4, 10, 64, 64]) torch.Size([4, 5, 128, 128])

                hand_final_refined = self.hand_final_refined[i - 1](input_refined, self.hand_final_channel[i - 1])  # torch.Size([4, 12, 64, 64]) torch.Size([4, 6, 128, 128])
                hand_final_raw = self.hand_final_raw[i - 1](input_raw, self.hand_final_channel[i - 1])  # torch.Size([4, 12, 64, 64]) torch.Size([4, 6, 128, 128])

                foot_final_refined = self.foot_final_refined[i - 1](input_refined, self.foot_final_channel[i - 1])
                foot_final_raw = self.foot_final_raw[i - 1](input_raw, self.foot_final_channel[i - 1])

                final_refined = self.refined_fuse[i - 1](
                    torch.cat((head_final_refined, hand_final_refined, foot_final_refined), 1),
                    self.fuse_channel[i - 1])  # torch.Size([4, 34, 64, 64])
                final_raw = self.raw_fuse[i - 1](
                    torch.cat((head_final_raw, hand_final_raw, foot_final_raw), 1),
                    self.fuse_channel[i - 1])  # torch.Size([4, 34, 64, 64])

                final_outputs.append(final_refined + final_raw)

        return final_outputs  # torch.Size([4, 34, 64, 64]) torch.Size([4, 17, 128, 128])

    def adjust_bn_according_to_idx(self, bn, idx):
        bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
        bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
        if type(bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
            bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
            bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)

    def re_organize_weights(self):
        # Firstly, reorganize first layer
        next_layer = self.stage[0][0].inv[0]
        importance = torch.sum(torch.abs(next_layer.weight.data), dim=(0, 2, 3))
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.first[2].weight.data = torch.index_select(self.first[2].weight.data, 0, sorted_idx)
        self.adjust_bn_according_to_idx(self.first[3], sorted_idx)
        next_layer.weight.data = torch.index_select(next_layer.weight.data, 1, sorted_idx)
        # Secondly, reorganize MobileNetV2 backbone
        for id_stage in range(len(self.stage) - 1):
            n = len(self.stage[id_stage])
            next_layer = self.stage[id_stage + 1][0].inv
            importance = torch.sum(torch.abs(next_layer[0].weight.data), dim=(0, 2, 3))
            sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
            next_layer[0].weight.data = torch.index_select(next_layer[0].weight.data, 1, sorted_idx)
            for id_block in range(n):
                point_conv = self.stage[id_stage][id_block].point_conv
                self.adjust_bn_according_to_idx(point_conv[1], sorted_idx)
                point_conv[0].weight.data = torch.index_select(point_conv[0].weight.data, 0, sorted_idx)
                # have residuals
                if id_block > 0:
                    inv = self.stage[id_stage][id_block].inv
                    inv[0].weight.data = torch.index_select(inv[0].weight.data, 1, sorted_idx)


def get_pose_net(cfg, is_train=False):
    model = SuperLitePose(cfg)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        print(cfg.MODEL.PRETRAINED)
        if os.path.isfile(cfg.MODEL.PRETRAINED):
            print("load pre-train model")
            need_init_state_dict = {}
            state_dict = torch.load(cfg.MODEL.PRETRAINED, map_location=torch.device('cpu'))
            for key, value in state_dict.items():
                if 'final' in key:
                    continue
                if 'deconv' in key:
                    continue
                if 'module' in key:
                    key = key[7:]
                need_init_state_dict[key] = value
            model.load_state_dict(need_init_state_dict, strict=False)
            model.re_organize_weights()
            print("re-organize success!")
    return model
