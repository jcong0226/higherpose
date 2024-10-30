from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math, copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

logger = logging.getLogger(__name__)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers,
                 norm=None, pe_only_at_begin=False, return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, h, w,
                pos: Optional[Tensor] = None):
        output = src
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(output, pos=pos)
                atten_maps_list.append(att_map)
            else:
                output = layer(output, h, w, pos=pos)

            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        else:
            return output


def _get_clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    """ Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="gelu", normalize_before=False, return_atten_map=False):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_atten_map = return_atten_map

        self.sr_ratio = 1
        self.dw = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=self.sr_ratio, stride=self.sr_ratio, groups=d_model, bias=True),
            nn.BatchNorm2d(d_model, eps=1e-5),
        )

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src, H, W,
                     pos: Optional[Tensor] = None):
        src1 = self.norm1(src)
        N, B, C = src1.shape
        q = self.with_pos_embed(src1, pos)
        src1 = src.permute(1, 2, 0).reshape(B, C, H, W)
        src1 = self.dw(src1).reshape(B, C, -1).permute(2, 0, 1)
        k = self.with_pos_embed(src1, pos)
        v = self.with_pos_embed(src1, pos)

        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, v)
        else:
            src2 = self.self_attn(q, k, v)[0]
        src = src + self.dropout1(src2)

        src3 = self.norm2(src)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src = src + self.dropout2(src3)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward_pre(self, src,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src)
        else:
            src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward(self, src, h, w,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, pos)
        return self.forward_post(src, h, w, pos)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = _get_clones(nn.Linear(d_model, d_model), 4)  # create 4 linear layers
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        query, key, value = [l(x)
                             for l, x in zip(self.linears, (query, key, value))]  # (batch_size, seq_length, d_model), use first 3 self.linears
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x in (query, key, value)]  # (batch_size, h, seq_length, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)
