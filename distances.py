from functools import partial

import torch
import torch.nn as nn
import numpy as np

import timm.models.vision_transformer

import torchvision
import pickle
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of https://github.com/facebookresearch/mae source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------
# 

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


class MAE_distance():
    """
    masked autoencoder distance
    """
    def __init__(self, chkpt_path='/scratch/jroth/mae_pretrain_vit_base.pth', model=vit_base_patch16):
        model = vit_base_patch16()
        checkpoint = torch.load(chkpt_path)

        checkpoint_model = checkpoint['model']  
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        self.model = model

    def __call__(self, x, y, embed_comparison=lambda x: np.linalg.norm(x, 1, -1, True)):

        return embed_comparison(model(x) - model(y))

def min_max_norm(x):
    x = x - x.min()
    return x / x.max()

def block_xy(x, y, splits_x=4, splits_y=4, dtype=torch.float16):
    """
    compute the matrix multiplication between x and y in blocks offloading partial results to 
    cpu to avoid CUDA OOM

    x is split rowwise <splits_x> number of times, y is split columnwise the same way

    1 splits corresponds to using the entire matrix
    """
    rax = torch.zeros((x.shape[0], y.shape[1]), dtype=dtype)

    assert splits_x > 0
    assert splits_y > 0
    
    split_size_x = max(x.shape[0] // max(splits_x, 1), 1)
    split_size_y = max(y.shape[1] // max(splits_y, 1), 1)

    if split_size_x == 1:
        splits_x = x.shape[0]
    if split_size_y == 1:
        splits_y = y.shape[0]
    
    x = x.to(1)
    y = y.to(1)
    # the trivial slice is slice(0, -1)
    for i in range(splits_x + 1):
        slice_i = slice(i*split_size_x, (i + 1)*split_size_x)
        x_slice = x[slice_i]
        for j in range(splits_y + 1):
            slice_j = slice(j*split_size_y, (j + 1)*split_size_y)
            y_slice = y[:, slice_j]
            if min(x_slice.shape) and min(y_slice.shape):
                rax[slice_i, slice_j] = (x_slice @ y_slice).cpu()
    
    return rax
            
def pairwise_cosine(x, y):
    # l2 normalize
    x, y = torch.nn.functional.normalize(x), torch.nn.functional.normalize(y)
    dist = block_xy(x, y.T, dtype=torch.float32)
    return 1 - dist

def block_elementwise_op_sum(x, y, lam, sum_axis=-1, splits_x=4, splits_y=4, dtype=torch.float16):
    """
    block the hadamard product sum of x and y along first axis of x and second axis of y
    x - first argument to block
    y - second argument to block
    lam - block seperable lambda function
    sum_axis - axis along which to sum after applying lam
    
    """
    # TODO: cover edge cases for shapes in general.
    rax = torch.zeros((x.shape[0], y.shape[1]), dtype=dtype)

    assert splits_x > 0
    assert splits_y > 0
    
    split_size_x = max(x.shape[0] // max(splits_x, 1), 1)
    split_size_y = max(y.shape[1] // max(splits_y, 1), 1)

    if split_size_x == 1:
        splits_x = x.shape[0]
    if split_size_y == 1:
        splits_y = y.shape[0]
    
    for i in range(splits_x + 1):
        slice_i = slice(i*split_size_x, (i + 1)*split_size_x)
        x_slice = x[slice_i]
        for j in range(splits_y + 1):
            slice_j = slice(j*split_size_y, (j + 1)*split_size_y)
            y_slice = y[:, slice_j]
            if min(x_slice.shape) and min(y_slice.shape):
                x_slice = x_slice.to(0)
                y_slice = y_slice.to(0)
                rax[slice_i, slice_j] = lam(x_slice, y_slice).sum(sum_axis).cpu()
    
    return rax


def split_elementwise_op_sum(x, y, lam, sum_axis=-1, splits_x=4, splits_y=4, dtype=torch.float16):
    """
    block the hadamard product sum of x and y along first axis of x and second axis of y
    x - first argument to block
    y - second argument to block
    lam - block seperable lambda function
    sum_axis - axis along which to sum after applying lam
    
    """
    # TODO: cover edge cases for shapes in general.
    rax = torch.zeros((x.shape[0], y.shape[1]), dtype=dtype)

    assert splits_x > 0
    assert splits_y > 0
    
    split_size_x = max(x.shape[0] // max(splits_x, 1), 1)
    split_size_y = max(y.shape[1] // max(splits_y, 1), 1)

    if split_size_x == 1:
        splits_x = x.shape[0]
    if split_size_y == 1:
        splits_y = y.shape[0]
    
    for i in range(splits_x + 1):
        slice_i = slice(i*split_size_x, (i + 1)*split_size_x)
        x_slice = x[slice_i]
        for j in range(splits_y + 1):
            slice_j = slice(j*split_size_y, (j + 1)*split_size_y)
            y_slice = y[:, slice_j]
            if min(x_slice.shape) and min(y_slice.shape):
                x_slice = x_slice.to(0)
                y_slice = y_slice.to(0)
                m = (x_slice + y_slice) / 2
                rax[slice_i, slice_j] += (lam(x_slice, m).sum(sum_axis) / 2).cpu()
                rax[slice_i, slice_j] += (lam(y_slice, m).sum(sum_axis) / 2).cpu()
    
    return rax
    

def pairwise_KL(x, y):
    
    x, y = min_max_norm(x), min_max_norm(y)
    x, y = x.unsqueeze(1), y.unsqueeze(0)

    # should be able to do this on gpu and block it.  The elementwise multiply is very expensive before the sum because of broadcasting.
    lam = lambda a, b: torch.where(a*b > 0, a * torch.log(a / b), torch.where(a > 0, -a*torch.log(1.00001 - a), 0.0))
    
    pairs = block_elementwise_op_sum(x, y, lam)
    
    return pairs


def pairwise_JS(x, y):
    x, y = min_max_norm(x), min_max_norm(y)
    x, y = x.unsqueeze(1), y.unsqueeze(0)
    lam = lambda a, b: torch.where(a*b > 0, a * torch.log(a / b), torch.where(a > 0, -a*torch.log(1.00001 - a), 0.0))
    return split_elementwise_op_sum(x, y, lam)

def pairwise_euclidean_dist(x, y):
    x2 = torch.sum(x**2, axis=1)
    y2 = torch.sum(y**2, axis=1)
    
    xy = block_xy(x, y.T, splits_x=4, splits_y=4, dtype=torch.float32)

    x2 = x2.reshape(-1, 1)
    dists = np.maximum(x2 - 2*xy + y2, 0)
    dists = np.sqrt(dists)
    # dists[np.isnan(dists)] = 0.0
    return dists

def pairwise_l1_dist(x, y):
    x, y = x.unsqueeze(1), y.unsqueeze(0)

    return (x - y).sum(-1)


def query(queries, embeddings, return_size=256, retrieval_metric=pairwise_cosine, descending=False):
    distances = retrieval_metric(queries, embeddings)
    distances, inds = distances.sort(descending=False)
    return inds[:, :return_size], distances[:, :return_size]