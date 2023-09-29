"""
Jay Rothenberger (jay.c.rothenberger@ou.edu/gmail.com)

some utility functions for performing a brute force nearest neighbors search via various distance metrics on the GPU.

time complexity of these functions can probably be reduced significantly with an intelligent data structure (kd tree),
but these implementations have been written to perform on heavyweight GPU hardware while being aware of the memory
constraints.
"""

import torch
import numpy as np

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
