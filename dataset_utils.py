import time
import torch


def find_closest_batch(loader1, loader2, distance, m=1, nearest_neighbors=1):
    """
    find the closest examples in loader2 to any example(s) in loader 1, where "closest" is the sum of squared distance of the "nearest_neighbors" nearest neighbors

    :DataLoader loader1: data loader object - unshuffled
    :DataLoader loader2: data loader object - unshuffled
    :callable distance: symmetric two argument distance function
    :int m: number of closest examples to return (returned in distance order)
    :int nearest_neighbors: strictly positive number of neighbors to compute for

    :return: List of (index, distance) tuples
    """
    outers, inners, computes = 0, 0, 0
    outer, inner, compute = 0, 0, 0
    sum_squared_distances = []
    start_outer = time.time()
    for i, (x2, y2) in enumerate(loader2):
        start_inner = time.time()
        print(i, end='\r')
        # to hold distances
        distances = []
        for j, (x1, y1) in enumerate(loader1):
            print(i, j, end='      \r')
            start_compute = time.time()
            # beware this line is not symmetric
            distances.append(distance(x2, x1))
            end_compute = time.time()
            compute += end_compute - start_compute
            computes += 1
        distances = torch.concat(distances, 0)
        distances, indices = torch.sort(distances, 0)
        sum_squared_distances.append(torch.sum(distances[:nearest_neighbors], 0))
        end_inner = time.time()
        inner += end_inner - start_inner
        inners += 1
    end_outer = time.time()
    outer += end_outer - start_outer
    outers += 1
    print(f'outer: {outer - inner - compute}, inner: {inner - compute}, compute: {compute}')
    print(torch.concat(sum_squared_distances, 0).shape)
    values, indices = torch.sort(torch.concat(sum_squared_distances, 0), 0)
    return list(zip(values, indices))[:m]


def add_to_imagefolder(paths, labels, dataset):
    """
    Adds the paths with the labels to an image classification dataset

    :list paths: a list of absolute image paths to add to the dataset
    :list labels: a list of labels for each path
    :Dataset dataset: the dataset to add the samples to
    """

    new_samples = list(zip(paths, labels))

    dataset.samples += new_samples

    return dataset