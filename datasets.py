# from io import BytesIO
import torch
import pandas as pd
import random
from torchvision.io import read_image
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import os


def get_file_name_tuples(dirname, class_map):
    """
    Takes a directory name and a dictionary that serves as a map between the class names and the class integger labels,
    and returns a list of tuples (filename, class)

    :param dirname: a directory name local to the running process
    :param classmap: a dictionary of str: int
    :return: a list of tuples (filename, class) parsed from dirname
    """
    return [(f'{dirname}/{f}', class_map[dirname.split('/')[-1]]) for f in os.listdir(dirname)]

def df_from_dirlist(dirlist, class_map=None):
    """
    Creates a pandas dataframe from the files in a list of directories

    :param dirlist: a list of directories where the last sub-directory is the class label for the images within
    :return: a pandas dataframe with columns=['filepath', 'class']
    """
    # remove trailing slash in path (it will be taken care of later)
    dirlist = [dirname if dirname[-1] != '/' else dirname[:-2] for dirname in dirlist]

    # determine the list of classes by directory names
    # E.g. "foo/bar/cat - (split) -> ['foo', 'bar', 'cat'][-1] == 'cat'
    classes = sorted(list(set([dirname.split('/')[-1] for dirname in dirlist])))

    class_map = {c: str(i) for (i, c) in enumerate(classes)} if class_map is None else class_map
    # find all of the file names
    names = sum([get_file_name_tuples(dirname, class_map) for dirname in dirlist], [])
    return pd.DataFrame(names, columns=['filepath', 'class']), class_map


class ImageDataset(Dataset):
    
    def __init__(self, dirlist, 
                 root_dir='./data/', 
                 image_col='filepath', 
                 label_col='class', 
                 transform=None, 
                 target_transform=None, 
                 cache_size=None, 
                 cache_path=None, 
                 shuffle=False,
                 class_map=None):
        """
        Loads images from a dataframe, and optionally caches them to main memory or local cache, but not both. 

        :param df: pd.DataFrame
        :param transform: transformation to apply to the images
        :param target_transform: transformation to apply to the labels
            
        DEPRECATED
        :param in_memory: bool, load the entire dataset into memory if True
        :param cache_size: maximum number of dataset images to cache to main memory
        
        TODO(mel): add warning for --> raises an error if both are cached
        """
        paths = [root_dir + p for p in dirlist]
        self.dirlist = dirlist
        self.df, self.class_map = df_from_dirlist(paths, class_map=class_map)
        
        self.q = mp.Queue()
        # storing the order of files in cache
        self.df['cache_index'] = self.df.index
        # shuffle the dataframe?  After each epoch?
        self.shuffle = shuffle
        # root directory for the images
        self.root_dir = root_dir
        # column name for the images in the dataframe
        self.image_col = image_col
        # column name for the labels in the dataframe
        self.label_col = label_col
        # transform function for the images
        self.transform = transform
        # transform function for the labels
        self.target_transform = target_transform
        # number of elements to store in the cache
        self.cache_size = cache_size
        # path to the cache file
        self.cache_path = None  # currently not a functional argument due to pickeling (TODO)
        # in-memory cache
        self.cache_dict = dict()
        
        if self.cache_size:
            self.cache_dict = {}
            
        self.tensorsize = None
        self.labelsize = None
        
        # self.fpw = open(cache_path, 'wb')
        # self.fpr = open(cache_path, 'rb')
        
    
    def __del__(self):
        #self.fpw.close()
        #self.fpr.close()
        pass

            
    def __len__(self):
        return len(self.df)
    
    
    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        

    def _read_tensors_from_iostream(self, index):
        # seek to the position of the array
        self.fpr.seek(index * (self.tensorsize + self.labelsize))
        # return read array resized to the correct shape
        tensor = torch.load(BytesIO(self.fpr.read(self.tensorsize)))
        self.fpr.seek(self.tensorsize, 1)
        label = torch.load(BytesIO(self.fpr.read(self.labelsize)))
        return tensor, label

    def _write_tensors_to_iostream(self, index, tensor, label):
        # seek to the position of the array
        self.fpw.seek(index * (self.tensorsize + self.labelsize))
        torch.save(tensor, self.fpw)
        torch.save(label, self.fpw)
        self.fpw.flush()
    
    
    def _fetch_from_in_memory_cache(self, idx):
        """ fetch the image you wanted from disk """
        cache_id = self.df.iloc[idx]['cache_index']

        return self.cache_dict[cache_id]
    
    
    def _fetch_from_local_cache(self, idx):
        """ fetch the image you wanted from file on disk """
        tensor, label = self._read_tensors_from_iostream(self.df.iloc[idx]['cache_index'])

        return tensor, label
    
    
    def _fetch_from_disk(self, idx):
        """ fetch the image you wanted from disk """
        image_path = self.df[self.image_col].iloc[idx]
        image = read_image(image_path)

        label = self.df[self.label_col].iloc[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image.to(torch.float32), int(label)
    
    
    def _add_to_local_cache(self, idx):
        
        image, label = self._fetch_from_disk(idx)
        
        self._write_tensors_to_iostream(self.df.iloc[idx]['cache_index'], image, label)
        
        return image, label
    
    
    def _add_to_in_memory_cache(self, idx):
        
        cache_id = self.df.iloc[idx]['cache_index']

        self.cache_dict[idx] = self._fetch_from_disk(idx)
            
        return self.cache_dict[idx]

    
    def _fetch_from_cache(self, idx):
        
        if idx in self.cache_dict:
            # If item is in cache, return it. 
            if self.cache_path is not None:
                cache_id = self.df[self.image_col].iloc[idx]

                return self._fetch_from_local_cache(idx)
            return self._fetch_from_in_memory_cache(cache_id, idx)
        
        elif self.cache_size is None:
            if self.cache_path:
                return self._add_to_local_cache(idx)
            
            return self._add_to_in_memory_cache(idx)
        
        elif len(self.cache_dict) <= self.cache_size:
            # If cache is not full, add item to cache and return it.
            
            if self.cache_path:
                return self._add_to_local_cache(idx)
            
            return self._add_to_in_memory_cache(idx)
            
        elif len(self.cache_dict) > self.cache_size:
            # If cache is full remove an entry.
            self.cache_dict.pop(random.choice(list(self.cache_dict.keys())))
            return self._fetch_from_disk(idx)
        
        
    def _fetch(self, idx):
        # if the idx is an integer
        if self.cache_dict is not None or self.cache_path:
            return self._fetch_from_cache(idx)
        else:
            # otherwise just fetch and return from disk
            return self._fetch_from_disk(idx)

        
    def _normalize_image(self, image):
        return (image - 127.5) / 128

    
    def __getitem__(self, idx):
        
        #if self.tensorsize is None:
        #    bytesio = BytesIO()
        #    tensor, label = self._fetch_from_disk(0)
        #    torch.save(tensor, bytesio)
        #    self.tensorsize = bytesio.getbuffer().nbytes # size is given in bytes
        #    torch.save(label, bytesio)
        #    self.labelsize = bytesio.getbuffer().nbytes - self.tensorsize # size is given in bytes
        
        if isinstance(idx, int):
            
            image, label = self._fetch(idx)
            # TODO: remove, this is best handled by a transform
            return image.to(torch.float32), label
        
        elif isinstance(idx, slice): 
            
            img_paths = self.df[self.image_col].iloc[idx]

            labels = self.df[self.label_col].iloc[idx]
                
            slice_dataset = ImageDataset(self.dirlist, self.root_dir, self.image_col, self.label_col, self.transform, self.target_transform, cache_path=self.cache_path)
            
            slice_dataset.df = self.df.iloc[idx]

            return slice_dataset
        
        else:
            raise ValueError(f'dataset index must be integer or slice found: {type(idx)}')
