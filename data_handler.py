__author__ = "Otto van der Himst"
__version__ = "1.0"
__email__ = "otto.vanderhimst@ru.nl"

import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from PIL import Image
import time
import os
import shutil
import traceback
import sys

import util
from util import dump, load, get_sinewave, get_firing_probability

class DataHandler():
    
    def __init__(self, P):
        
        self.pd_data = P.pd_data                   # Path to data directory
        self.pf_mnist = P.pf_mnist                 # Path to MNIST file
        self.pdt_mnist_spikes_train = P.pdt_mnist_spikes_train # Path template to MNIST spikes directory
        self.pdt_mnist_spikes_test = P.pdt_mnist_spikes_test # Path template to MNIST spikes directory
        self.nft_mnist_spikes = P.nft_mnist_spikes # Filename template of MNIST spike slice
        self.tag_mnist_normal = P.tag_mnist_normal
        self.tag_mnist_shuffled = P.tag_mnist_shuffled
        self.tag_mnist_noisy = P.tag_mnist_noisy
        
        self.hertz_i = P.hertz_i     # The hertz of each spike train
        self.ms = P.ms           # The duration in miliseconds of each spike train
        self.s_slice = P.s_slice # The size of each slice of data
        
        self.always_fire = P.always_fire
        
        self.path_affix = "" if not self.always_fire else "_always-fire"
        
        self.p_fire = self.hertz_i/1000 if not self.always_fire else 1
            
        self.rng = np.random.default_rng(P.seed)
        
    def load_mnist(self, tag_mnist):
        """ Load the MNIST dataset. """
        mnist = load(self.pf_mnist.format(tag_mnist))
        return mnist
    
    def shuffle_mnist(self):
        """ Shuffle the MNIST images whilst keeping the same order of labels. """
        
        pf_mnist_shuffled = self.pf_mnist.format(self.tag_mnist_shuffled)
        if os.path.isfile(pf_mnist_shuffled):
            print(f"File <{pf_mnist_shuffled}> already exists, not creating new shuffled MNIST.")
            return
        
        # Load the normally ordered MNIST data and create new arrays of the same shape
        X_train, labels_train, X_test, labels_test = self.load_mnist(self.tag_mnist_normal)
        X_train_shuffled, X_test_shuffled = np.zeros_like(X_train), np.zeros_like(X_test)
        
        for digit in range(10): # Iterate over all ten digits
            
            # Get the indices of the current digit
            i_train = np.where(labels_train==digit)[0]
            i_test = np.where(labels_test==digit)[0]
            
            # Shuffle the indices (note while this only shuffles the images, not the labels)
            np.shuffle(i_train)
            np.shuffle(i_test)
            
            # Reorganize according to the shuffled indices
            X_train_shuffled[i_train] = X_train[np.where(labels_train==digit)[0]]
            X_test_shuffled[i_test] = X_test[np.where(labels_test==digit)[0]]
    
        # Store the shuffled MNIST data
        mnist_shuffled = X_train_shuffled, labels_train, X_test_shuffled, labels_test
        dump(mnist_shuffled, pf_mnist_shuffled)
        print(f"Stored a newly shuffled MNIST to <{pf_mnist_shuffled}>.")
        
    def binarize_pixels(self, pixels):
        """ Convert non-binary (grayscale) pixels to binary (black-and-white) pixels. """
        pixels[pixels > 0] = 1
        return pixels

    def pixels_to_spikes(self, pixel_data):
        
        # Extract data dimensions
        dim_batch, dim_row, dim_col = pixel_data.shape
        
        # Generate poisson spike trains
        spike_data = self.rng.choice([0, 1], (dim_batch, dim_row, dim_col, 2, self.ms), p=[1-self.p_fire, self.p_fire])
        
        # Let either the black or white neuron be active, according to the input
        pixel_data = self.binarize_pixels(pixel_data).flatten()
        spike_data = spike_data.reshape(-1, spike_data.shape[-1])
        pixel_data = np.arange(pixel_data.size) * 2 - pixel_data + 1
        spike_data[pixel_data] *= 0
        spike_data = spike_data.reshape((dim_batch, dim_row, dim_col, 2, self.ms))
        
        spike_data = spike_data.astype(bool) # Convert to a boolean array
        return spike_data
    
    def spikes_to_pixels(self, spike_data):
        
        """ Turn spike data into pixels, where a pixel becomes more white proportional to the number of spikes. """
        
        # Sum over the number of spikes, normalize, then convert to 8-bit pixels
        pixel_data = spike_data.sum(-1)
        max_val = np.max(pixel_data)
        pixel_data = (pixel_data[..., 1] - pixel_data[..., 0]) + max_val
        pixel_data = pixel_data/(max_val*2) * 255
        
        return pixel_data.astype(np.uint8)
        
    def store_mnist_as_spikes(self, tag_mnist, train=True):
        
        """ Convert MNIST (train or test) data to spike data and store it. """
        
        if train: # Store as training data
            pd_mnist_spikes = self.pdt_mnist_spikes_train.format(tag_mnist, self.hertz_i, self.ms, self.s_slice, self.path_affix)
        else:     # Store as test data
            pd_mnist_spikes = self.pdt_mnist_spikes_test.format(tag_mnist, self.hertz_i, self.ms, self.s_slice, self.path_affix)
        
        # Create a directory to store the data
        if os.path.isdir(pd_mnist_spikes): # Only proceed if the directory does not exist
            print("Data directory <{}> already exists, returning ...".format(pd_mnist_spikes))
            #print("NOT IN FACT RETURNING BECAUSE TESTING!!!")
            return
        else:
            os.mkdir(pd_mnist_spikes)
            print("Created directory <{}> for storing spike data.".format(pd_mnist_spikes))
        
        if train: # Get training data
            X, _, _, _ = self.load_mnist(tag_mnist)
        else:     # Get test data
            _, _, X, _ = self.load_mnist(tag_mnist)
        
        # Store the data in slices of size <self.s_slice>
        for i in range(0, X.shape[0], self.s_slice):
            slice_start = i
            slice_end = slice_start + self.s_slice
            X_slice = X[slice_start:slice_end]
            X_spikes = self.pixels_to_spikes(X_slice)
            pf_mnist_spikes = pd_mnist_spikes + self.nft_mnist_spikes.format(tag_mnist, self.hertz_i, self.ms, slice_start, slice_end, self.path_affix)
            dump(X_spikes, pf_mnist_spikes)
            print("Stored a new slice of spike data: <{}>.".format(pf_mnist_spikes))
        print("Done storing MNIST as spikes.")
    
    def create_noisy_mnist(self, noise_percentage, tag_mnist):
        
        # Get data
        X_train, labels_train, X_test, labels_test = self.load_mnist(tag_mnist)
        
        X_train = self.binarize_pixels(X_train)
        invert_X_train = self.rng.choice([True, False], size=X_train.shape, p=[noise_percentage/100, (100-noise_percentage)/100])
        X_train[invert_X_train] = (X_train[invert_X_train]+1) % 2
        
        labels_train = self.binarize_pixels(labels_train)
        invert_labels_train = self.rng.choice([True, False], size=labels_train.shape, p=[noise_percentage/100, (100-noise_percentage)/100])
        labels_train[invert_labels_train] = (labels_train[invert_labels_train]+1) % 2
            
        X_test = self.binarize_pixels(X_test)
        invert_X_test = self.rng.choice([True, False], size=X_test.shape, p=[noise_percentage/100, (100-noise_percentage)/100])
        X_test[invert_X_test] = (X_test[invert_X_test]+1) % 2
        
        labels_test = self.binarize_pixels(labels_test)
        invert_labels_test = self.rng.choice([True, False], size=labels_test.shape, p=[noise_percentage/100, (100-noise_percentage)/100])
        labels_test[invert_labels_test] = (labels_test[invert_labels_test]+1) % 2
        
        noisy_mnist = X_train, labels_train, X_test, labels_test
        pf_noisy_mnist = self.pf_mnist.format(tag_mnist+self.tag_mnist_noisy.format(noise_percentage))
        dump(noisy_mnist, pf_noisy_mnist)
        print("Stored noisy mnist as <{}>.".format(pf_noisy_mnist))
    
    def get_mnist_spikes(self, ith_slice, tag_mnist, train=True):
        
        # Check whether the spike data directory exists
        if train: # Get training spike data
            pd_mnist_spikes = self.pdt_mnist_spikes_train.format(tag_mnist, self.hertz_i, self.ms, self.s_slice, self.path_affix)
        else:     # Get test spike data
            pd_mnist_spikes = self.pdt_mnist_spikes_test.format(tag_mnist, self.hertz_i, self.ms, self.s_slice, self.path_affix)
            
        if not os.path.isdir(pd_mnist_spikes): # Only proceed if the directory does not exist
            print("Data directory <{}> does not exist, returning ...".format(pd_mnist_spikes))
            return
        
        slice_start = ith_slice*self.s_slice
        slice_end = ith_slice*self.s_slice + self.s_slice
        pf_mnist_spikes = pd_mnist_spikes + self.nft_mnist_spikes.format(tag_mnist, self.hertz_i, self.ms, slice_start, slice_end, self.path_affix)
        spike_data = load(pf_mnist_spikes)
        return spike_data
    
    def inspect_spike_data(self, spike_data, ith_slice, s_slice, tag_mnist, n_inspections=3, train=True):
        
        """ Inspect spike data by converting it to pixels and then plotting it.  """
        
        if train:
            X, labels, _, _ = self.load_mnist(tag_mnist)
        else:
            _, _, X, labels = self.load_mnist(tag_mnist)
        
        # Select a few random samples to inspect
        indices = self.rng.integers(0, spike_data.shape[0], n_inspections)
        spike_data = spike_data[indices]
        
        # Obtain pixel version of the spike data samples
        pixel_data = self.spikes_to_pixels(spike_data)
        
        # Compare the original mnist images and the pixel representations of the spike data
        fig, axes = plt.subplots(nrows=n_inspections, ncols=2, figsize=(5, 5))
        for ith_row in range(n_inspections):
            index = indices[ith_row]
            axes[ith_row, 0].imshow(X[ith_slice*s_slice + index], cmap=plt.get_cmap('gray'))
            axes[ith_row, 1].imshow(pixel_data[ith_row], cmap=plt.get_cmap('gray'))
        fig.tight_layout()
        plt.show()
        print("The images concern the numbers {}.".format(labels[indices+ith_slice*s_slice]))