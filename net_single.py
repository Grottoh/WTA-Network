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
from util import save_img, squeeze_all, is_none
import gc
from network import Network, WTALayer

class NetworkSingle(Network):
    
    """ A fully connected single-layer feedforward WTA network. """

    def __init__(self, P):
        super().__init__(P)
        
        # Designates which MNIST to use
        self.tag_mnist = P.tag_mnist
        
        # The number of output neurons
        self.K = P.K
        
        # True if the network should learn (i.e. update the weights)
        self.learn = P.learn
        
        # Create the input layer, which consists of multiple WTA circuits
        self.layer_i = WTALayer(self.dim_row, self.dim_col, 2, P.ms)
        
        # Create the output layer, which consists of a single WTA circuit
        self.layer_o = WTALayer(1, 1, self.K, P.ms)
        
        # Initialize the weights
        if not self.load_weights_io(P.pf_pretrained_weights): # If weights are not loaded ...
            self.weights = self.initialize_weights((self.K, self.dim_row, self.dim_col, 2))
        
        # Used to keep track of the number of spikes per neurons since the new input image
        self.n_spikes_since_reset = np.zeros(self.K, dtype=np.uint16)
        
        self.e_min_o = -self.log_c * self.dim_row*self.dim_col * self.hertz_o * 0.2 * 0.0001 if is_none(P.e_min_o) else P.e_min_o # ***
        
    def load_weights_io(self, pf_pretrained_weights):
        weights = self.load_weights(pf_pretrained_weights)
        if str(weights) == "nonexistent":
            print("Could not load input-output weights, terminating ...")
            sys.exit()
        if str(weights) != "nopath":
            self.weights = weights
            print(f"Succesfully loaded input-output weights from <{pf_pretrained_weights}>.")
            return True
        return False
    
    def propagate_input(self, spike_times, t):
        """ Fire input neurons according to the current time and the given spike times. """
        
        # Fire input neurons according to the given spike times
        self.layer_i.fire_multiple(spike_times[:, :, :, t], t)
        
        # Compute the excitation and inhibition and update membrane potentials accordingly
        excitation = ( ((self.weights+self.log_c) * spike_times[:, :, :, t]).sum((1, 2, 3)) )
        inhibition = 0 if self.softmax_probabilities else self.inhibitor.get_control(spike_times[:, :, :, t].sum())
        self.layer_o.excitation[0, 0] += excitation - inhibition
        
        if not self.softmax_probabilities: # Limit the membrane potential of neurons
            self.layer_o.excitation[0, 0] = np.clip(self.layer_o.excitation[0, 0], 0, -self.e_min_o)
    
    def propagate(self, spike_times, t, learn=True):
        """ 
        Propogate activity through the network for a single timestep according
        to the given spike times.
        """
        
        # Propogate through the input layer
        self.propagate_input(spike_times, t) 
        
        # Propogate through the output layer
        k_o = self.propagate_layer(self.layer_i, self.layer_o, None,
                                   self.weights, None, 
                             self.n_spikes_since_reset, self.K, 
                             t, self.hertz_o, learn=learn)
        self.weights = np.clip(self.weights, -self.log_c, self.w_max)
        
    def reset(self):
        """ Reset the input and output layer. """
        self.layer_i.reset()
        self.layer_o.reset()
        self.t_last_spike = 0
# =============================================================================
#         print(self.ls_spike_times)
#         self.ls_spike_times = []
# =============================================================================
        
    def train(self, data_handler, P):
        
        # Load the actual MNIST (train) pixel data and corresponding labels
        X, labels, _, _ = data_handler.load_mnist(tag_mnist=self.tag_mnist)
        
        time_start = time.time()
        # Iterate over the spike data in slices of size <data_handler.s_slice>
        for ith_slice in range(0, util.SHAPE_MNIST_TRAIN[0]//data_handler.s_slice):
            
            # Retrieve slice <ith_slice> of the spike data
            spike_data = data_handler.get_mnist_spikes(ith_slice, tag_mnist=self.tag_mnist, train=True)
            for index_im, spike_times in enumerate(spike_data):
                
                # Determine the complete dataset index (rather than that of the slice)
                index_im_all = ith_slice*data_handler.s_slice+index_im
                self.index_im_all = index_im_all # $$$
                self.print_interval = P.print_interval # $$$
                
                # Propogate through the network according to the current timestep and given spike times
                for t in range(data_handler.ms):
                    self.propagate(spike_times, t, learn=self.learn)
                self.reset() # Reset the network between images
                
                # Print, plot, and save data according to the given parameters
                time_start = self.print_plot_save(data_handler, X, labels, index_im_all, util.SHAPE_MNIST_TRAIN[0], P, time_start)
                
                
                # Reset spike counter
                self.n_spikes_since_reset = np.zeros(self.K, dtype=np.uint16)
        
            gc.collect() # Clear memory of unused items # ***
                
    def test(self, data_handler, P):
        
        # Load the actual MNIST (test) pixel data and corresponding labels
        _, _, X, labels = data_handler.load_mnist(tag_mnist=self.tag_mnist)
        
        # The number of times each neuron spiked for each label
        neuron_label_counts = np.zeros((self.K, 10), dtype=np.uint32)
        
        time_start = time.time()
        # Iterate over the spike data in slices of size <data_handler.s_slice>
        for ith_slice in range(0, util.SHAPE_MNIST_TEST[0]//data_handler.s_slice):
        
            # Retrieve slice <ith_slice> of the spike data
            spike_data = data_handler.get_mnist_spikes(ith_slice, tag_mnist=self.tag_mnist, train=False)
            for index_im, spike_times in enumerate(spike_data):
                
                # Determine the complete dataset index (rather than that of the slice)
                index_im_all = ith_slice*data_handler.s_slice+index_im
                self.index_im_all = index_im_all # $$$
                self.print_interval = P.print_interval # $$$
                
                # Retrieve the label of the current image
                label = labels[index_im_all]
            
                # Propogate through the network according to the current timestep and given spike times
                for t in range(data_handler.ms):
                    self.propagate(spike_times, t, learn=self.learn)
                self.reset() # Reset the network between images
                
                # Keep track of the results
                neuron_label_counts[:, label] += self.n_spikes_since_reset
                neuron_image_counts[index_im_all] = self.n_spikes_since_reset
                
                # Print, plot, and save data according to the given parameters
                time_start = self.print_plot_save(data_handler, X, labels, index_im_all, util.SHAPE_MNIST_TEST[0], P, time_start)
                
                # Evaluate results after every <evaluation_interval> images
                if index_im_all > 0 and index_im_all%P.evaluation_interval == 0:
                    print("\nEvaluating results after {} images:".format(index_im_all))
                    self.evaluate_results(data_handler, neuron_label_counts[:, :index_im_all+1],
                                          neuron_image_counts[:index_im_all+1], labels[:index_im_all+1])
                                     
                # Reset spike counter
                self.n_spikes_since_reset = np.zeros(self.K, dtype=np.uint16)
            
            gc.collect() # Clear memory of unused items # ***
        
        print("\nDone, final evaluation:")
        self.evaluate_results(data_handler, neuron_label_counts[:, :index_im_all+1],
                              neuron_image_counts[:index_im_all+1], labels[:index_im_all+1]) # Evaluate the results
                         
    def weights_to_pixels(self):
        
        """ Convert network weights to pixels. """
        # Subtract black weights from white weights
        weights = self.weights[:, :, :, 1] - self.weights[:, :, :, 0]
        weights = (self.weights[:, :, :, 1] + self.log_c) - (self.weights[:, :, :, 0] + self.log_c)
        
        # Normalize the weights, then scale them to pixel values (0-255)
        max_value = np.max(np.abs(weights))
        weight_pixels = ((weights+max_value)/(max_value*2) * 255).astype(np.uint8)
        
        return weight_pixels
      
    def print_plot_save(self, data_handler, X, labels, index_im_all, size_dataset, P, time_start):
        
        # &&&
        self.ls_n_spikes.append(self.n_spikes_since_reset.sum())
        if len(self.ls_n_spikes) > P.print_interval:
            self.ls_n_spikes.pop(0)
            if index_im_all % P.print_interval== 0 and index_im_all > 0:
                print(f"avg spikes over last {P.print_interval}:", np.mean(self.ls_n_spikes))
                print(f"std spikes over last {P.print_interval}:", np.std(self.ls_n_spikes))
                print(f"Min spikes: {np.min(self.ls_n_spikes)}, Max spikes: {np.max(self.ls_n_spikes)}")
                        
        # Print some results if enabled
        if P.print_results and (index_im_all%P.print_interval == 0 or
                                index_im_all == size_dataset-1):
            if index_im_all > 0:
                print("\nProcessing {} images took {} seconds".format(
                    P.print_interval, int(time.time() - time_start)))
                time_start = time.time()
            print("Spikes at image {:5d}, displaying digit {}: {}, total: {:2d}".format(
                index_im_all, labels[index_im_all], 
                self.n_spikes_since_reset, self.n_spikes_since_reset.sum()))
        
        # Plot the networks weights if enabled
        if P.plot_weight_ims and (index_im_all%P.plot_interval == 0 or
                                  index_im_all == size_dataset-1):
            indices = [k for k in range(self.K)] # Determine which neurons to plot
            weight_pixels = self.weights_to_pixels()
            self.plot_weights(weight_pixels, self.n_spikes_since_reset, indices, 
                              index_im_all, [X[index_im_all]], labels[index_im_all])
        
        # Save network weight images if enabled
        if P.save_weight_ims and (index_im_all%P.save_weight_ims_interval == 0 or
                                  index_im_all == size_dataset-1):
            for k, p in enumerate(self.weights_to_pixels()):
                pf_image = P.pd_weight_ims + P.nft_weight_ims.format(index_im_all, k)
                save_img(p, pf_image, normalized=False)
        
        # Save network weights if enabled
        if P.save_weights and index_im_all > 0 and (index_im_all%P.save_weights_interval == 0 or
                                                    index_im_all == size_dataset-1):
            
            pf_weights = P.pd_weights + P.nft_weights.format(index_im_all)
            np.save(pf_weights, self.weights)
        
        return time_start