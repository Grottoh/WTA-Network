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
from util import save_img, squeeze_all, get_firing_probability, get_sinewave, is_none
import gc
from network import Network, WTALayer, Inhibitor
from net_hierarchical import NetworkHierarchical

class NetworkIntegration(Network):
    
    def __init__(self, P, P_a, P_b):
        super().__init__(P)
        
        # Number of neurons in the final layer
        self.K = P.K
        
        # True if the network should learn (i.e. update the weights)
        self.learn = P.learn
        
        # Create the networks to be integrated
        self.net_a = NetworkHierarchical(P_a)
        self.net_b = NetworkHierarchical(P_b)
        
        # Keep the parameters of networks a and b
        self.P_a = P_a
        self.P_b = P_b
        
        # Create the final layer, which consists of a single WTA circuit
        self.layer = WTALayer(1, 1, self.K, P.ms)
        
        # Set the weights from the two hierarchical networks to the final layer
        if not self.load_weights_f(P.pf_pretrained_weights): # If weights are not loaded ...
            self.weights = self.initialize_weights((2, self.K, self.net_a.K_o))
        
        # Used to keep track of the number of spikes per neurons since the new input image
        self.n_spikes_since_reset = np.zeros(self.K, dtype=np.uint16)
        
        self.e_min_o = -self.log_c * self.dim_row*self.dim_col * self.hertz_o * 0.2 * 0.0001 if is_none(P.e_min_o) else P.e_min_o # ***
        
        # ***
        inh_start_o = self.log_c if is_none(P.inh_start_o) else P.inh_start_o
        self.inhibitor = Inhibitor(P.hertz_o, inh_start_o, P.inh_change_o, P.inh_ratio_o, P.ms)
        
        
        if self.topdown_enabled:
            self.weights_td = self.initialize_weights((2, self.net_a.K_o, self.K))
            self.weights_td = self.weights_td - self.log_c/1.2
    
    def load_weights_f(self, pf_pretrained_weights):
        weights = self.load_weights(pf_pretrained_weights)
        if str(weights) == "nonexistent":
            print("Could not load input-output weights, terminating ...")
            sys.exit()
        if str(weights) != "nopath":
            self.weights = weights
            print(f"Succesfully loaded input-output weights from <{pf_pretrained_weights}>.")
            return True
        return False
    
    def propagate(self, spike_times_a, spike_times_b, t):
        
        # Propagate through the two to-be-integrated networks
        ks_o_a = self.net_a.propagate(spike_times_a, t, learn_h=self.net_a.learn_h, learn_o=self.net_a.learn_o)
        ks_o_b = self.net_b.propagate(spike_times_b, t, learn_h=self.net_b.learn_h, learn_o=self.net_b.learn_o)
        
        if self.topdown_enabled:
            if self.net_a.learn_o:
                for k_o_a in ks_o_a:
                    self.net_a.stdp(self.layer, self.net_a.layer_o, self.weights_td[0, k_o_a], t, k_o_a, self.topdown_stdp_mult)
            if self.net_b.learn_o:
                for k_o_b in ks_o_b:
                    self.net_b.stdp(self.layer, self.net_b.layer_o, self.weights_td[1, k_o_b], t, k_o_b, self.topdown_stdp_mult)
            self.weights_td = np.clip(self.weights_td, -self.log_c, self.w_max)
        
        # Update the excitation in the final layer if neurons in the previous layers fired
        for k_o_a in ks_o_a:
            self.layer.excitation[0, 0] += self.weights[0, :, k_o_a] + self.log_c
        for k_o_b in ks_o_b:
            self.layer.excitation[0, 0] += self.weights[1, :, k_o_b] + self.log_c
        if not self.softmax_probabilities: # Limit the membrane potential of neurons
            self.layer.excitation[0, 0] -= self.inhibitor.get_control(ks_o_a.size+ks_o_b.size)
            self.layer.excitation[0, 0] = np.clip(self.layer.excitation[0, 0], 0, -self.e_min_o)
        
        # Determine which hidden neurons fire and retrieve their indices
        ks, n_spikes = self.get_spike_indices(self.layer.excitation[0, 0], self.K, self.hertz_o)
        
        if not self.softmax_probabilities: # Adapt inhibition according to spiking activity
            self.inhibitor.update(t, t-self.t_last_spike, n_spikes>0)
        
        # Fire the selected neurons and propogate their signal to the next layer
        for k in ks:
            
            # Fire neuron <k>
            self.layer.fire(0, 0, k, t)
            self.n_spikes_since_reset[k] += 1
    
            if self.topdown_enabled: # Send feedback to the hierarchical output layer
                if not self.net_a.t_last_spike == t:
                    excitation = self.weights_td[0, :, k] + self.log_c
                    if not isinstance(self.td_curve, type(None)):
                        curve_index = min(self.n_spikes_since_reset[k]-1, len(self.td_curve)-1)
                        excitation = excitation * self.td_curve[curve_index]
                    inhibition = 0 if self.softmax_probabilities else self.net_a.inhibitor.get_control(1)
                    self.net_a.layer_o.excitation[0, 0] += excitation - inhibition
                    self.net_a.layer_o.excitation = np.clip(self.net_a.layer_o.excitation, 0, -self.net_a.e_min_o)
                if not self.net_b.t_last_spike == t:
                    excitation = self.weights_td[1, :, k] + self.log_c
                    if not isinstance(self.td_curve, type(None)):
                        curve_index = min(self.n_spikes_since_reset[k]-1, len(self.td_curve)-1)
                        excitation = excitation * self.td_curve[curve_index]
                    inhibition = 0 if self.softmax_probabilities else self.net_b.inhibitor.get_control(1)
                    self.net_b.layer_o.excitation[0, 0] += excitation - inhibition
                    self.net_b.layer_o.excitation = np.clip(self.net_b.layer_o.excitation, 0, -self.net_b.e_min_o)
                
            if self.learn: # Update the incoming connections weights of neuron <k>
                self.stdp(self.net_a.layer_o, self.layer, self.weights[0, k], t, k)
                self.stdp(self.net_b.layer_o, self.layer, self.weights[1, k], t, k)
                self.weights = np.clip(self.weights, -self.log_c, self.w_max)
        
        if n_spikes > 0: # Update information and inhibit the layer
            self.t_last_spike = t
            self.layer.inhibit() # ***
        
        return ks
    
    def reset(self):
        """ Reset the input and output layer. """
        self.net_a.reset()
        self.net_b.reset()
        self.layer.reset()
        self.t_last_spike = 0
        
    def train(self, data_handler, P):
        
        # Load the actual MNIST (train) pixel data and corresponding labels
        X_a, labels, _, _ = data_handler.load_mnist(tag_mnist=self.net_a.tag_mnist)
        X_b, _, _, _    = data_handler.load_mnist(tag_mnist=self.net_b.tag_mnist)
        
        time_start = time.time()
        # Iterate over the spike data in slices of size <data_handler.s_slice>
        for ith_slice in range(0, util.SHAPE_MNIST_TRAIN[0]//data_handler.s_slice):
            
            # Retrieve slice <ith_slice> of the spike data
            spike_data_a = data_handler.get_mnist_spikes(ith_slice, tag_mnist=self.net_a.tag_mnist, train=True)
            spike_data_b = data_handler.get_mnist_spikes(ith_slice, tag_mnist=self.net_b.tag_mnist, train=True)
            for index_im, (spike_times_a, spike_times_b) in enumerate(zip(spike_data_a, spike_data_b)):
                
                # Convert the flat <spike_times> to a tiled array
                spike_times_a = self.tile_input(spike_times_a, self.net_a.s_os, self.net_a.s_is)
                spike_times_b = self.tile_input(spike_times_b, self.net_b.s_os, self.net_b.s_is)
                
                # Determine the complete dataset index (rather than that of the slice)
                index_im_all = ith_slice*data_handler.s_slice+index_im
            
                # Propagate through the network according to the current timestep and given spike times
                for t in range(data_handler.ms):
                    self.propagate(spike_times_a, spike_times_b, t)
                self.reset() # Reset the network between images
                    
                # Print, plot, and save data according to the given parameters
                self.net_a.print_plot_save(data_handler, X_a, labels, index_im_all, util.SHAPE_MNIST_TRAIN[0], self.P_a, time_start)
                self.net_b.print_plot_save(data_handler, X_b, labels, index_im_all, util.SHAPE_MNIST_TRAIN[0], self.P_b, time_start)
                time_start = self.print_plot_save(data_handler, X_a, X_b, labels, index_im_all, util.SHAPE_MNIST_TRAIN[0], P, time_start)
                
                # Reset spike counters
                self.net_a.n_spikes_since_reset_h = np.zeros((self.net_a.s_os, self.net_a.s_os, self.net_a.K_h), dtype=np.uint16)
                self.net_a.n_spikes_since_reset_o = np.zeros(self.net_a.K_o, dtype=np.uint16)
                self.net_b.n_spikes_since_reset_h = np.zeros((self.net_b.s_os, self.net_b.s_os, self.net_b.K_h), dtype=np.uint16)
                self.net_b.n_spikes_since_reset_o = np.zeros(self.net_b.K_o, dtype=np.uint16)
                self.n_spikes_since_reset = np.zeros(self.K, dtype=np.uint16)
            
# =============================================================================
#                 if index_im_all >10 :
#                     return
# =============================================================================
    def test(self, data_handler, P):
        
        # Load the actual MNIST (test) pixel data and corresponding labels
        _, _, X_a, labels = data_handler.load_mnist(tag_mnist=self.net_a.tag_mnist)
        _, _, X_b, _    = data_handler.load_mnist(tag_mnist=self.net_b.tag_mnist)
        
        # The number of times each neuron spiked for each label
        neuron_label_counts = np.zeros((self.K, 10), dtype=np.uint32)
        
        # For each image the number of times each neuron spiked
        neuron_image_counts = np.zeros((util.SHAPE_MNIST_TEST[0], self.K))
        
        index_im_all = 0
        time_start = time.time()
        # Iterate over the spike data in slices of size <data_handler.s_slice>
        for ith_slice in range(0, util.SHAPE_MNIST_TEST[0]//data_handler.s_slice):
        
            # Retrieve slice <ith_slice> of the spike data
            spike_data_a = data_handler.get_mnist_spikes(ith_slice, tag_mnist=self.net_a.tag_mnist, train=False)
            spike_data_b = data_handler.get_mnist_spikes(ith_slice, tag_mnist=self.net_b.tag_mnist, train=False)
            for index_im, (spike_times_a, spike_times_b) in enumerate(zip(spike_data_a, spike_data_b)):
                
                # Convert the flat <spike_times> to a tiled array
                spike_times_a = self.tile_input(spike_times_a, self.net_a.s_os, self.net_a.s_is)
                spike_times_b = self.tile_input(spike_times_b, self.net_b.s_os, self.net_b.s_is)
                
                # Determine the complete dataset index (rather than that of the slice)
                index_im_all = ith_slice*data_handler.s_slice+index_im
                
                # Retrieve the label of the current image
                label = labels[index_im_all]
            
                # Propagate through the network according to the current timestep and given spike times
                for t in range(data_handler.ms):
                    self.propagate(spike_times_a, spike_times_b, t)
                self.reset() # Reset the network between images
                
                # Keep track of the results
                neuron_label_counts[:, label] += self.n_spikes_since_reset
                neuron_image_counts[index_im_all] = self.n_spikes_since_reset
                
                # Print, plot, and save data according to the given parameters
                self.net_a.print_plot_save(data_handler, X_a, labels, index_im_all, util.SHAPE_MNIST_TRAIN[0], self.P_a, time_start)
                self.net_b.print_plot_save(data_handler, X_b, labels, index_im_all, util.SHAPE_MNIST_TRAIN[0], self.P_b, time_start)
                time_start = self.print_plot_save(data_handler, X_a, X_b, labels, index_im_all, util.SHAPE_MNIST_TEST[0], P, time_start)
                
                # Evaluate results after every <evaluation_interval> images
                if index_im_all > 0 and index_im_all%P.evaluation_interval == 0:
                    print("\nEvaluating results after {} images:".format(index_im_all))
                    self.evaluate_results(data_handler, neuron_label_counts[:, :index_im_all+1],
                                          neuron_image_counts[:index_im_all+1], labels[:index_im_all+1])
                
                # Reset spike counter
                self.net_a.n_spikes_since_reset_h = np.zeros((self.net_a.s_os, self.net_a.s_os, self.net_a.K_h), dtype=np.uint16)
                self.net_a.n_spikes_since_reset_o = np.zeros(self.net_a.K_o, dtype=np.uint16)
                self.net_b.n_spikes_since_reset_h = np.zeros((self.net_b.s_os, self.net_b.s_os, self.net_b.K_h), dtype=np.uint16)
                self.net_b.n_spikes_since_reset_o = np.zeros(self.net_b.K_o, dtype=np.uint16)
                self.n_spikes_since_reset = np.zeros(self.K, dtype=np.uint16)
            
# =============================================================================
#             if index_im_all > 10:
#                 break
# =============================================================================
            
            gc.collect() # Clear memory of unused items # ***
        
        # Evaluate the results for the final time
        print("\nDone, final evaluation:")
        results_string, cm_classifications, cm_spike_uncertainty = self.evaluate_results(data_handler, 
                                                                         neuron_label_counts[:, :index_im_all+1],
                                                                         neuron_image_counts[:index_im_all+1], 
                                                                         labels[:index_im_all+1])
        
        if self.save_test_results:
            print("Saving test results ...")    
            os.mkdir(self.pd_test_results)
            with open(self.pd_test_results+'parameters.txt', 'w') as f:
                f.write(str(self.parameters))
            util.dump(self.parameters, self.pd_test_results+self.nf_test_parameters)
            np.save(self.pd_test_results+self.nf_test_weights_of    , self.weights)
            np.save(self.pd_test_results+self.nf_test_labels        , labels)
            np.save(self.pd_test_results+self.nf_neuron_label_counts, neuron_label_counts)
            np.save(self.pd_test_results+self.nf_neuron_image_counts, neuron_image_counts)
            #np.save(self.pd_test_results+self.nf_cm_classifications, cm_classifications)
            #np.save(self.pd_test_results+self.nf_cm_spike_uncertainty, cm_spike_uncertainty)
            print("Test results are saved!")
    
    def weights_to_pixels(self):
        
        """ Convert network weights to pixels. """
        
        pixels = np.zeros((self.K, 28, 28))        
        
        for i in range(2):
            pixels_out = self.net_a.out_to_pixels() if i == 0 else self.net_b.out_to_pixels()
            for k in range(self.K):
                
                # Only show weights within a given range of the highest weights ***
                show_range = 1
                indices = (-self.weights[i, k, :]).argsort()
                w_sorted = self.weights[i, k, :][indices]
                n_highest = w_sorted[w_sorted > w_sorted[0] - show_range].size
                indices = indices[:n_highest]
                highest_weights = self.weights[i, k, :][indices]
                pixel = 0
                for ind, w in zip(indices, highest_weights):
                    pixel += (self.log_c + w) * pixels_out[ind]
                pixel /= n_highest
                pixels[k] += pixel                
        
        # Obtain the largest weight
        max_value = np.max(np.abs(pixels))
        
        # Normalize the weights, then scale them to pixel values (0-255)
        weight_pixels = ((pixels+max_value)/(max_value*2) * 255).astype(np.uint8)
        
        return weight_pixels
    
    def print_plot_save(self, data_handler, X_a, X_b, labels, index_im_all, size_dataset, P, time_start):
        
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
                                index_im_all == util.SHAPE_MNIST_TRAIN[0]-1):
            if index_im_all > 0:
                print("\n\n\nProcessing {} images took {} seconds".format(
                    P.print_interval, int(time.time() - time_start)))
                time_start = time.time()
            print("\nFinal spikes at image {:5d}, displaying digit {}:\n{}\nTotal: {}".format(
                index_im_all, labels[index_im_all], 
                self.n_spikes_since_reset, np.sum(self.n_spikes_since_reset)))
        
        # Plot the networks weights if enabled
        if P.plot_weight_ims and (index_im_all%P.plot_interval == 0 or
                                  index_im_all == size_dataset-1):
            indices = [k for k in range(self.K)] # Determine which neurons to plot
            weight_pixels = self.weights_to_pixels()
            self.plot_weights(weight_pixels, self.n_spikes_since_reset, indices, 
                              index_im_all, [X_a[index_im_all], X_b[index_im_all]], labels[index_im_all])
        
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
