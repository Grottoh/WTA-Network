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
from network import Network, WTALayer
from network import Inhibitor
#from network InhibitorConstant

R_O = 1
C_O = 1

class NetworkHierarchical(Network):
    
    """ A hierarchical WTA network. """

    def __init__(self, P):
        
        super().__init__(P)
        
        # Network outer and inner square size
        self.s_os = 4 # The network has <s_os**2> hidden layers
        self.s_is = 7 # Each hidden layer receives input from <s_is**2> neurons of the input layer
        
        # Designates which MNIST to use
        self.tag_mnist = P.tag_mnist
        
        # The number of output neurons
        self.K_h = P.K_h
        self.K_o = P.K_o
        
        # True if the part of the network should learn (i.e. update the weights)
        self.learn_h = P.learn_h
        self.learn_o = P.learn_o
        
        # Create the input layer, which consists of multiple WTA circuits
        self.layer_i = np.empty((self.s_os, self.s_os), object)
        for r_o in range(self.s_os):
            for c_o in range(self.s_os):
                self.layer_i[r_o, c_o] = WTALayer(self.s_is, self.s_is, 2, P.ms)
        
        # Create the hidden layer
        self.layer_h = np.empty((self.s_os, self.s_os), object)
        for r_o in range(self.s_os):
            for c_o in range(self.s_os):
                self.layer_h[r_o, c_o] = WTALayer(1, 1, self.K_h, P.ms)
                
        # Create the output layer, which consists of a single WTA circuit
        self.layer_o = WTALayer(1, 1, self.K_o, P.ms)
        
        # Initialize the weights
        if not self.load_weights_ih(P.pf_pretrained_weights_ih): # If weights are not loaded ...
            self.weights_ih = self.initialize_weights((self.s_os, self.s_os, self.K_h, self.s_is, self.s_is, 2))
        if not self.load_weights_ho(P.pf_pretrained_weights_ho): # If weights are not loaded ...
            self.weights_ho = self.initialize_weights((self.K_o, self.s_os, self.s_os, self.K_h))
        
        if self.topdown_enabled:
            self.weights_oh = self.initialize_weights((self.K_h, self.s_os, self.s_os, self.K_o))
            self.weights_oh = self.weights_oh - self.log_c/1.2
        
        # Used to keep track of the number of spikes per neurons since the new input image
        self.n_spikes_since_reset_h = np.zeros((self.s_os, self.s_os, self.K_h), dtype=np.uint16)
        self.n_spikes_since_reset_o = np.zeros(self.K_o, dtype=np.uint16)
        
        inh_start_h = self.log_c if is_none(P.inh_start_h) else P.inh_start_h
        self.inhibitors_h = np.empty((self.s_os, self.s_os), object)
        for r_o in range(self.s_os):
            for c_o in range(self.s_os):
                self.inhibitors_h[r_o, c_o]  = Inhibitor(P.hertz_o, inh_start_h, P.inh_change_h, P.inh_ratio_h, P.ms)
                
        self.ts_last_spike = np.zeros((self.s_os, self.s_os), dtype=np.int16)
        self.t_last_spike = 0
        
        self.e_min_h = -self.log_c * 784 * self.hertz_o * 0.2 * 0.0001 if is_none(P.e_min_h) else P.e_min_h # ***
        self.e_min_o = -self.log_c * 784 * self.hertz_o * 0.2 * 0.0001 if is_none(P.e_min_o) else P.e_min_o
        
        self.hertz_h = P.hertz_h
        
# =============================================================================
#         print(self.e_min_h)
#         print(self.e_min_o)
#         sys.exit()
# =============================================================================
    
    def load_weights_ih(self, pf_pretrained_weights_ih):
        weights_ih = self.load_weights(pf_pretrained_weights_ih)
        if str(weights_ih) == "nonexistent":
            print("Could not load input-hidden weights, terminating ...")
            sys.exit()
        if str(weights_ih) != "nopath":
            self.weights_ih = np.zeros((self.s_os, self.s_os, self.K_h, self.s_is, self.s_is, 2))
            for r_o in range(self.s_os):
                for c_o in range(self.s_os):
                    self.weights_ih[r_o, c_o] = weights_ih[r_o*self.s_os+c_o]
            print(f"Succesfully loaded  input-hidden weights from <{pf_pretrained_weights_ih}>.")
            return True
        return False
    
    def load_weights_ho(self, pf_pretrained_weights_ho):
        weights_ho = self.load_weights(pf_pretrained_weights_ho)
        if str(weights_ho) == "nonexistent":
            print("Could not load hidden-output weights, terminating ...")
            sys.exit()
        if str(weights_ho) != "nopath":
            self.weights_ho = weights_ho[0]
            print(f"Succesfully loaded hidden-output weights from <{pf_pretrained_weights_ho}>.")
            return True
        return False
    
    def propagate_input(self, spike_times, t):
        
        """ Fire input neurons according to the current time and the given spike times. """
        
        for r_o in range(self.s_os):
            for c_o in range(self.s_os):
                
                # Fire input neurons according to the given spike times
                self.layer_i[r_o, c_o].fire_multiple(spike_times[r_o, c_o, :, :, :, t], t)
                
                # Compute the excitation and inhibition and update membrane potentials accordingly
                excitation = ((self.weights_ih[r_o, c_o]+self.log_c) * spike_times[r_o, c_o, :, :, :, t]).sum((1, 2, 3))
                inhibition = 0 if self.softmax_probabilities else self.inhibitors_h[r_o, c_o].get_control(spike_times[r_o, c_o, :, :, :, t].sum())
                self.layer_h[r_o, c_o].excitation[0, 0] += excitation - inhibition
                
                if not self.softmax_probabilities: # Limit the membrane potential of neurons
                    self.layer_h[r_o, c_o].excitation[0, 0] = np.clip(self.layer_h[r_o, c_o].excitation[0, 0], 0, -self.e_min_h)
    
    
    def propagate_hidden(self, t, r_o, c_o, learn=True):
        
        # Determine which hidden neurons fire and retrieve their indices
        ks_h, n_spikes_h = self.get_spike_indices(self.layer_h[r_o, c_o].excitation[0,0], self.K_h, self.hertz_h)
        
        if not self.softmax_probabilities: # Adapt inhibition according to spiking activity
            self.inhibitors_h[r_o, c_o].update(t, t-self.ts_last_spike[r_o, c_o], n_spikes_h>0)
        
        # Fire the selected neurons and propogate their signal to the next layer
        for k_h in ks_h:
            
            # Fire neuron <k_h>
            self.layer_h[r_o, c_o].fire(0, 0, k_h, t)
            self.n_spikes_since_reset_h[r_o, c_o, k_h] += 1
            
            # Propagate to the next layer
            excitation = self.weights_ho[:, r_o, c_o, k_h] + self.log_c
            self.layer_o.excitation[0, 0] += excitation
            
            if learn: # Update the incoming connections weights of neuron <k_h>
                self.stdp(self.layer_i[r_o, c_o], self.layer_h[r_o, c_o], self.weights_ih[r_o, c_o, k_h], t, k_h)
                if self.topdown_enabled:
                    self.stdp(self.layer_o, self.layer_h[r_o, c_o], self.weights_oh[k_h, r_o, c_o], t, k_h, self.topdown_stdp_mult)
        
        if n_spikes_h > 0: # Update information and inhibit the layer
# =============================================================================
#             if r_o == 1 and c_o == 1:
#                 self.ls_spike_times_h.append(f"{t:3d}")
# =============================================================================
            self.ts_last_spike[r_o, c_o] = t
            self.layer_h[r_o, c_o].inhibit() # ***
        
        return ks_h, n_spikes_h
    
    def propagate_output(self, t, learn=True):
        
        """ 
        Fire output neurons according to their excitation, and update their
        incoming connection weights.
        """

        # Determine which output neurons fire and retrieve their indices
        ks_o, n_spikes_o = self.get_spike_indices(self.layer_o.excitation[0, 0], self.K_o, self.hertz_o)
        
        if not self.softmax_probabilities: # Adapt inhibition according to spiking activity
            self.inhibitor.update(t, t-self.t_last_spike, n_spikes_o>0)
        
        # Fire the selected neurons and propogate their signal to the next layer
        for k_o in ks_o:
            
            # Fire neuron <k>
            self.layer_o.fire(0, 0, k_o, t)
            self.n_spikes_since_reset_o[k_o] += 1
            
            if self.topdown_enabled:
                excitation = self.weights_oh[..., k_o] + self.log_c
                if not isinstance(self.td_curve, type(None)):
                    curve_index = min(self.n_spikes_since_reset_o[k_o]-1, len(self.td_curve)-1)
                    excitation = excitation * self.td_curve[curve_index]
                for r_o in range(self.s_os):
                    for c_o in range(self.s_os):
                        if not self.ts_last_spike[r_o, c_o] == t:
                            inhibition = 0 if self.softmax_probabilities else self.inhibitors_h[r_o, c_o].get_control(1)
                            self.layer_h[r_o, c_o].excitation[0, 0] += excitation[:, r_o, c_o] - inhibition
            
            if learn: # Update the incoming connections weights of neuron <k_o> of each hidden circuit
                for r_o in range(self.s_os):
                    for c_o in range(self.s_os):
                        self.stdp(self.layer_h[r_o, c_o], self.layer_o, self.weights_ho[k_o, r_o, c_o], t, k_o)
        
        if n_spikes_o > 0: # Update information and inhibit the layer
            self.ls_spike_times.append(f"{t:3d}")
            self.t_last_spike = t
            self.layer_o.inhibit() # ***
        
        return ks_o, n_spikes_o
    
    def propagate(self, spike_times, t, learn_h=True, learn_o=True):
        """ 
        Propogate activity through the network for a single timestep according
        to the given spike times.
        """
        
        # Propogate through the input layer
        self.propagate_input(spike_times, t)
        
        # Propogate through the hidden layer
        n_spikes_h_total = 0
        for r_o in range(self.s_os):
            for c_o in range(self.s_os):
                ks_h, n_spikes_h = self.propagate_hidden(t, r_o, c_o, learn=self.learn_h)
                n_spikes_h_total += n_spikes_h
                
        self.weights_ih = np.clip(self.weights_ih, -self.log_c, self.w_max)
        if self.topdown_enabled:
            self.weights_oh = np.clip(self.weights_oh, -self.log_c, self.w_max)
        
        if not self.softmax_probabilities: # Include inhibition and limit the membrane potential of neurons
            inhibition = self.inhibitor.get_control(n_spikes_h_total)
            self.layer_o.excitation = np.clip(self.layer_o.excitation-inhibition, 0, -self.e_min_o)
        
        # Propagate through the output layer
        ks_o, n_spikes_o = self.propagate_output(t, learn=self.learn_o)
        self.weights_ho = np.clip(self.weights_ho, -self.log_c, self.w_max)
        
        if not self.softmax_probabilities and self.topdown_enabled: 
            for r_o in range(self.s_os):
                for c_o in range(self.s_os):
                    if not self.softmax_probabilities: # Limit the membrane potential of neurons
                        self.layer_h[r_o, c_o].excitation[0, 0] = np.clip(self.layer_h[r_o, c_o].excitation[0, 0], 0, -self.e_min_h)
                
        return ks_o
        
    def reset(self):
        """ Reset the input and output layer. """
        for circuit in self.layer_i.flatten():
            circuit.reset()
        for circuit in self.layer_h.flatten():
            circuit.reset()
        self.layer_o.reset()
        self.ts_last_spike = np.zeros((self.s_os, self.s_os), dtype=np.int16)
        self.t_last_spike = 0
        
# =============================================================================
#         print("\nH:",self.ls_spike_times_h)
#         print("S:",self.ls_spike_times)
#         self.ls_spike_times_h = []
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
                
                # Convert the flat <spike_times> to a tiled array
                spike_times = self.tile_input(spike_times, self.s_os, self.s_is)
                
                # Determine the complete dataset index (rather than that of the slice)
                index_im_all = ith_slice*data_handler.s_slice+index_im
                self.index_im_all = index_im_all # $$$
                self.print_interval = P.print_interval # $$$
            
                # Propogate through the network according to the current timestep and given spike times
                for t in range(data_handler.ms):
                    self.propagate(spike_times, t, learn_h=True, learn_o=True)
                self.reset() # Reset the network between images
                    
                # Print, plot, and save data according to the given parameters
                time_start = self.print_plot_save(data_handler, X, labels, index_im_all, util.SHAPE_MNIST_TRAIN[0], P, time_start)
                    
                # Reset spike counters
                self.n_spikes_since_reset_h = np.zeros((self.s_os, self.s_os, self.K_h), dtype=np.uint16)
                self.n_spikes_since_reset_o = np.zeros(self.K_o, dtype=np.uint16)
                
# =============================================================================
#                 if index_im_all > 5500: ###############@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                     print("EXITING AT 5500 images!!!")
#                     return
# =============================================================================
        
    def test(self, data_handler, P):
        
        # Load the actual MNIST (test) pixel data and corresponding labels
        _, _, X, labels = data_handler.load_mnist(tag_mnist=self.tag_mnist)
        
        # The number of times each neuron spiked for each label
        neuron_label_counts = np.zeros((self.K_o, 10), dtype=np.uint32)
        
        # For each image the number of times each neuron spiked
        neuron_image_counts = np.zeros((util.SHAPE_MNIST_TEST[0], self.K_o))
        
        index_im_all = 0
        time_start = time.time()
        # Iterate over the spike data in slices of size <data_handler.s_slice>
        for ith_slice in range(0, util.SHAPE_MNIST_TEST[0]//data_handler.s_slice):
        
            # Retrieve slice <ith_slice> of the spike data
            spike_data = data_handler.get_mnist_spikes(ith_slice, tag_mnist=self.tag_mnist, train=False)
            for index_im, spike_times in enumerate(spike_data):
                
                # Convert the flat <spike_times> to a tiled array
                spike_times = self.tile_input(spike_times, self.s_os, self.s_is)
                
                # Determine the complete dataset index (rather than that of the slice)
                index_im_all = ith_slice*data_handler.s_slice+index_im
                self.index_im_all = index_im_all # $$$
                self.print_interval = P.print_interval # $$$
                
                # Retrieve the label of the current image
                label = labels[index_im_all]
            
                # Propogate through the network according to the current timestep and given spike times
                for t in range(data_handler.ms):
                    self.propagate(spike_times, t, learn_h=False, learn_o=False)
                self.reset() # Reset the network between images
                
                # Keep track of the results
                neuron_label_counts[:, label] += self.n_spikes_since_reset_o
                neuron_image_counts[index_im_all] = self.n_spikes_since_reset_o
                
                # Print, plot, and save data according to the given parameters
                time_start = self.print_plot_save(data_handler, X, labels, index_im_all, util.SHAPE_MNIST_TEST[0], P, time_start)
                
                # Evaluate results after every <evaluation_interval> images
                if index_im_all > 0 and index_im_all%P.evaluation_interval == 0:
                    print("\nEvaluating results after {} images:".format(index_im_all))
                    self.evaluate_results(data_handler, neuron_label_counts[:, :index_im_all+1],
                                          neuron_image_counts[:index_im_all+1], labels[:index_im_all+1])
                                     
                # Reset spike counters
                self.n_spikes_since_reset_h = np.zeros((self.s_os, self.s_os, self.K_h), dtype=np.uint16)
                self.n_spikes_since_reset_o = np.zeros(self.K_o, dtype=np.uint16)
                
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
            np.save(self.pd_test_results+self.nf_test_weights_ih    , self.weights_ih)
            np.save(self.pd_test_results+self.nf_test_weights_ho    , self.weights_ho)
            np.save(self.pd_test_results+self.nf_test_labels        , labels)
            np.save(self.pd_test_results+self.nf_neuron_label_counts, neuron_label_counts)
            np.save(self.pd_test_results+self.nf_neuron_image_counts, neuron_image_counts)
            #np.save(self.pd_test_results+self.nf_cm_classifications, cm_classifications)
            #np.save(self.pd_test_results+self.nf_cm_spike_uncertainty, cm_spike_uncertainty)
            print("Test results are saved!")
    
    def hidden_to_pixels(self, index_circuit):
        
        """ Convert network weights to pixels. """
        
        # Subtract black weights from white weights
        weights_black = self.weights_ih[index_circuit[0], index_circuit[1], :, :, :, 0] + self.log_c
        weights_white = self.weights_ih[index_circuit[0], index_circuit[1], :, :, :, 1] + self.log_c
        weights = weights_white - weights_black
        
        # Normalize the weights, then scale them to pixel values (0-255)
        max_value = np.max(np.abs(weights))
        weight_pixels = ((weights+max_value)/(max_value*2) * 255).astype(np.uint8)
        
        return weight_pixels
        
    def out_to_pixels(self):
        
        """ Convert network weights to pixels. """
        
        pixels = np.zeros((self.K_o, 28, 28))
        for r_o in range(self.s_os):
            for c_o in range(self.s_os):
                pixels_hidden = self.hidden_to_pixels((r_o, c_o))
                for k_o in range(self.K_o):
                    # Only show weights within a given range of the highest weights ***
                    show_range = 1
                    indices = (-self.weights_ho[k_o, r_o, c_o, :]).argsort()
                    w_sorted = self.weights_ho[k_o, r_o, c_o, :][indices]
                    n_highest = w_sorted[w_sorted > w_sorted[0] - show_range].size
                    indices = indices[:n_highest]
                    highest_weights = self.weights_ho[k_o, r_o, c_o, :][indices]
                    pixel = 0
                    for ind, w in zip(indices, highest_weights):
                        pixel += (self.log_c + w) * pixels_hidden[ind]
                    pixel /= n_highest
                    pixels[k_o, r_o*7:r_o*7+7, c_o*7:c_o*7+7] += pixel
        
        # Normalize the weights, then scale them to pixel values (0-255)
        max_value = np.max(np.abs(pixels))
        weight_pixels = ((pixels+max_value)/(max_value*2) * 255).astype(np.uint8)
        
        return weight_pixels
    
    def print_plot_save(self, data_handler, X, labels, index_im_all, size_dataset, P, time_start):
        
        # &&&
        self.ls_n_spikes.append(self.n_spikes_since_reset_h[R_O, C_O].sum())
        if len(self.ls_n_spikes) > P.print_interval:
            self.ls_n_spikes.pop(0)
            if index_im_all % P.print_interval== 0 and index_im_all > 0:
                print(f"avg spikes over last {P.print_interval}:", np.mean(self.ls_n_spikes))
                print(f"std spikes over last {P.print_interval}:", np.std(self.ls_n_spikes))
                print(f"Min spikes: {np.min(self.ls_n_spikes)}, Max spikes: {np.max(self.ls_n_spikes)}")
        
        # Print some results if enabled
        if index_im_all%P.print_interval == 0 or index_im_all == size_dataset-1:
            if index_im_all > 0 and (P.print_results_h or P.print_results_o):
                print("\n\n\nProcessing {} images took {} seconds".format(P.print_interval, int(time.time() - time_start)))
                time_start = time.time()
            
            if P.print_results_h:
                print("\nHidden spikes at image {:5d}, displaying digit {}:\n{}\nTotal per circuit:\n{}\nTotal overall: {}, Average over circuits: {:.2f}, Standard Deviation: {:.2f}".format(
                    index_im_all, labels[index_im_all], 
                    "<commented>",#self.n_spikes_since_reset_h, 
                    self.n_spikes_since_reset_h.sum(2),
                    self.n_spikes_since_reset_h.sum(), np.mean(self.n_spikes_since_reset_h.sum(2)), np.std(self.n_spikes_since_reset_h.sum(2))))
                
# =============================================================================
#             if P.print_results_h:
#                 print("\nTotal overall: {}, Average over circuits: {:.2f}, Standard Deviation: {:.2f}".format(
#                     self.n_spikes_since_reset_h.sum(), np.mean(self.n_spikes_since_reset_h.sum(2)), np.std(self.n_spikes_since_reset_h.sum(2))))
# =============================================================================

            if P.print_results_o:
                print("\nOutput spikes at image {:5d}, displaying digit {}:\n{}\nTotal: {}".format(
                    index_im_all, labels[index_im_all], 
                    self.n_spikes_since_reset_o, np.sum(self.n_spikes_since_reset_o)))
        
        # Plot the hidden weights if enabled
        if P.plot_weight_ims_h and (index_im_all%P.plot_interval == 0 or
                                    index_im_all == size_dataset-1):
            r_o, c_o = (R_O, C_O) # Determine which circuit to plot
            indices = [k for k in range(self.K_h)] # Determine which neurons to plot
            weight_pixels = self.hidden_to_pixels((r_o, c_o))
            input_segment = X[index_im_all, r_o*self.s_is:r_o*self.s_is+self.s_is, c_o*self.s_is:c_o*self.s_is+self.s_is]
            
            self.plot_weights(weight_pixels, self.n_spikes_since_reset_h[r_o, c_o], indices, 
                              index_im_all, [X[index_im_all], input_segment], labels[index_im_all])
            
            r_o, c_o = (3, 3) # Determine which circuit to plot
            indices = [k for k in range(self.K_h)] # Determine which neurons to plot
            weight_pixels = self.hidden_to_pixels((r_o, c_o))
            input_segment = X[index_im_all, r_o*self.s_is:r_o*self.s_is+self.s_is, c_o*self.s_is:c_o*self.s_is+self.s_is]
            self.plot_weights(weight_pixels, self.n_spikes_since_reset_h[r_o, c_o], indices, 
                              index_im_all, [X[index_im_all], input_segment], labels[index_im_all])
                                  
        # Plot the output weights if enabled
        if P.plot_weight_ims_o and (index_im_all%P.plot_interval == 0 or
                                    index_im_all == size_dataset-1):
              indices = [k for k in range(self.K_o)] # Determine which neurons to plot
              weight_pixels = self.out_to_pixels()
              self.plot_weights(weight_pixels, self.n_spikes_since_reset_o, indices, 
                                index_im_all, [X[index_im_all]], labels[index_im_all])
        
        # Save hidden network weight images if enabled
        if P.save_weight_ims_h and (index_im_all%P.save_weight_ims_interval == 0 or
                                    index_im_all == size_dataset-1):
            for r_o in range(self.s_os):
                for c_o in range(self.s_os):
                    for k_h, p in enumerate(self.hidden_to_pixels((r_o, c_o))):
                        pf_weight_im_h = P.pd_weight_ims_h + P.nft_weight_ims_h.format(index_im_all, r_o, c_o, k_h)
                        save_img(p, pf_weight_im_h, normalized=False)
        
        # Save network weight images if enabled
        if P.save_weight_ims_o and (index_im_all%P.save_weight_ims_interval == 0 or
                                    index_im_all == size_dataset-1):
            for k_o, p in enumerate(self.out_to_pixels()):
                pf_weight_im_o = P.pd_weight_ims_o + P.nft_weight_ims_o.format(index_im_all, k_o)
                save_img(p, pf_weight_im_o, normalized=False)
        
        # Save hidden weights if enabled
        if P.save_weights_h and (index_im_all > 0 and index_im_all%P.save_weights_interval == 0 or
                                 index_im_all == size_dataset-1):
            
            # Move pas weights to a separate directory
            if len(os.listdir(P.pd_weights_h)) > 0: # If there are already stored weights
                if len(os.listdir(P.pd_weights_h)) == self.s_os**2: # Create a directory for past weights
                    os.mkdir(P.pd_weights_h+self.nd_weights_past)
                for nf in os.listdir(P.pd_weights_h): # Move past weights to separate directory
                    if os.path.isfile(P.pd_weights_h+nf): # Only move (weight) files
                        shutil.move(P.pd_weights_h+nf, P.pd_weights_h+self.nd_weights_past+nf)
        
            # Save the weights of the input-hidden connections
            for r_o in range(self.s_os):
                for c_o in range(self.s_os):
                    pf_weights_h = P.pd_weights_h + P.nft_weights_h.format(index_im_all, r_o, c_o)
                    np.save(pf_weights_h, self.weights_ih[r_o, c_o])
            
        # Save output weights if enabled
        if P.save_weights_o and (index_im_all > 0 and index_im_all%P.save_weights_interval == 0 or
                                 index_im_all == size_dataset-1):
            
            # Move pas weights to a separate directory
            if len(os.listdir(P.pd_weights_o)) > 0: # If there are already stored weights
                if len(os.listdir(P.pd_weights_o)) == 1: # Create a directory for past weights
                    os.mkdir(P.pd_weights_o+self.nd_weights_past)
                for nf in os.listdir(P.pd_weights_o): # Move past weights to separate directory
                    if os.path.isfile(P.pd_weights_o+nf): # Only move (weight) files
                        shutil.move(P.pd_weights_o+nf, P.pd_weights_o+self.nd_weights_past+nf)
            
            # Save the weights of the hidden-output connections
            pf_weights_o = P.pd_weights_o + P.nft_weights_o.format(index_im_all)
            np.save(pf_weights_o, self.weights_ho)
        
        return time_start
    
    
    def load_test_results(run_id, pdt_test_results="[01LF] results/run-hierarchical_{}/test_results/", nf_test_parameters="test_parameters.pkl"):
        pd_test_results = pdt_test_results.format(run_id)
        test_parameters = util.load(pd_test_results+nf_test_parameters)
        weights_ho = np.load(pd_test_results+test_parameters.nf_test_weights_ho)
        weights_ih = np.load(pd_test_results+test_parameters.nf_test_weights_ih)
        test_labels = np.load(pd_test_results+test_parameters.nf_test_labels)
        neuron_label_counts = np.load(pd_test_results+test_parameters.nf_neuron_label_counts)
        neuron_image_counts = np.load(pd_test_results+test_parameters.nf_neuron_image_counts)
        return test_parameters, weights_ho, weights_ih, test_labels, neuron_label_counts, neuron_image_counts