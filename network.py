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
from util import save_img, squeeze_all, get_sinewave, get_firing_probability, is_none, normalize
import gc

class Inhibitor():
    
    def __init__(self, hertz_o, inh_start, inh_change, inh_ratio, ms):
        
        self.inh_strength = inh_start
        self.target_interval = int(np.ceil(1000/hertz_o))
        self.prev_errors_u = []
        print(f"   inh_strength={self.inh_strength}")
        print(f"target_interval={self.target_interval}")
        print()
        
        
        self.inh_change = inh_change
        self.inh_ratio = inh_ratio
        
        self.accumulated_error = self.inh_strength/self.inh_change
        self.error_prev = 0
        
        self.mse_accumulated= 0
    
    def get_control(self, n_spikes_in):
        return self.inh_strength*n_spikes_in
        
    def update(self, t, t_since_last_spike, fired):
        
        """ 
        Adapt the inhibition signal according to spiking activity and the 
        desired firing rate of neurons. 
        """
        
        error = self.target_interval - t_since_last_spike
        if error < 0 or (error > 0 and fired):
            if error < 0:
                self.accumulated_error += error / self.inh_ratio
            else:
                self.accumulated_error += error
            self.inh_strength = max(0, self.inh_change*self.accumulated_error)
            self.mse_accumulated += error**2
        if random.random() < 0.00000: # ***
            print(f"t={t}, error={error:3d}, accumulated_error={self.accumulated_error:8.2f},"+
                  f" error_d={error-self.error_prev:3d}, inh_strength={self.inh_strength:8.2f}"+
                  f" --- mse_accumulated={self.mse_accumulated}")
        self.error_prev = error

class WTALayer():
    
    def __init__(self, n_rows, n_cols, K, ms):
        
        self.n_rows = n_rows # The number of rows of WTA circuits
        self.n_cols = n_cols # The number of columns of WTA circuits
        self.K = K           # The number of neurons per WTA circuit
        
        # The excitation of each neuron
        self.excitation = np.zeros((self.n_rows, self.n_cols, self.K), dtype=np.float32)
        
        # Track the last time each neuron spiked (for simple STDP)
        self.t_last_spike = np.full((self.n_rows, self.n_cols, self.K), -99999, dtype=np.int32)
        
        # Track the time at which each neuron spiked (for complex STDP)
        self.ms = ms # The amount of miliseconds the stimulus is presented
        self.t_spikes_pre = np.zeros((self.n_rows, self.n_cols, self.K, self.ms))
        
        # Track the total amount of times each neuron spiked
        self.n_spikes_total = np.zeros((self.n_rows, self.n_cols, self.K), dtype=np.uint32)
    
    def fire(self, r, c, k, t):
        """ Fire a single neuron, reseting the excitation of said neuron, and updating certain information. """
        self.excitation[r, c, k] = 0
        self.t_last_spike[r, c, k] = t
        self.t_spikes_pre[r, c, k, t] = 1
        self.n_spikes_total[r, c, k] += 1
    
    def fire_multiple(self, indices, t):
        """ Fire multiple neurons, reseting the excitation of said neurons, and updating certain information. """
        self.excitation[indices] = 0
        self.t_last_spike[indices] = t
        self.t_spikes_pre[indices, t] = 1
        self.n_spikes_total[indices] += 1
    
    def inhibit(self):
        """ Set the excitation of all neurons to zero. """
        self.excitation = np.zeros((self.n_rows, self.n_cols, self.K), dtype=np.float32)
    
    def reset(self):
        """ Reset certain aspects of the network. """
        self.excitation = np.zeros((self.n_rows, self.n_cols, self.K), dtype=np.float32)
        self.t_last_spike = np.full((self.n_rows, self.n_cols, self.K), -99999, dtype=np.int32)
        self.t_spikes_pre = np.zeros((self.n_rows, self.n_cols, self.K, self.ms))

class Network():
    
    """ A generic network containing some common network features. """
    
    def __init__(self, P):

        self.parameters = P
        
        # Extract spike data dimensions (batch, row, column)
        self.dim_row, self.dim_col = util.SHAPE_MNIST_TRAIN[1:]
        
        # How often neurons should approximately fire
        self.hertz_o = P.hertz_o # The frequency with which output neurons should spike
        self.ms = P.ms # The number of miliseconds an image is presented
        
        # STDP parameters
        self.stdp_simple = P.stdp_simple
        if self.stdp_simple:
            self.sigma = P.sigma # The time delay during which pre-synaptic spikes are remembered (for simple STDP)
        else: # Determine the strength of the STDP update as a function of time difference between pre- and post-synaptic spike
            t_diff = np.arange(P.ms)
            self.stdp_curve = 1/(P.t_f - P.t_s) * (np.exp(-t_diff/P.t_f) - np.exp(-t_diff/P.t_s)) * np.heaviside(t_diff, 0)
        self.eta_root_decay = P.eta_root_decay # Lower <eta> to <1 / n_spikes_k^eta_root_decay>
        
        # Bounds on weights and weight changes
        self.w_min_exp = P.w_min_exp # The exponent of the minimal value the weights
        self.w_max = P.w_max # The maximum value of the weights
        self.log_c = -np.log(P.w_min_exp) # A linear shift that brings weights to the range [0, -np.log(w_min_exp)]
        self.delta_w_max = self.log_c/10 if is_none(P.delta_w_max) else P.delta_w_max # The maximum (absolute) value <delta_w> may take
        
        # The name of the directory where past weights are stored when new ones are saved
        self.nd_weights_past = P.nd_weights_past
        
        self.always_fire = P.always_fire
        
        self.rng = np.random.default_rng(P.seed)
        
        inh_start_o = self.log_c if is_none(P.inh_start_o) else P.inh_start_o
        self.inhibitor = Inhibitor(P.hertz_o, inh_start_o, P.inh_change_o, P.inh_ratio_o, P.ms)
        
        
        self.eta_mult = P.eta_mult
        
        self.t_last_spike = 0
        
        # ***
        self.softmax_probabilities = P.softmax_probabilities
        if not self.softmax_probabilities:
            self.allow_concurrent_spikes = P.allow_concurrent_spikes
                
        self.ls_n_spikes = [] # &&&
        self.ls_spike_times = []
        self.ls_spike_times_h = []
    
        self.topdown_enabled = P.topdown_enabled
        self.topdown_stdp_mult = P.topdown_stdp_mult
        if isinstance(P.params_td_curve, type(None)):
            self.td_curve = None
        else:
            self.td_curve = util.get_td_curve(*P.params_td_curve, plot=True)
        
        self.save_test_results = P.save_test_results
        if self.save_test_results:
            self.pd_test_results = P.pd_test_results
            self.nf_test_parameters = P.nf_test_parameters
            self.nf_test_weights_io = P.nf_test_weights_io
            self.nf_test_weights_ih = P.nf_test_weights_ih
            self.nf_test_weights_ho = P.nf_test_weights_ho
            self.nf_test_weights_of = P.nf_test_weights_of
            self.nf_test_labels = P.nf_test_labels
            self.nf_neuron_label_counts = P.nf_neuron_label_counts
            self.nf_neuron_image_counts = P.nf_neuron_image_counts
    
    def load_weights(self, pf_pretrained_weights):
        if pf_pretrained_weights != None and pf_pretrained_weights != "None": # If the path is not <None> ...
            if os.path.exists(pf_pretrained_weights): # If the path exists ...
                if os.path.isdir(pf_pretrained_weights): # If the path leads to a directory ...
                    nfs = os.listdir(pf_pretrained_weights)
                    nfs.sort()
                    weights = [np.load(pf_pretrained_weights+nf) for nf in nfs if os.path.isfile(pf_pretrained_weights+nf)]
                else: # If the path leads to a file ...
                    weights = np.load(pf_pretrained_weights)
                return weights
            else:
                print("File <{}> does not exist, could not load weights, returning ...".format(pf_pretrained_weights))
                return "nonexistent"
        else:
            print("No path was provided to load weights, leaving them unchanged.")
            return "nopath"
        
    def tile_input(self, spike_times, s_os, s_is):
        """ Convert the flat <spike_times> to a tiled array. """
        tiled_input = np.empty((s_os, s_os, s_is, s_is, 2, spike_times.shape[-1]), dtype=np.bool8)
        for r in range(s_os):
            for c in range(s_os):
                tiled_input[r, c] = spike_times[r*s_is:r*s_is+s_is, c*s_is:c*s_is+s_is] 
        return tiled_input
    
    def initialize_weights(self, dims):
        """ 
        Initialize weights at high values, such that they do not become
        overshadowed by trained weights. """
        return -(self.rng.random(dims) * self.log_c)/200
    
    def stdp(self, layer_a, layer_b, weights, t, k, stdp_mult=1):
        
        """ Update weights according to a STDP learning rule. """
        
        eta = self.eta_mult / np.power(layer_b.n_spikes_total[0, 0, k], self.eta_root_decay)
        
        if self.stdp_simple: # Apply the simple STDP rule
        
            # Determine which pre-synaptic neurons spiked within <sigma> timesteps ago
            potentiation = t - squeeze_all(layer_a.t_last_spike) < self.sigma
            
            # Determine the weight update
            delta_w = np.clip((potentiation*np.exp(-weights) - 1*stdp_mult),
                              -self.delta_w_max/eta, self.delta_w_max/eta)
            
            # Apply the weight update (multiplied with the learning rate)
            weights += eta * delta_w
            # *** CLIPPING A REFERENCE DOES NOT WORK ***
            
        else: # Apply the complex STDP rule
        
            # Determine the strength of the STDP update as a function of time difference between pre- and post-synaptic spike
            ys = squeeze_all((layer_a.t_spikes_pre * self.stdp_curve).sum(-1))
            
            # Determine the weight update and apply it (multiplied with the learning rate)
            delta_w = np.clip((ys*np.exp(-(weights)) - 1*stdp_mult), -self.delta_w_max/eta, self.delta_w_max/eta)
            weights += eta * delta_w
    
    def get_spike_indices(self, excitation, K, hz):
        
        """ Determine which neurons fire and retrieve their indices. """
        
        if self.softmax_probabilities: # Use global information to compute softmax probabilities based on membrane potentials
        
            if self.rng.random() < hz/1000: # Fire a single neuron
                p = np.exp(excitation-np.max(excitation))/np.exp(excitation-np.max(excitation)).sum()
                ks = np.array([np.random.choice(np.arange(K), p=p)], dtype=np.uint16)
            else: # Do not fire any neuron
                ks = np.array([], dtype=np.uint16)
               
        else: # Use a biologically plausible link between membrane potential and firing probabilities
 
             # Determine the probability of each neuron firing
             p = np.exp(excitation+self.e_min_o)
             p[p>1] = 1
             
             # Fire neurons according to their spiking probabilities
             spikes = self.rng.random(K) < p
             ks = np.where(spikes)[0].astype(np.uint16)
             if not self.allow_concurrent_spikes: # Allow but a single neuron to spike at a given timestep
                 ks = np.array([], dtype=np.uint16) if ks.size == 0 else np.array([np.random.choice(ks)], dtype=np.uint16)
                
        n_spikes = ks.size # Keep track of the number of neurons that fired
        return ks, n_spikes
    
    def propagate_layer(self, layer_a, layer_b, layer_c, weights_ab, weights_bc,
                        n_spikes_since_reset, K, t, hz, learn=True):
        
        """ 
        Fire output neurons according to their excitation, update their
        incoming connection weights, and update outgoing connected neuron 
        excitation.
        """
        
        # Determine which neurons fire and retrieve their indices
        ks, n_spikes = self.get_spike_indices(layer_b.excitation[0, 0], K, hz)
        
        if not self.softmax_probabilities: # Adapt inhibition according to spiking activity
            self.inhibitor.update(t, t-self.t_last_spike, n_spikes>0)
        
        # Fire the selected neurons and propogate their signal to the next layer
        for k in ks:
            
            
            # Fire neuron <k>
            layer_b.fire(0, 0, k, t)
            n_spikes_since_reset[k] += 1
            
            if not isinstance(layer_c, type(None)): # Propagate to the next layer
                layer_c.excitation += (weights_bc[..., k] + self.log_c)
                layer_c.excitation = np.clip(layer_c.excitation, 0, -self.e_min_o) # *** Clipping like this probably does not work with a reference @@@
            
            if learn: # Update the incoming connection weights of neuron <k>
                self.stdp(layer_a, layer_b, weights_ab[k], t, k)
        
        if n_spikes > 0: # Update information and inhibit the layer
            self.ls_spike_times.append(f"{t:3d}")
            self.t_last_spike = t
            layer_b.inhibit() # ***
        
        return ks
    
    def plot_weights(self, weight_pixels, n_spikes_since_reset, indices, index_im, input_images, label, alpha_diff=0.5):
        
        """ Plot the incoming connection weights of the given output neurons. """
        
        def set_ax(pixels, ax, alpha, title):
            if np.max(pixels) > 1: # ***
                ax.imshow(pixels, cmap=plt.get_cmap('gray'), alpha=alpha, vmin=0, vmax=255)
            else:
                ax.imshow(pixels, cmap=plt.get_cmap('gray'), alpha=alpha, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=10)
        
        # Determine the proportion that each neuron fired for the last image, used to highlight said neurons
        spike_proportions = np.array(n_spikes_since_reset) / max(1, n_spikes_since_reset.sum())
        spike_proportions = spike_proportions/max(1, np.max(spike_proportions))
        spike_proportions = np.nan_to_num(spike_proportions, nan=0)

        # Determine certain dimensions of the multi-image plot
        n_ims_in = len(input_images)
        n_indices= len(indices)+n_ims_in
        n_cols = min(n_indices, 10)
        n_rows = int(np.ceil(n_indices/n_cols))
        
        # Plot the weights
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(25, 6 + 2*n_indices//10))
        if n_rows == 1: # Add an extra dimension if necessary
            axes = np.expand_dims(axes, 0)
        for i, input_image in enumerate(input_images): # Plot the input images
            set_ax(input_image, ax=axes[0, i], alpha=1, title="Input digit (label={})".format(label))
        for k, pixels in enumerate(weight_pixels): # Plot the weights
            ax = axes[(k+n_ims_in)//n_cols, (k+n_ims_in)%n_cols]
            set_ax(pixels=pixels, 
                   ax=ax,
                   alpha=(1-alpha_diff)+alpha_diff*spike_proportions[k],
                   title=f'Neuron {indices[k]}')
            if spike_proportions[k] > 0: # Highlight the border of the spiking neurons
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
        fig.suptitle('Trained on {} images'.format(index_im), fontsize=16)
        fig.tight_layout()
        plt.show()
    
    def evaluate_results(self, data_handler, neuron_label_counts, neuron_image_counts, labels, plot_cms=False):
        
        # Determine for each neuron to which label it most strongly responds
        neuron_label_dict = dict()
        for k in range(neuron_label_counts.shape[0]):
            neuron_label_dict[k] = np.argmax(neuron_label_counts[k])
        
# =============================================================================
#         total_entropy = 0
#         for i in range(10):
#             entropy_i = util.calc_entropy(neuron_label_counts[:, i])
#             #print("Entropy for digit {} is {:.2f}".format(i, entropy_i))
#             print("Entropy for digit {} is {:.2f} - {}".format(i, entropy_i, neuron_label_counts[:, i]))
#             total_entropy += entropy_i
#         print("Average entropy is {:.2f}".format(total_entropy/10))
# =============================================================================
        
        # Count the number of right and wrong classifications
        cnt_wrong = cnt_correct = 0
        label_entropies = [[] for _ in range(10)]
        label_spike_counts = [[] for _ in range(10)]
        cm_classifications = np.zeros((10, 10), dtype=np.uint32)
        cm_spike_confidence = np.zeros((10, 10), dtype=np.uint32)
        for index_im, spike_counts in enumerate(neuron_image_counts):
            
            # Count for each label how often its corresponding neurons spike
            label_counts = np.zeros(10, dtype=np.uint16)
            for k, spike_count in enumerate(spike_counts):
                neuron_label = neuron_label_dict[k]
                label_counts[neuron_label] += spike_count
                
            # The label for which the most neurons spiked is considered the prediction
            prediction = np.argmax(label_counts)
            cm_spike_confidence[:, prediction] += label_counts
            
            # Compare the prediction to the true label
            label = labels[index_im]
            if prediction == label: 
                cnt_correct += 1
            else: 
                cnt_wrong += 1
            cm_classifications[label, prediction] += 1
            #print("Image {:4d} has label {}, prediction was {}; cnt_correct={:04d}, cnt_wrong={:04d}".format(index_im, label, prediction, cnt_correct, cnt_wrong))
            
            entropy = util.calc_entropy(label_counts)
            #print(index_im, label_counts, "{:.2f}".format(entropy))
            label_entropies[label].append(entropy)
            label_spike_counts[label].append(label_counts.sum())
        
        label_spike_total = [np.sum(label_spike_counts[i]) for i in range(10)]
        label_spike_avg = [np.mean(label_spike_counts[i]) for i in range(10)]
        entropies_avg = [np.mean(label_entropies[i]) for i in range(10)]
        entropies_std = [np.std(label_entropies[i]) for i in range(10)]
        cm_confidence_error = normalize(cm_classifications, axis=0) - (normalize(cm_spike_confidence, axis=0))
        stdv = np.std(cm_confidence_error)
        
        accuracy = 100*cnt_correct/(cnt_correct+cnt_wrong)
        
        results_string = ""
        results_string += "Achieved an accuracy of {:.2f}%".format(accuracy) + "\n"
        results_string += "Spike count total: "+str(["{}={:8.2f}".format(i, label_spike_total[i]) for i in range(10)]) + "\n"
        results_string += "Spike count means: "+str(["{}={:8.2f}".format(i, label_spike_avg[i]) for i in range(10)]) + "\n"
        results_string += "    Entropy means: "+str(["{}={:8.2f}".format(i, entropies_avg[i]) for i in range(10)]) + "\n"
        results_string += "    Entropy stdvs: "+str(["{}={:8.2f}".format(i, entropies_std[i]) for i in range(10)]) + "\n"
        results_string += "  Mean overall spike count: {:5.2f}".format(np.mean(label_spike_avg)) + "\n"
        results_string += "      Mean overall entropy: {:5.2f}".format(np.mean(entropies_avg)) + "\n"
        results_string += "      Stdv overall entropy: {:5.2f}".format(np.mean(entropies_std)) + "\n"
        results_string += "Mean confidence deviation: {:5.2f}".format(np.mean(np.abs(cm_confidence_error))) + "\n"
        results_string += "Stdv confidence deviation: {:5.2f}".format(np.std(cm_confidence_error)) + "\n"
        print(results_string)
        
        if plot_cms:
            
            util.plot_confusion_matrix(normalize(cm_classifications, axis=0), 
                                       xticks=[digit for digit in range(10)],
                                       yticks=[digit for digit in range(10)], 
                                       title="Confusion Matrix Classifications - Normalized", 
                                       xlabel="Network classification", ylabel="True label",
                                       vmin=0, vmax=100, figsize=(10,7))
            
            util.plot_confusion_matrix(normalize(cm_spike_confidence, axis=0), 
                                       xticks=[digit for digit in range(10)],
                                       yticks=[digit for digit in range(10)], 
                                       title="Confusion Matrix Spike confidence - Normalized", 
                                       xlabel="Dominant neuron", ylabel="Spike counts", 
                                       vmin=0, vmax=100, figsize=(10,7))
            util.plot_confusion_matrix(cm_confidence_error, xticks=[digit for digit in range(10)],
                                       yticks=[digit for digit in range(10)], 
                                       title="Confusion Matrix confidence error", 
                                       xlabel="Dominant neuron", ylabel="Spike counts", 
                                       vmin=-stdv, vmax=stdv, figsize=(10,7))
        
        return accuracy, cm_classifications, cm_spike_confidence, cm_confidence_error
# =============================================================================
#         return (results_string, cm_classifications, cm_spike_confidence, cm_confidence_error,
#                 accuracy, np.mean(label_spike_avg), np.mean(entropies_avg), np.mean(entropies_std), 
#                 np.mean(np.abs(cm_confidence_error)), np.std(cm_confidence_error))
# =============================================================================
    