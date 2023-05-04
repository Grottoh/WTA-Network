__author__ = "Otto van der Himst"
__version__ = "1.0"
__email__ = "otto.vanderhimst@ru.nl"

import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import time
import pandas as pd

# Constants
SHAPE_MNIST_TRAIN = (60000, 28, 28)
SHAPE_MNIST_TEST  = (10000, 28, 28)

# Names of inner folders of integration run, for network a and b respectively
ND_NET_A = "net_a/"
ND_NET_B = "net_b/"

def calc_entropy(x):
    if np.any(x<0):
        print("Error when comptuing entropy: array should not contain negative values!")
        sys.exit()
    if x.sum() == 0:
        x += 1
    p = x / x.sum()
    entropy = (-p*np.ma.log(p).filled(0)).sum()
    return entropy
    
def dump(obj, filename):
    """ Save a pickle file. """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    
def load(filename):
    """ Load a pickle file. """
    with open(filename,'rb') as f:
        return pickle.load(f)
    
def save_img(np_array, filepath, normalized=False):
    """ Save a single channel image. """
    if normalized: # Scale the image as appropriate
        img = Image.fromarray((np_array*255).astype(np.uint8))
    else:
        img = Image.fromarray(np_array)
    img = img.convert("L")
    img.save(filepath) # Show the image

def squeeze_all(array):
    indices_empty = tuple(np.where(np.array(array.shape)==1)[0])
    np.squeeze(array, indices_empty)    
    return np.squeeze(array, indices_empty)
   
# =============================================================================
# def get_sinewave(amplitude, frequency, phase, delay, ms):
#     """ Get a sine wave that lasts for a given amount (<ms>) time. """
#     x = np.linspace(0, ms, ms)
#     w = (2*np.pi)/ms * frequency/(1000/ms)
#     wave = amplitude * np.sin(w*x + phase - delay) + 100
#     return x, wave
# =============================================================================
   
def get_sinewave(amplitude, frequency, phase, delay, ms):
    """ Get a sine wave that lasts for a given amount (<ms>) time. """
    x = np.linspace(0, ms, ms)
    w = (2*np.pi)/ms * frequency/(1000/ms)
    wave = amplitude * np.sin(w*x + phase - delay) + 1*amplitude
    return x, wave

def get_firing_probability(frequency, inhibition_wave, t, always_fire=False):
    if always_fire:
        return 1
    if not isinstance(inhibition_wave, type(None)):
        return (frequency*((200-inhibition_wave[t])/100))/1000
    return frequency/1000

def get_td_curve(intercept, slope, slope_mult, limit, plot=False):
    curve = []
    value = intercept
    if slope != 0:
        while value < limit:
            curve.append(value)
            value += slope
            slope *= slope_mult
    curve.append(limit)
    if plot:
        plt.plot([x for x in range(len(curve))], curve)
        plt.title("Feedback curve")
        plt.show()
    return curve

def normalize(cm, axis):
    return cm / cm.sum(axis) * 100
            
def plot_confusion_matrix(cm, xticks, yticks, title, xlabel, ylabel, vmin=None, vmax=None, figsize=(10,7)):
    try:
        import seaborn as sn
        from matplotlib.colors import LinearSegmentedColormap
    except:
        return
    df_cm = pd.DataFrame(cm, index = yticks, columns = xticks)
    plt.figure(figsize = (10,7))
    
    #cmap = LinearSegmentedColormap.from_list('RedGreenRed', ['crimson', 'lime', 'crimson'])
    #s = sn.heatmap(df_cm, annot=True, vmin=vmin, vmax=vmax, cmap=cmap)
    s = sn.heatmap(df_cm, annot=True, vmin=vmin, vmax=vmax)
    
    plt.title(title)
    s.set(xlabel=xlabel, ylabel=ylabel)
    plt.show()
            
def save_confusion_matrix(pd_figure, cm, xticks, yticks, title, xlabel, ylabel, vmin=None, vmax=None, figsize=(10,7)):
    try:
        import seaborn as sn
        from matplotlib.colors import LinearSegmentedColormap
    except:
        return
    df_cm = pd.DataFrame(cm, index = yticks, columns = xticks)
    plt.figure(figsize = (10,7))
    
    #cmap = LinearSegmentedColormap.from_list('RedGreenRed', ['crimson', 'lime', 'crimson'])
    #s = sn.heatmap(df_cm, annot=True, vmin=vmin, vmax=vmax, cmap=cmap)
    s = sn.heatmap(df_cm, annot=True, vmin=vmin, vmax=vmax)
    
    plt.title(title)
    s.set(xlabel=xlabel, ylabel=ylabel)
    plt.savefig(pd_figure)
    
    
def is_none(param):
    return param == "None" or isinstance(param, type(None))

class Parameters():
    
    def __init__(self):
        self.params_default     = "UNINITIALIZED."
        self.params_custom_str  = "Custom parameters:\nUNINITIALIZED."
    
    def load_all(self, params_custom, P_loaded, pf_pre_parameters):
        
        # Override the parameters with the loaded ones
        self.__dict__ = P_loaded.__dict__
        params_custom = P_loaded.__dict__
        
        # Maintain the information of where the parameters come from
        self.load_essential_params_only = True
        self.pf_pre_parameters = pf_pre_parameters
        
        # Indicate where all parameters where loaded from
        message = f"\nLoaded all parameters from <{pf_pre_parameters}>."
        self.params_custom_str += message
        print(message)
        
        #sys.exit("Not yet implemented, requires a way to iterate over all relevant attributes.")
    
    def customize_parameters(self, params_default, argv):
        """ Adjust the parameters according to the arguments in <argv> """
        params_custom = params_default.copy()
        self.params_custom_str = "Custom parameters:\n"
        for arg in argv:
            key, value = arg.split('=')
            if key in params_custom:
                params_custom[key] = value
                self.params_custom_str += "  {}={}\n".format(key, value)
            else:
                print("Argument <{}> is unknown, terminating.".format(arg))
                sys.exit()
        if argv == []:
            self.params_custom_str = "Custom parameters: None."
        return params_custom
    
    def interpret_boolean(self, param):
        if type(param) == bool:
            return param
        elif param in ['True', '1']:
            return True
        elif param in ['False', '0']:
            return False
        else:
            sys.exit("param '{}' cannot be interpreted as boolean".format(param))
    
    def interpret_tuple(self, param):
        if type(param) == tuple:
            return param
        try:
            l = param.strip("()").split(",")
            return tuple([float(item) for item in l])
        except:
            sys.exit("param '{}' cannot be interpreted as tuple".format(param))
    
    def set_common_default(self):
        
        """ Set the parameters that are common to most or all WTA networks. """
        
        self.params_default = {
            "pd_data":"./[00LF] data/",
            "nf_mnist":"mnist{}.pkl",
            "ndt_mnist_spikes_train":"mnist-spikes-train{}_{:.0f}hz_{:03d}ms_per{:05d}{}/",
            "ndt_mnist_spikes_test":"mnist-spikes-test{}_{:.0f}hz_{:03d}ms_per{:05d}{}/",
            "nft_mnist_spikes":"mnist-spikes{}_{:.0f}hz_{:03d}ms_slice-{:05d}-{:05d}{}",
            "tag_mnist_normal":"-normal",
            "tag_mnist_shuffled":"-shuffled",
            "tag_mnist_noisy":"_noisy-{}",
            "tag_mnist":None,
            
            "pd_results":"./[01LF] results/",
            "nd_weight_ims":"weight-images/",
            "nd_weights":"weights/",
            "nd_weights_past":"weights_past/",
            "nf_parameters_txt":"parameters.txt",
            "nf_parameters_pkl":"parameters.pkl",
            
            "load_essential_params_only":True,
            "pf_pre_parameters":None,
            
            "print_interval":500, 
            "plot_interval":500, 
            "save_weight_ims_interval":500,
            "save_weights_interval": 5000,
            "evaluation_interval":500,
            
            "hertz_i":200, "ms":150, "s_slice":100,
            "hertz_o":10/150*1000,
            "stdp_simple":False, "sigma":10, "t_f":2, "t_s":8, "eta_root_decay":0.8,
            "w_min_exp":1e-8, "w_max":0, "delta_w_max":None,
            
            "always_fire":False,
            
            "seed":2022,
            
            "inh_start_o":None,
            "inh_change_o":0.01,
            "inh_ratio_o":10,
            
            "e_min_o":None,
            
            "eta_mult":1,
            
            "softmax_probabilities": False,
            "allow_concurrent_spikes": False,
            
            "topdown_enabled": False,
            "topdown_stdp_mult":1,
            "params_td_curve":None,
            
            "save_test_results":False,
            "nd_test_results":"test_results/",
            "nf_test_parameters":"test_parameters.pkl",
            "nf_test_weights_io":"test_weights_io.npy",
            "nf_test_weights_ih":"test_weights_ih.npy",
            "nf_test_weights_ho":"test_weights_ho.npy",
            "nf_test_weights_of":"test_weights_of.npy",
            "nf_test_labels":"nf_test_labels.npy",
            "nf_neuron_label_counts":"nf_neuron_label_counts.npy",
            "nf_neuron_image_counts":"nf_neuron_image_counts.npy",
            }
    
    def set_common_custom(self, params_custom, argv):
        
        # Set all parameters
        self.pd_data = params_custom['pd_data'] # Path to data directory
        self.pf_mnist = self.pd_data + params_custom['nf_mnist'] # Path to MNIST file
        self.pdt_mnist_spikes_train = self.pd_data + params_custom['ndt_mnist_spikes_train'] # Path template to MNIST spikes directory
        self.pdt_mnist_spikes_test = self.pd_data + params_custom['ndt_mnist_spikes_test'] # Path template to MNIST spikes directory
        self.nft_mnist_spikes = params_custom['nft_mnist_spikes'] # Filename template of MNIST spike files
        self.tag_mnist_normal = params_custom['tag_mnist_normal'] # Tag that identifies normally ordered MNIST data
        self.tag_mnist_shuffled = params_custom['tag_mnist_shuffled'] # Tag that identifies shuffled MNIST data
        self.tag_mnist_noisy = params_custom['tag_mnist_noisy'] # Tag that identifies noisy MNIST
        self.tag_mnist = self.tag_mnist_normal if is_none(params_custom['tag_mnist']) else params_custom['tag_mnist']
        
        # The path to the directory where results are stored
        self.pd_results = params_custom['pd_results']
        
        # The name of the directory where past weights are stored when new ones are saved
        self.nd_weights_past = params_custom['nd_weights_past']
        
        # Pickle and text file names for the parameters
        self.nf_parameters_txt = params_custom['nf_parameters_txt']
        self.nf_parameters_pkl = params_custom['nf_parameters_pkl']
        
        # If True only load the most essential parameters, otherwise load them all (if a path is given)
        self.load_essential_params_only = self.interpret_boolean(params_custom['load_essential_params_only'])
        
        # The path to the to be loaded parameters
        self.pf_pre_parameters = params_custom['pf_pre_parameters']
        
        # How often certain data should be printed, plotted, or saved
        self.print_interval = int(params_custom['print_interval'])
        self.plot_interval = int(params_custom['plot_interval'])
        self.save_weight_ims_interval = int(params_custom['save_weight_ims_interval'])
        self.save_weights_interval = int(params_custom['save_weights_interval'])
        self.evaluation_interval = int(params_custom['evaluation_interval'])
        
        # Parameters for the spike data
        self.hertz_i = float(params_custom['hertz_i']) # The firing frequency of each input spike train
        self.ms    = int(params_custom['ms']) # The duration in miliseconds of each input spike train
        self.s_slice = int(params_custom['s_slice']) # The size of each slice of data
        
        # The combined firing frequency of neurons in the output layer
        self.hertz_o = float(params_custom['hertz_o'])
        
        # STDP parameters
        self.stdp_simple = self.interpret_boolean(params_custom['stdp_simple'])
        self.sigma = int(params_custom['sigma'])
        self.t_f = int(params_custom['t_f'])
        self.t_s = int(params_custom['t_s'])
        self.eta_root_decay = float(params_custom['eta_root_decay'])
        
        # Bounds on weights and weight changes
        self.w_min_exp = float(params_custom['w_min_exp'])
        self.w_max = float(params_custom['w_max'])
        self.delta_w_max = None if is_none(params_custom['delta_w_max']) else float(params_custom['delta_w_max'])
        
        self.always_fire = self.interpret_boolean(params_custom['always_fire'])
        
        self.seed = int(params_custom['seed'])
        
        self.inh_start_o = None if is_none(params_custom['inh_start_o']) else float(params_custom['inh_start_o'])
        self.inh_change_o  = float(params_custom['inh_change_o'])
        self.inh_ratio_o  = float(params_custom['inh_ratio_o'])
        
        self.e_min_o = None if is_none(params_custom['e_min_o']) else float(params_custom['e_min_o'])
        
        self.eta_mult = float(params_custom['eta_mult'])
        
        self.softmax_probabilities = self.interpret_boolean(params_custom['softmax_probabilities'])
        self.allow_concurrent_spikes = self.interpret_boolean(params_custom['allow_concurrent_spikes'])
        
        self.topdown_enabled = self.interpret_boolean(params_custom['topdown_enabled'])
        self.topdown_stdp_mult = float(params_custom['topdown_stdp_mult'])
        self.params_td_curve =  None if is_none(params_custom['params_td_curve']) else self.interpret_tuple(params_custom['params_td_curve'])
        
        self.save_test_results = params_custom['save_test_results']
        self.nd_test_results = params_custom['nd_test_results']
        self.nf_test_parameters = params_custom['nf_test_parameters']
        self.nf_test_weights_io = params_custom['nf_test_weights_io']
        self.nf_test_weights_ih = params_custom['nf_test_weights_ih']
        self.nf_test_weights_ho = params_custom['nf_test_weights_ho']
        self.nf_test_weights_of = params_custom['nf_test_weights_of']
        self.nf_test_labels = params_custom['nf_test_labels']
        self.nf_neuron_label_counts = params_custom['nf_neuron_label_counts']
        self.nf_neuron_image_counts = params_custom['nf_neuron_image_counts']
    
        # Load parameters if a valid path is given
        pf_pre_parameters = self.pf_pre_parameters
        if is_none(pf_pre_parameters):
            print("No path was provided to load parameters, leaving them unchanged.")
        else:
            print(f"Loading parameters from <{pf_pre_parameters}>.")
            P_loaded = load(pf_pre_parameters)
            
            # Load only the essential common parameters
            if self.load_essential_params_only:
                self.hertz_i = P_loaded.hertz_i
                self.ms = P_loaded.ms
                self.hertz_o = P_loaded.hertz_o
                self.w_min_exp = P_loaded.w_min_exp
                
                self.always_fire = P_loaded.always_fire
                
                self.seed = P_loaded.seed
                
                self.inh_start_o  = P_loaded.inh_start_o
                self.inh_change_o  = P_loaded.inh_change_o
                self.inh_ratio_o  = P_loaded.inh_ratio_o
                
                self.e_min_o = P_loaded.e_min_o
                
                self.softmax_probabilities = P_loaded.softmax_probabilities
                self.allow_concurrent_spikes = P_loaded.allow_concurrent_spikes
                
                self.topdown_enabled = P_loaded.topdown_enabled
                self.topdown_stdp_mult = P_loaded.topdown_stdp_mult
                self.params_td_curve = P_loaded.params_td_curve
                #self.params_td_curve = (2, 0, 1, 2)#P_loaded.params_td_curve
                
                self.pd_test_results = P_loaded.pd_test_results
                #self.pd_test_results = "[01LF] results/run-hierarchical_0030/" + params_custom['nd_test_results']
                
                message = ("\nLoaded essential parameters:\n"+
                           f" hertz_i={self.hertz_i}, ms={self.ms}, hertz_o={self.hertz_o}\n"+
                           f" w_min_exp={self.w_min_exp}"+
                           f" always_fire={self.always_fire}\n"+
                           f" seed={self.seed}\n"+
                           f" inh_start_o={self.inh_start_o}, inh_change_o={self.inh_change_o}, inh_ratio_o={self.inh_ratio_o}\n"+
                           f" e_min_o={self.e_min_o}\n"+
                           f" softmax_probabilities={self.softmax_probabilities}, allow_concurrent_spikes={self.allow_concurrent_spikes}\n"+
                           f" topdown_enabled={self.topdown_enabled}, topdown_stdp_mult={self.topdown_stdp_mult}, params_td_curve={self.params_td_curve}\n"
                           f" pd_test_results={self.pd_test_results}\n")
                self.params_custom_str += message
                
        
    def set_single(self, argv):
        
        """ Set the parameters corresponding to a single layer WTA network. """
        
        print("\n------------ Setting Parameters Single Network ------------")
        
        # Set the parameters that are common to most or all WTA networks
        self.set_common_default()
        
        # The default parameters
        self.params_default.update({
            
            "K":99,
            "learn":True,
            
            "pf_pretrained_weights":None,
            
            "print_results":True,
            "plot_weight_ims":True,
            
            "ndt_run":"run-single_{:04d}/",
            
            "save_weight_ims":True,
            "nft_weight_ims":"weights_t{:05d}_{:03d}.jpg",
            
            "save_weights":False,
            "nft_weights":"weights_t{:05d}.npy",
            })
     
        # Adjust the custom parameters according to the arguments in argv
        params_custom = self.customize_parameters(self.params_default.copy(), argv)
        
        # Customize the common parameters
        self.set_common_custom(params_custom, argv)
        
        # Load parameters if a valid path is given
        pf_pre_parameters = self.pf_pre_parameters
        if pf_pre_parameters != None and pf_pre_parameters != "None":
            P_loaded = load(pf_pre_parameters)
            if self.load_essential_params_only: # Load only some essential parameters
                # params_custom['tag_mnist'] = P_loaded.tag_mnist #***
                params_custom['K'] = P_loaded.K
                message = (f" tag_mnist={params_custom['tag_mnist']}\n"+
                           f" K={params_custom['K']}\n")
                self.params_custom_str += message
            else: # Load all parameters
                self.load_all(params_custom, P_loaded, pf_pre_parameters)
        
        # Designate which MNIST to use
        self.tag_mnist = params_custom['tag_mnist'] # The tag to be used for a given run
        
        # The number of neurons in the WTA circuit
        self.K = int(params_custom['K'])
        
        # Determine whether the network should learn (i.e. update the weights)
        self.learn = self.interpret_boolean(params_custom['learn'])
        
        # The path to previously trained weights
        self.pf_pretrained_weights = params_custom['pf_pretrained_weights']
        
        # Whether certain results should be printed
        self.print_results = self.interpret_boolean(params_custom['print_results'])
        
        # Whether network weights (converted to pixels) are plotted
        self.plot_weight_ims = self.interpret_boolean(params_custom['plot_weight_ims'])
        
        # Whether or not pixel representation of the weight and/or the weights themselves should be stored
        self.save_weight_ims = self.interpret_boolean(params_custom['save_weight_ims'])
        self.save_weights = self.interpret_boolean(params_custom['save_weights'])
        self.pd_weight_ims = self.nft_weight_ims = self.pd_weights = self.nft_weights = "None"
        if self.save_weight_ims or self.save_weights or self.save_test_results:
            
            # Create a directory for this run
            ndt_run = params_custom['ndt_run']
            self.run_id = len([name for name in os.listdir(self.pd_results) if os.path.isdir(self.pd_results+name) and name[:-5]==ndt_run[:-8]])
            self.pd_run = self.pd_results + ndt_run.format(self.run_id)
            os.mkdir(self.pd_run)
            print("Created directory for storing data from this run: <{}>".format(self.pd_run))
            
            # Create the directory where pixel representations of the weights will be stored
            if self.save_weight_ims:
                self.pd_weight_ims = self.pd_run + params_custom['nd_weight_ims']
                os.mkdir(self.pd_weight_ims)
                self.nft_weight_ims = params_custom['nft_weight_ims']
            
            # Create the directory where the weights will be stored
            if self.save_weights:
                self.pd_weights = self.pd_run + params_custom['nd_weights']
                os.mkdir(self.pd_weights)
                self.nft_weights = params_custom['nft_weights']
            
            self.pd_test_results = self.pd_run + params_custom['nd_test_results']
                
            # Write the parameters to a text file and store it            
            with open(self.pd_run+'parameters.txt', 'w') as f:
                f.write(str(self))
                
            # Store the parameters as a .pkl file
            dump(self, self.pd_run+self.nf_parameters_pkl)
        
        print(self)

    def set_hierarchical(self, argv):
        
        """ Set the parameters corresponding to a hierarchical WTA network. """
        
        print("\n--------- Setting Parameters Hierarchical Network ---------")
        
        # Set the parameters that are common to most or all WTA networks
        self.set_common_default()
        
        # The default parameters
        self.params_default.update({
        
            "K_h":38, "K_o":99,
            "learn_h":True, "learn_o":True,
            
            "pf_pretrained_weights_ih":None, 
            
            "pf_pretrained_weights_ho":None,
            
            "print_results_h":True, "print_results_o":True,
            "plot_weight_ims_h":True, "plot_weight_ims_o":True,
            
            "ndt_run":"run-hierarchical_{:04d}/",

            "save_weight_ims_h":True, 
            "nd_weight_ims_h":"hidden/",
            "nft_weight_ims_h":"weights-h_t{:05d}_{}-{}_{:03d}.jpg",
            "save_weight_ims_o":True,
            "nd_weight_ims_o":"output/",
            "nft_weight_ims_o":"weights-o_t{:05d}_{:03d}.jpg",
            
            
            "save_weights_h":False,
            "nd_weights_h":"hidden/",
            "nft_weights_h":"weights-h_t{:05d}_{}-{}.npy",
            "save_weights_o":False, 
            "nd_weights_o":"output/",
            "nft_weights_o":"weights-o_t{:05d}.npy",
            
            "inh_start_h":None,
            "inh_change_h":0.01,
            "inh_ratio_h":10,
            
            "e_min_h":None,
            
            "hertz_h":10/150*1000,
            })
     
        # Adjust the custom parameters according to the arguments in argv
        params_custom = self.customize_parameters(self.params_default.copy(), argv)
        
        # Customize the common parameters
        self.set_common_custom(params_custom, argv)
        
        # Load parameters if a valid path is given
        pf_pre_parameters = self.pf_pre_parameters
        if pf_pre_parameters != None and pf_pre_parameters != "None":
            P_loaded = load(pf_pre_parameters)
            if self.load_essential_params_only: # Load only some essential parameters
                params_custom['tag_mnist'] = P_loaded.tag_mnist # *** !!!! $$$$$$$$$$
                params_custom['K_h'] = P_loaded.K_h
                params_custom['K_o'] = P_loaded.K_o
                params_custom['inh_start_h'] = P_loaded.inh_start_h
                params_custom['inh_change_h'] = P_loaded.inh_change_h
                params_custom['inh_ratio_h'] = P_loaded.inh_ratio_h
                params_custom['e_min_h'] = P_loaded.e_min_h
                message = (f" tag_mnist={params_custom['tag_mnist']}\n"+
                           f" K={params_custom['K_h']},  K={params_custom['K_o']}\n"+
                           f" inh_start_h={params_custom['inh_start_h']}, inh_change_h={params_custom['inh_change_h']},"+
                           f" inh_change_h={params_custom['inh_change_h']}, inh_ratio_h={params_custom['inh_ratio_h']},"+
                           f" inh_ratio_h={params_custom['inh_ratio_h']}\n"+
                           f" e_min_o={params_custom['e_min_o']}\n")
                self.params_custom_str += message
            else: # Load all parameters
                self.load_all(params_custom, P_loaded, pf_pre_parameters)
                
        # Designate which MNIST to use
        self.tag_mnist = params_custom['tag_mnist'] # The tag to be used for a given run
        
        # Set the number of neurons per WTA circuit
        self.K_h = int(params_custom['K_h']) # The number of neurons per hidden circuit
        self.K_o = int(params_custom['K_o']) # The number of neurons for the output circuit
    
        # Determine which parts of the network should learn (i.e. update the weights)
        self.learn_h = self.interpret_boolean(params_custom['learn_h'])
        self.learn_o = self.interpret_boolean(params_custom['learn_o'])
        
        self.inh_start_h = None if is_none(params_custom['inh_start_h']) else float(params_custom['inh_start_h'])
        self.inh_change_h  = float(params_custom['inh_change_h'])
        self.inh_ratio_h  = float(params_custom['inh_ratio_h'])
        
        self.e_min_h = None if is_none(params_custom['e_min_h']) else float(params_custom['e_min_h'])
        
        self.hertz_h = float(params_custom['hertz_h']) # The target firing frequency of the hidden layer
        
        # The paths to previously trained weights
        self.pf_pretrained_weights_ih = params_custom['pf_pretrained_weights_ih']
        self.pf_pretrained_weights_ho = params_custom['pf_pretrained_weights_ho']
        
        # Whether certain results should be printed
        self.print_results_h = self.interpret_boolean(params_custom['print_results_h'])
        self.print_results_o = self.interpret_boolean(params_custom['print_results_o'])
        
        # Whether network weights (converted to pixels) are plotted
        self.plot_weight_ims_h = self.interpret_boolean(params_custom['plot_weight_ims_h'])
        self.plot_weight_ims_o = self.interpret_boolean(params_custom['plot_weight_ims_o'])
        
        # Whether or not pixel representation of the weight and/or the weights themselves should be stored
        self.save_weight_ims_h = self.interpret_boolean(params_custom['save_weight_ims_h'])
        self.save_weight_ims_o = self.interpret_boolean(params_custom['save_weight_ims_o'])
        self.save_weights_h = self.interpret_boolean(params_custom['save_weights_h'])
        self.save_weights_o = self.interpret_boolean(params_custom['save_weights_o'])
        self.pd_weight_ims_h = self.nft_weight_ims_h = self.pd_weight_ims_o = self.nft_weight_ims_o = "NONE"
        self.pd_weights_h = self.pd_weights_o = self.nft_weights_h = self.nft_weights_o = "NONE"
        if self.save_weight_ims_h or self.save_weight_ims_o or self.save_weights_h or self.save_weights_o:
            
            # Create a directory for this run
            self.pd_results = params_custom['pd_results']
            ndt_run = params_custom['ndt_run']
            if ndt_run[-len(ND_NET_A):] == ND_NET_A or ndt_run[-len(ND_NET_B):] == ND_NET_B:
                self.pd_run = ndt_run
            else:
                self.run_id = len([name for name in os.listdir(self.pd_results) if os.path.isdir(self.pd_results+name) and name[:-5]==ndt_run[:-8]])
                self.pd_run = self.pd_results + ndt_run.format(self.run_id)
            os.mkdir(self.pd_run)
            print("Created directory for storing data from this run: <{}>".format(self.pd_run))
            
            # # Create the directories where pixel representations of the weights will be stored 
            if self.save_weight_ims_h or self.save_weight_ims_o:
                pd_weight_ims = self.pd_run + params_custom['nd_weight_ims']
                os.mkdir(pd_weight_ims) # Create the upper directory
                if self.save_weight_ims_h: # Create a directory for images of the hidden weights
                    self.pd_weight_ims_h = pd_weight_ims + params_custom['nd_weight_ims_h']
                    os.mkdir(self.pd_weight_ims_h)
                    self.nft_weight_ims_h = params_custom['nft_weight_ims_h']
                if self.save_weight_ims_o:  # Create a directory for images of the output weights
                    self.pd_weight_ims_o = pd_weight_ims + params_custom['nd_weight_ims_o']
                    os.mkdir(self.pd_weight_ims_o)
                    self.nft_weight_ims_o = params_custom['nft_weight_ims_o']
            
            # Create the directory where the weights will be stored
            if self.save_weights_h or self.save_weights_o:
                pd_weights = self.pd_run + params_custom['nd_weights']
                os.mkdir(pd_weights) # Create the upper directory for the weights
                if self.save_weights_h: # Create a directory for the hidden weights    
                    self.pd_weights_h = pd_weights + params_custom['nd_weights_h']
                    os.mkdir(self.pd_weights_h)
                    self.nft_weights_h = params_custom['nft_weights_h']
                if self.save_weights_o: # Create a directory for the output weights
                    self.pd_weights_o = pd_weights + params_custom['nd_weights_o']
                    os.mkdir(self.pd_weights_o)
                    self.nft_weights_o = params_custom['nft_weights_o']
                
            self.pd_test_results = self.pd_run + params_custom['nd_test_results']
            
            # Write the parameters to a text file and store it            
            with open(self.pd_run+self.nf_parameters_txt, 'w') as f:
                f.write(str(self))
                
            # Store the parameters as a .pkl file
            dump(self, self.pd_run+self.nf_parameters_pkl)
        
        print(self)

    def set_integration(self, argv):
        
        """ Set the parameters corresponding to an integration WTA network. """
        
        print("\n---------- Setting Parameters Integration Network ----------")
    
        # Set the parameters that are common to most or all WTA networks
        self.set_common_default()
        
        # The default parameters
        self.params_default.update({            
            "K":98,
            "learn":True,
            
            "pf_pretrained_weights":None, 
            
            "print_results":True,
            "plot_weight_ims":True,
            
            "ndt_run":"run-integration_{:04d}/",
            
            "save_weight_ims":True,
            "nft_weight_ims":"weights_t{:05d}_{:03d}.jpg",
            
            "save_weights":False,
            "nft_weights":"weights_t{:05d}.npy",
            })
        
        # Adjust the custom parameters according to the arguments in argv
        params_custom = self.customize_parameters(self.params_default.copy(), argv)
        
        # Customize the common parameters
        self.set_common_custom(params_custom, argv)
        
        # Load parameters if a valid path is given
        pf_pre_parameters = self.pf_pre_parameters
        if pf_pre_parameters != None and pf_pre_parameters != "None":
            P_loaded = load(pf_pre_parameters)
            if self.load_essential_params_only: # Load only some essential parameters
                params_custom['K'] = P_loaded.K
                message = f" K={params_custom['K']}\n"
                self.params_custom_str += message
            else: # Load all parameters
                self.load_all(params_custom, P_loaded, pf_pre_parameters)
        
        # Set the number of neurons per WTA circuit
        self.K = int(params_custom['K']) # The number of final neurons
        
        # Enable or disabling learning for each layer
        self.learn = self.interpret_boolean(params_custom['learn'])
        
        # The path to previously trained weights
        self.pf_pretrained_weights = params_custom['pf_pretrained_weights']
        
        # Whether certain results should be printed
        self.print_results = self.interpret_boolean(params_custom['print_results'])
        
        # Whether network weights (converted to pixels) are plotted
        self.plot_weight_ims = self.interpret_boolean(params_custom['plot_weight_ims'])
        
        # Whether or not pixel representation of the weight and/or the weights themselves should be stored
        self.save_weight_ims = self.interpret_boolean(params_custom['save_weight_ims'])
        self.save_weights = self.interpret_boolean(params_custom['save_weights'])
        self.pd_weight_ims = self.nft_weight_ims = self.pd_weights = self.nft_weights = "None"
        if self.save_weight_ims or self.save_weights:
            
            # Create a directory for this run
            ndt_run = params_custom['ndt_run']
            self.run_id = len([name for name in os.listdir(self.pd_results) if os.path.isdir(self.pd_results+name) and name[:-5]==ndt_run[:-8]])
            self.pd_run = self.pd_results + ndt_run.format(self.run_id)
            os.mkdir(self.pd_run)
            print("Created directory for storing data from this run: <{}>".format(self.pd_run))
            
            # Create the directory where pixel representations of the weights will be stored
            if self.save_weight_ims:
                self.pd_weight_ims = self.pd_run + params_custom['nd_weight_ims']
                os.mkdir(self.pd_weight_ims)
                self.nft_weight_ims = params_custom['nft_weight_ims']
            
            # Create the directory where the weights will be stored
            if self.save_weights:
                self.pd_weights = self.pd_run + params_custom['nd_weights']
                os.mkdir(self.pd_weights)
                self.nft_weights = params_custom['nft_weights']
                
            # Write the parameters to a text file and store it            
            with open(self.pd_run+'parameters.txt', 'w') as f:
                f.write(str(self))
            
            self.pd_test_results = self.pd_run + params_custom['nd_test_results']
                
            # Store the parameters as a .pkl file
            dump(self, self.pd_run+self.nf_parameters_pkl)
        
        print(self)

    def __repr__(self):
        string = ("\nDefault parameters:\n{}\n".format(str(self.params_default)))
        string += "\n"
        string += self.params_custom_str
        return string
      
    def __str__(self):
        return self.__repr__()
        
