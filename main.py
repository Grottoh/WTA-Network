__author__ = "Otto van der Himst"
__version__ = "1.0"
__email__ = "otto.vanderhimst@ru.nl"

import sys
import util
from util import Parameters, save_img, load
from data_handler import DataHandler
from net_single import NetworkSingle
from net_hierarchical import NetworkHierarchical
from net_integration import NetworkIntegration
import numpy as np
import traceback
import gc
import time
import multiprocessing
import os
import param_manager

try:
    os.nice(19)
except:
    pass

def run_single(argv, train=True):
        
    # Set the parameters
    P = Parameters()
    P.set_single(argv)
    
    # Initialize the single-layer network
    network = NetworkSingle(P)
    
    # Initialize the DataHandler
    data_handler = DataHandler(P)
    
    # Create and store new spike data if it is not in storage already
    data_handler.store_mnist_as_spikes(train=train, tag_mnist=network.tag_mnist)
    
    if train: # Train the network
        network.train(data_handler=data_handler, P=P)
        return P.run_id
    else: # Test the network
        network.test(data_handler=data_handler, P=P)

def run_hierarchical(argv, train=True):
        
    # Set the parameters
    P = Parameters()
    P.set_hierarchical(argv)
    
    # Initialize the hierarchical network
    network = NetworkHierarchical(P)
    
    # Initialize the DataHandler
    data_handler = DataHandler(P)
    
    # Create and store new spike data if it is not in storage already
    data_handler.store_mnist_as_spikes(train=train, tag_mnist=network.tag_mnist)
    
    if train: # Train the network
        network.train(data_handler=data_handler, P=P)
        return P.run_id
    else:
        network.test(data_handler=data_handler, P=P)

def run_integration(argv, argv_a, argv_b, train=True):
        
    # Set the parameters
    P = Parameters()
    P_a = Parameters()
    P_b = Parameters()
    P.set_integration(argv)
    if train:
        argv_a.append("ndt_run={}".format(P.pd_run+util.ND_NET_A))
        argv_b.append("ndt_run={}".format(P.pd_run+util.ND_NET_B))
    P_a.set_hierarchical(argv_a)
    P_b.set_hierarchical(argv_b)
    
    # Initialize the integration network
    network = NetworkIntegration(P, P_a, P_b)
    
    # Initialize the DataHandler
    data_handler = DataHandler(P)
    
    # Create and store new spike data if it is not in storage already
    data_handler.store_mnist_as_spikes(train=train, tag_mnist=network.net_a.tag_mnist)
    data_handler.store_mnist_as_spikes(train=train, tag_mnist=network.net_b.tag_mnist)
    
    data = data_handler.get_mnist_spikes(0, network.net_a.tag_mnist)
    
    if train: # Train the network
        network.train(data_handler=data_handler, P=P)
        return P.run_id
    else:
        network.test(data_handler=data_handler, P=P)
 
def train_and_test_hierarchical(argv_train):
    run_id = run_hierarchical(argv_train, train=True)
                               
    load_run_id = f"{run_id:04d}"
    load_run_t = "59999"
    print_plot_interval = 1000
    evaluation_interval = 1000
    
    plot_and_save_ims = False
    save_test_results = True

    argv = [
        "pf_pretrained_weights_ih=[01LF] results/run-hierarchical_{}/weights/hidden/".format(load_run_id),
        
        "pf_pretrained_weights_ho=[01LF] results/run-hierarchical_{}/weights/output/".format(load_run_id),
                    
        "load_essential_params_only=True",
        "pf_pre_parameters=[01LF] results/run-hierarchical_{}/parameters.pkl".format(load_run_id),
        
          "print_results_h=True",    "print_results_o=True", f"print_interval={print_plot_interval}", 
        f"plot_weight_ims_h={plot_and_save_ims}",  f"plot_weight_ims_o={plot_and_save_ims}", f"plot_interval={print_plot_interval}",
        "save_weight_ims_h=False", "save_weight_ims_o=False", 
           "save_weights_h=False",    "save_weights_o=False",
                                                          f"evaluation_interval={evaluation_interval}",
          
        "learn_h=False", "learn_o=False",
        #"tag_mnist={}".format(tag_mnist),
        "save_test_results={save_test_results}",
        ]
    run_hierarchical(argv, train=False)
 
def test_hierarchical(load_run_id):
    load_run_t = "59999"
    print_plot_interval = 1000
    evaluation_interval = 1000
    
    plot_and_save_ims = False
    save_test_results = True

    argv = [
        "pf_pretrained_weights_ih=[01LF] results/run-hierarchical_{}/weights/hidden/".format(load_run_id),
        
        "pf_pretrained_weights_ho=[01LF] results/run-hierarchical_{}/weights/output/".format(load_run_id),
                    
        "load_essential_params_only=True",
        "pf_pre_parameters=[01LF] results/run-hierarchical_{}/parameters.pkl".format(load_run_id),
        
          "print_results_h=True",    "print_results_o=True", f"print_interval={print_plot_interval}", 
        f"plot_weight_ims_h={plot_and_save_ims}",  f"plot_weight_ims_o={plot_and_save_ims}", f"plot_interval={print_plot_interval}",
        "save_weight_ims_h=False", "save_weight_ims_o=False", 
           "save_weights_h=False",    "save_weights_o=False",
                                                          f"evaluation_interval={evaluation_interval}",
          
        "learn_h=False", "learn_o=False",
        #"tag_mnist={}".format(tag_mnist),
        "save_test_results={save_test_results}",
        ]
    run_hierarchical(argv, train=False)
 
def train_and_test_integration(argvs_train):
    run_id = run_integration(*argvs_train, train=True)
                               
    load_run_id = f"{run_id:04d}"
    load_run_t = "59999"
    print_plot_interval = 1000
    evaluation_interval = 1000
    
    plot_and_save_ims = False
    save_test_results = True

    net = "a"
    #tag_mnist = "-normal"
    argv = [ 
        "pf_pretrained_weights_ih=[01LF] results/run-integration_{}/net_{}/weights/hidden/".format(load_run_id, net),
        
        "pf_pretrained_weights_ho=[01LF] results/run-integration_{}/net_{}/weights/output/".format(load_run_id, net),
                    
        "load_essential_params_only=True",
        "pf_pre_parameters=[01LF] results/run-integration_{}/net_{}/parameters.pkl".format(load_run_id, net),
        
          "print_results_h=True",    "print_results_o=True", f"print_interval={print_plot_interval}", 
        f"plot_weight_ims_h={plot_and_save_ims}",  f"plot_weight_ims_o={plot_and_save_ims}", f"plot_interval={print_plot_interval}",
        "save_weight_ims_h=False", "save_weight_ims_o=False", 
           "save_weights_h=False",    "save_weights_o=False",
                                                          f"evaluation_interval={evaluation_interval}",
          
        "learn_h=False", "learn_o=False",
        #"tag_mnist={}".format(tag_mnist),
        "save_test_results={save_test_results}",
        ]
    run_hierarchical(argv, train=False)

    net = "b"
    #tag_mnist = "-shuffled"
    argv = [ 
        "pf_pretrained_weights_ih=[01LF] results/run-integration_{}/net_{}/weights/hidden/".format(load_run_id, net),
        
        "pf_pretrained_weights_ho=[01LF] results/run-integration_{}/net_{}/weights/output/".format(load_run_id, net),
                    
        "load_essential_params_only=True",
        "pf_pre_parameters=[01LF] results/run-integration_{}/net_{}/parameters.pkl".format(load_run_id, net),
        
          "print_results_h=True",    "print_results_o=True", f"print_interval={print_plot_interval}", 
        f"plot_weight_ims_h={plot_and_save_ims}",  f"plot_weight_ims_o={plot_and_save_ims}", f"plot_interval={print_plot_interval}",
        "save_weight_ims_h=False", "save_weight_ims_o=False", 
           "save_weights_h=False",    "save_weights_o=False",
                                                          f"evaluation_interval={evaluation_interval}",
          
        "learn_h=False", "learn_o=False",
        #"tag_mnist={}".format(tag_mnist),
        "save_test_results={save_test_results}",
        ]
    run_hierarchical(argv, train=False)
    
    argv_a = [        
        "pf_pretrained_weights_ih={}".format("[01LF] results/run-integration_{}/{}weights/hidden/".format(load_run_id, util.ND_NET_A)),
        "pf_pretrained_weights_ho={}".format("[01LF] results/run-integration_{}/{}weights/output/".format(load_run_id, util.ND_NET_A)),
        
        "load_essential_params_only=True",
        "pf_pre_parameters={}".format("[01LF] results/run-integration_{}/{}parameters.pkl".format(load_run_id, util.ND_NET_A)),
        
          "print_results_h=False",    "print_results_o=False", f"print_interval={print_plot_interval}", 
        "plot_weight_ims_h=False",  f"plot_weight_ims_o={plot_and_save_ims}",   f"plot_interval={print_plot_interval}",
        "save_weight_ims_h=False", "save_weight_ims_o=False", 
           "save_weights_h=False",    "save_weights_o=False",
        
        "learn_h=False", f"learn_o=False",
        "tag_mnist=-normal",
        f"save_test_results={save_test_results}",
        ]
    argv_b = [        
        "pf_pretrained_weights_ih={}".format("[01LF] results/run-integration_{}/{}weights/hidden/".format(load_run_id, util.ND_NET_B)),
        "pf_pretrained_weights_ho={}".format("[01LF] results/run-integration_{}/{}weights/output/".format(load_run_id, util.ND_NET_B)),
        
        "load_essential_params_only=True",
        "pf_pre_parameters={}".format("[01LF] results/run-integration_{}/{}parameters.pkl".format(load_run_id, util.ND_NET_B)),
        
          "print_results_h=False",    "print_results_o=False", f"print_interval={print_plot_interval}", 
        "plot_weight_ims_h=False",  f"plot_weight_ims_o={plot_and_save_ims}",   f"plot_interval={print_plot_interval}",
        "save_weight_ims_h=False", "save_weight_ims_o=False", 
           "save_weights_h=False",    "save_weights_o=False",
        
        "learn_h=False", f"learn_o=False",
        "tag_mnist=-shuffled",
        f"save_test_results={save_test_results}",
        ]
    
    argv = [
       "pf_pretrained_weights={}".format("[01LF] results/run-integration_{}/weights/weights_t{}.npy".format(load_run_id, load_run_t)),
        
        "load_essential_params_only=True",
        "pf_pre_parameters={}".format("[01LF] results/run-integration_{}/parameters.pkl".format(load_run_id)),
        
          "print_results=False", f"print_interval={print_plot_interval}",
        f"plot_weight_ims={plot_and_save_ims}",   f"plot_interval={print_plot_interval}",
        "save_weight_ims=False",
           "save_weights=False",
                            f"evaluation_interval={evaluation_interval}",
 
        "learn=False",
        f"save_test_results={save_test_results}",
        ]
    run_integration(argv, argv_a, argv_b, train=False)         

def main(argv):
    
    #task = "train-single" # No longer supported
    #task = "test-single" # No longer supported
    #task = "train-hierarchical"
    #task = "test-hierarchical"
    task = "train-integration"
    #task = "test-integration"
    #task = "train-and-test-hierarchical"
    #task = "train-and-test-integration"
    
# =============================================================================
#     SINGLE - TRAIN
# =============================================================================
    
    if task == "train-single":
        
        argv = [
              "print_results=True",           "print_interval=100",
            "plot_weight_ims=True",            "plot_interval=100",
            "save_weight_ims=True", "save_weight_ims_interval=1000", 
               "save_weights=True",    "save_weights_interval=1000",
            
            "K=99",
            ]
        
        # Train a single layer WTA network on the MNIST
        run_single(argv, train=True)
        
# =============================================================================
#     SINGLE - TEST
# =============================================================================
    load_run_id = "0000"
    #load_run_t = "52000"
    load_run_t = "59999"
    if task == "test-single":
        
        argv = [
            #"pf_pretrained_weights=None",
            "pf_pretrained_weights=[01LF] results/run-single_{}/weights/weights_t{}.npy".format(load_run_id, load_run_t),
            
            "load_essential_params_only=True",
            #"pf_pre_parameters=None",
            "pf_pre_parameters=[01LF] results/run-single_{}/parameters.pkl".format(load_run_id),
            
              "print_results=False", "print_interval=100",
            "plot_weight_ims=True",   "plot_interval=100",
            "save_weight_ims=False",
               "save_weights=False",
                                "evaluation_interval=500",
             
            "learn=False",
            ]
        
        # Test a single layer WTA network on the MNIST
        run_single(argv, train=False)
      
# =============================================================================
#     HIERARCHICAL - TRAIN
# =============================================================================
    if task == "train-hierarchical":
        
        plot = "True"
        save = "True"
        hertz_h = hertz_o = 20/150*1000
        argv = [
              "print_results_h=True",  "print_results_o=True",           "print_interval=500", 
            f"plot_weight_ims_h={plot}", f"plot_weight_ims_o={plot}",            "plot_interval=500",
            f"save_weight_ims_h={save}", f"save_weight_ims_o={save}", "save_weight_ims_interval=5000", 
               f"save_weights_h={save}",    f"save_weights_o={save}",    "save_weights_interval=5000",
            
            "K_h=38", "K_o=99",
            
            "tag_mnist=-normal",
            
            f"hertz_h={hertz_h}", f"hertz_o={hertz_o}",
            
            "topdown_enabled=False",
            "params_td_curve=(2.0, 0.0, 1, 2)",
            ]
        
        run_hierarchical(argv, train=True)
      
# =============================================================================
#     HIERARCHICAL - TEST
# =============================================================================
    load_run_id = "0000"
    net = "a"
    
    tag_mnist = "-normal"
    #tag_mnist = "-shuffled"
    #tag_mnist = "-normal_noisy-20"
    #tag_mnist = "-shuffled_noisy-10"
    
    if task == "test-hierarchical":
        argv = [ 
            #"pf_pretrained_weights_ih=None",
            #"pf_pretrained_weights_ih=manual_saves/run-integration_{}/net_{}/weights/hidden/".format(load_run_id, net),
            "pf_pretrained_weights_ih=[01LF] results/run-hierarchical_{}/weights/hidden/".format(load_run_id),
            
            #"pf_pretrained_weights_ho=None",
            #"pf_pretrained_weights_ho=manual_saves/run-integration_{}/net_{}/weights/output/".format(load_run_id, net),
            "pf_pretrained_weights_ho=[01LF] results/run-hierarchical_{}/weights/output/".format(load_run_id),
                        
            "load_essential_params_only=True",
            #"pf_pre_parameters=None",
            #"pf_pre_parameters=manual_saves/run-integration_{}/net_{}/parameters.pkl".format(load_run_id, net),
            "pf_pre_parameters=[01LF] results/run-hierarchical_{}/parameters.pkl".format(load_run_id),
            
              "print_results_h=True",    "print_results_o=True", "print_interval=500", 
            "plot_weight_ims_h=True",  "plot_weight_ims_o=True",   "plot_interval=1",
            "save_weight_ims_h=False", "save_weight_ims_o=False", 
               "save_weights_h=False",    "save_weights_o=False",
                                                              "evaluation_interval=500",
              
            "learn_h=False", "learn_o=False",
            
            "tag_mnist={}".format(tag_mnist),
            
            "save_test_results=True",
            ]
        run_hierarchical(argv, train=False)
    
# =============================================================================
#     INTEGRATION - TRAIN
# =============================================================================
    if task == "train-integration":
        
        K_h = "38"
        K_o = "99"
        K_f = "98"
        print_plot_interval = 1
        save_weights_interval = 5000
        
        hertz_h = hertz_o = 20/150*1000
        hertz_f = 10/150*1000
        
        topdown_enabled = True
        params_td_curve = (2.0, 0.0, 1.0, 2)
            
        plot_and_save_ims = True
        argv_a = [                
              "print_results_h=False",  "print_results_o=False",                                        f"print_interval={print_plot_interval}", 
            f"plot_weight_ims_h={plot_and_save_ims}", f"plot_weight_ims_o={plot_and_save_ims}",            f"plot_interval={print_plot_interval}",
            f"save_weight_ims_h={plot_and_save_ims}", f"save_weight_ims_o={plot_and_save_ims}", f"save_weight_ims_interval={print_plot_interval}", 
               "save_weights_h=True",    "save_weights_o=True",                                   f"save_weights_interval={save_weights_interval}",
         
            f"K_h={K_h}", f"K_o={K_o}",
            
            "tag_mnist=-normal",
            
            f"hertz_h={hertz_h}", f"hertz_o={hertz_o}",
            
            f"topdown_enabled={topdown_enabled}",
            f"params_td_curve={params_td_curve}",
            
            "learn_h=False",
            "learn_o=False",
            ]
        argv_b = [
              "print_results_h=False",  "print_results_o=False",                                        f"print_interval={print_plot_interval}", 
            f"plot_weight_ims_h={plot_and_save_ims}", f"plot_weight_ims_o={plot_and_save_ims}",            f"plot_interval={print_plot_interval}",
            f"save_weight_ims_h={plot_and_save_ims}", f"save_weight_ims_o={plot_and_save_ims}", f"save_weight_ims_interval={print_plot_interval}", 
               "save_weights_h=True",    "save_weights_o=True",                                   f"save_weights_interval={save_weights_interval}",
         
            f"K_h={K_h}", f"K_o={K_o}",
            
            "tag_mnist=-shuffled",
            
            f"hertz_h={hertz_h}", f"hertz_o={hertz_o}",
            
            f"topdown_enabled={topdown_enabled}",
            f"params_td_curve={params_td_curve}",
            
            "learn_h=False",
            "learn_o=False",
            ]
        
        argv = [
            
            "pf_pretrained_weights=None",
            #"pf_pretrained_weights=,
            
            "load_essential_params_only=True",
            "pf_pre_parameters=None",
            #"pf_pre_parameters=,
            
              "print_results=True",                          f"print_interval={print_plot_interval}",
            f"plot_weight_ims={plot_and_save_ims}",            f"plot_interval={print_plot_interval}",
            f"save_weight_ims={plot_and_save_ims}", f"save_weight_ims_interval={print_plot_interval}", 
               "save_weights=True",                    f"save_weights_interval={save_weights_interval}",
            
            f"K={K_f}",
            
            f"hertz_o={hertz_f}",
            
            f"topdown_enabled={topdown_enabled}",
            f"params_td_curve={params_td_curve}",
            
            "learn=False",
            ]
        
        run_integration(argv, argv_a, argv_b, train=True)
    
# =============================================================================
#     INTEGRATION - TEST
# =============================================================================
    if task == "test-integration":
        
        load_run_id = "0024"
        load_run_t = "05000"
        load_run_t = "59999"
        print_plot_interval = 100
        
        tag_noisy_a = ""
        tag_noisy_b = ""
        #tag_noisy_a = "_noisy-20"
        #tag_noisy_b = "_noisy-10"
        
        plot_and_save_ims = True
        
        save_test_results = True 
        argv_a = [
            #"pf_pretrained_weights_ih=None",
            "pf_pretrained_weights_ih={}".format("[01LF] results/run-integration_{}/{}weights/hidden/".format(load_run_id, util.ND_NET_A)),
            
            #"pf_pretrained_weights_ho=None",
            "pf_pretrained_weights_ho={}".format("[01LF] results/run-integration_{}/{}weights/output/".format(load_run_id, util.ND_NET_A)),
            
            "load_essential_params_only=True",
            #"pf_pre_parameters=None",
            "pf_pre_parameters={}".format("[01LF] results/run-integration_{}/{}parameters.pkl".format(load_run_id, util.ND_NET_A)),
            
              "print_results_h=False",    "print_results_o=False", f"print_interval={print_plot_interval}", 
            "plot_weight_ims_h=False",  f"plot_weight_ims_o={plot_and_save_ims}",   f"plot_interval={print_plot_interval}",
            "save_weight_ims_h=False", "save_weight_ims_o=False", 
               "save_weights_h=False",    "save_weights_o=False",
            
            "learn_h=False", f"learn_o=False",
            
            "tag_mnist=-normal{}".format(tag_noisy_a),
            
            f"save_test_results={save_test_results}",
            ]
        argv_b = [
            #"pf_pretrained_weights_ih=None",
            "pf_pretrained_weights_ih={}".format("[01LF] results/run-integration_{}/{}weights/hidden/".format(load_run_id, util.ND_NET_B)),
            
            #"pf_pretrained_weights_ho=None",
            "pf_pretrained_weights_ho={}".format("[01LF] results/run-integration_{}/{}weights/output/".format(load_run_id, util.ND_NET_B)),
            
            "load_essential_params_only=True",
            #"pf_pre_parameters=None",
            "pf_pre_parameters={}".format("[01LF] results/run-integration_{}/{}parameters.pkl".format(load_run_id, util.ND_NET_B)),
            
              "print_results_h=False",    "print_results_o=False", f"print_interval={print_plot_interval}", 
            "plot_weight_ims_h=False",  f"plot_weight_ims_o={plot_and_save_ims}",   f"plot_interval={print_plot_interval}",
            "save_weight_ims_h=False", "save_weight_ims_o=False", 
               "save_weights_h=False",    "save_weights_o=False",
            
            "learn_h=False", f"learn_o=False",
            
            "tag_mnist=-shuffled{}".format(tag_noisy_b),
            
            f"save_test_results={save_test_results}",
            ]
        
        argv = [
            #"pf_pretrained_weights=None",
            "pf_pretrained_weights={}".format("[01LF] results/run-integration_{}/weights/weights_t{}.npy".format(load_run_id, load_run_t)),
            
            "load_essential_params_only=True",
            "pf_pre_parameters=None",
            "pf_pre_parameters={}".format("[01LF] results/run-integration_{}/parameters.pkl".format(load_run_id)),
            
              "print_results=False", f"print_interval={print_plot_interval}",
            f"plot_weight_ims={plot_and_save_ims}",   f"plot_interval={print_plot_interval}",
            "save_weight_ims=False",
               "save_weights=False",
                                "evaluation_interval=500",
 
            "learn=False",
            
            f"save_test_results={save_test_results}",
            ]
        run_integration(argv, argv_a, argv_b, train=False)
 
    
# =============================================================================
#     HIERARCHICAL - TRAIN & TEST
# =============================================================================  
    
    if task == "train-and-test-hierarchical":
        
        def get_default_params():
            
            K_h = "38"
            K_o = "99"
            K_f = "98"
            print_plot_interval = 5000
            save_weights_interval = 5000
            
            ms = 150
            hertz_h = hertz_o = 20/150*1000
            hertz_f = 10/150*1000
            t_f = 2
            t_s = 8
            
            topdown_enabled = False
            topdown_stdp_mult = 1
            params_td_curve = None
            
            learn_h = learn_o = "True"
            pft_pre_run = "None{}"
            nd_weights_h = nd_weights_o = nf_pre_parameters = load_run_id_a = load_run_id_b = ""
                
            plot_and_save_ims = False
            
            tag_mnist_a = "-normal"
            tag_mnist_b = "-shuffled"
            
            seed = 2022
            
            pd_results = "./[01LF] results/"  # IF NOT DEFAULT MAKE SURE TO ALTER CODE FOR TESTS ***
            
            return (K_h, K_o, K_f, print_plot_interval, save_weights_interval, 
                    ms, hertz_h, hertz_o, hertz_f, t_f, t_s,
                    topdown_enabled, topdown_stdp_mult, params_td_curve, 
                    learn_h, learn_o, pft_pre_run, 
                    nd_weights_h, nd_weights_o, nf_pre_parameters, load_run_id_a, load_run_id_b, 
                    plot_and_save_ims, 
                    tag_mnist_a, tag_mnist_b,
                    seed,
                    pd_results)
                
        params = param_manager.get_params_02(segment=-1, n_experiments=10)
        print(f"Retrieved {len(params)} dicts of custom parameters.")
        for param in params:
            
            (K_h, K_o, K_f, print_plot_interval, save_weights_interval, 
             ms, hertz_h, hertz_o, hertz_f, t_f, t_s,
             topdown_enabled, topdown_stdp_mult, params_td_curve, 
             learn_h, learn_o, pft_pre_run, 
             nd_weights_h, nd_weights_o, nf_pre_parameters, load_run_id_a, load_run_id_b, 
             plot_and_save_ims, 
             tag_mnist_a, tag_mnist_b,
             seed,
             pd_results) = get_default_params()
            
            print(f"\nparam = {param}\n")
            if 'topdown_enabled' in param.keys():
                topdown_enabled = param['topdown_enabled']
            if 'topdown_stdp_mult' in param.keys():
                topdown_stdp_mult = param['topdown_stdp_mult']
            if 'params_td_curve' in param.keys():
                params_td_curve = param['params_td_curve']
            if 'tag_mnist_a' in param.keys():
                tag_mnist_a = param['tag_mnist_a']
            if 'tag_mnist_b' in param.keys():
                tag_mnist_b = param['tag_mnist_b']
            if 'hertz_f' in param.keys():
                hertz_f = param['hertz_f']
            if 't_f' in param.keys():
                t_f = param['t_f']
            if 't_s' in param.keys():
                t_s = param['t_s']
            if 'ms' in param.keys():
                ms = param['ms']
            if 'seed' in param.keys():
                seed = param['seed']
            if 'pd_results' in param.keys(): # IF USED MAKE SURE TO ALTER CODE FOR TESTS ***
                pd_results = param['pd_results']
            
            argv = [
                "pf_pretrained_weights_ih={}{}".format(pft_pre_run.format(load_run_id_a), nd_weights_h),
                
                "pf_pretrained_weights_ho={}{}".format(pft_pre_run.format(load_run_id_a), nd_weights_o),
                
                "load_essential_params_only=True",
                "pf_pre_parameters={}{}".format(pft_pre_run.format(load_run_id_a), nf_pre_parameters),
                
                  "print_results_h=False",  "print_results_o=False",                                        f"print_interval={print_plot_interval}", 
                f"plot_weight_ims_h={plot_and_save_ims}", f"plot_weight_ims_o={plot_and_save_ims}",            f"plot_interval={print_plot_interval}",
                f"save_weight_ims_h={plot_and_save_ims}", f"save_weight_ims_o={plot_and_save_ims}", f"save_weight_ims_interval={print_plot_interval}", 
                   "save_weights_h=True",    "save_weights_o=True",                                   f"save_weights_interval={save_weights_interval}",
             
                f"K_h={K_h}", f"K_o={K_o}",
                f"learn_h={learn_h}", f"learn_o={learn_o}",
                
                f"tag_mnist={tag_mnist_a}",
                
                f"ms={ms}",
                f"hertz_h={hertz_h}", f"hertz_o={hertz_o}",
                f"t_f={t_f}", f"t_s={t_s}",
                
                f"topdown_enabled={topdown_enabled}",
                f"topdown_stdp_mult={topdown_stdp_mult}",
                f"params_td_curve={params_td_curve}",
                
                f"seed={seed}",
                
                f"pd_results={pd_results}",
                ]
            
            #train_and_test_integration((argv, argv_a, argv_b))
            time.sleep(0.5)
            p = multiprocessing.Process(target=train_and_test_hierarchical,args=[argv])
            p.start()
     
        
# =============================================================================
#     INTEGRATION - TRAIN & TEST
# =============================================================================  
    
    if task == "train-and-test-integration":
        
        def get_default_params():
            
            K_h = "38"
            K_o = "99"
            K_f = "98"
            print_plot_interval = 5000
            save_weights_interval = 5000
            
            ms = 150
            hertz_h = hertz_o = 20/150*1000
            hertz_f = 10/150*1000
            t_f = 2
            t_s = 8
            
            topdown_enabled = False
            topdown_stdp_mult = 1
            params_td_curve = None
            
            learn_h = learn_o = "True"
            pft_pre_run = "None{}"
            nd_weights_h = nd_weights_o = nf_pre_parameters = load_run_id_a = load_run_id_b = ""
                
            plot_and_save_ims = False
            
            tag_mnist_a = "-normal"
            tag_mnist_b = "-shuffled"
            
            seed = 2022
            
            pd_results = "./[01LF] results/"  # IF NOT DEFAULT MAKE SURE TO ALTER CODE FOR TESTS ***
            
            return (K_h, K_o, K_f, print_plot_interval, save_weights_interval, 
                    ms, hertz_h, hertz_o, hertz_f, t_f, t_s,
                    topdown_enabled, topdown_stdp_mult, params_td_curve, 
                    learn_h, learn_o, pft_pre_run, 
                    nd_weights_h, nd_weights_o, nf_pre_parameters, load_run_id_a, load_run_id_b, 
                    plot_and_save_ims, 
                    tag_mnist_a, tag_mnist_b,
                    seed,
                    pd_results)
                
        params = param_manager.get_params_00(segment=-1, n_experiments=10)
        print(f"Retrieved {len(params)} dicts of custom parameters.")
        for param in params:
            
            (K_h, K_o, K_f, print_plot_interval, save_weights_interval, 
             ms, hertz_h, hertz_o, hertz_f, t_f, t_s,
             topdown_enabled, topdown_stdp_mult, params_td_curve, 
             learn_h, learn_o, pft_pre_run, 
             nd_weights_h, nd_weights_o, nf_pre_parameters, load_run_id_a, load_run_id_b, 
             plot_and_save_ims, 
             tag_mnist_a, tag_mnist_b,
             seed,
             pd_results) = get_default_params()
            
            print(f"\nparam = {param}\n")
            if 'topdown_enabled' in param.keys():
                topdown_enabled = param['topdown_enabled']
            if 'topdown_stdp_mult' in param.keys():
                topdown_stdp_mult = param['topdown_stdp_mult']
            if 'params_td_curve' in param.keys():
                params_td_curve = param['params_td_curve']
            if 'tag_mnist_a' in param.keys():
                tag_mnist_a = param['tag_mnist_a']
            if 'tag_mnist_b' in param.keys():
                tag_mnist_b = param['tag_mnist_b']
            if 'hertz_f' in param.keys():
                hertz_f = param['hertz_f']
            if 't_f' in param.keys():
                t_f = param['t_f']
            if 't_s' in param.keys():
                t_s = param['t_s']
            if 'ms' in param.keys():
                ms = param['ms']
            if 'seed' in param.keys():
                seed = param['seed']
            if 'pd_results' in param.keys(): # IF USED MAKE SURE TO ALTER CODE FOR TESTS ***
                pd_results = param['pd_results']
            
            argv_a = [
                "pf_pretrained_weights_ih={}{}".format(pft_pre_run.format(load_run_id_a), nd_weights_h),
                
                "pf_pretrained_weights_ho={}{}".format(pft_pre_run.format(load_run_id_a), nd_weights_o),
                
                "load_essential_params_only=True",
                "pf_pre_parameters={}{}".format(pft_pre_run.format(load_run_id_a), nf_pre_parameters),
                
                  "print_results_h=False",  "print_results_o=False",                                        f"print_interval={print_plot_interval}", 
                f"plot_weight_ims_h={plot_and_save_ims}", f"plot_weight_ims_o={plot_and_save_ims}",            f"plot_interval={print_plot_interval}",
                f"save_weight_ims_h={plot_and_save_ims}", f"save_weight_ims_o={plot_and_save_ims}", f"save_weight_ims_interval={print_plot_interval}", 
                   "save_weights_h=True",    "save_weights_o=True",                                   f"save_weights_interval={save_weights_interval}",
             
                f"K_h={K_h}", f"K_o={K_o}",
                f"learn_h={learn_h}", f"learn_o={learn_o}",
                
                f"tag_mnist={tag_mnist_a}",
                
                f"ms={ms}",
                f"hertz_h={hertz_h}", f"hertz_o={hertz_o}",
                f"t_f={t_f}", f"t_s={t_s}",
                
                f"topdown_enabled={topdown_enabled}",
                f"topdown_stdp_mult={topdown_stdp_mult}",
                f"params_td_curve={params_td_curve}",
                
                f"seed={seed}",
                
                f"pd_results={pd_results}",
                ]
            argv_b = [
                "pf_pretrained_weights_ih={}{}".format(pft_pre_run.format(load_run_id_b), nd_weights_h),
                
                "pf_pretrained_weights_ho={}{}".format(pft_pre_run.format(load_run_id_b), nd_weights_o),
                
                "load_essential_params_only=True",
                "pf_pre_parameters={}{}".format(pft_pre_run.format(load_run_id_b), nf_pre_parameters),
                
                  "print_results_h=False",  "print_results_o=False",                                        f"print_interval={print_plot_interval}", 
                f"plot_weight_ims_h={plot_and_save_ims}", f"plot_weight_ims_o={plot_and_save_ims}",            f"plot_interval={print_plot_interval}",
                f"save_weight_ims_h={plot_and_save_ims}", f"save_weight_ims_o={plot_and_save_ims}", f"save_weight_ims_interval={print_plot_interval}", 
                   "save_weights_h=True",    "save_weights_o=True",                                   f"save_weights_interval={save_weights_interval}",
             
                f"K_h={K_h}", f"K_o={K_o}",
                f"learn_h={learn_h}", f"learn_o={learn_o}",
                
                f"tag_mnist={tag_mnist_b}",
                
                f"ms={ms}",
                f"hertz_h={hertz_h}", f"hertz_o={hertz_o}",
                f"t_f={t_f}", f"t_s={t_s}",
                
                f"topdown_enabled={topdown_enabled}",
                f"topdown_stdp_mult={topdown_stdp_mult}",
                f"params_td_curve={params_td_curve}",
                
                f"seed={seed}",
                
                f"pd_results={pd_results}",
                ]
            
            argv = [
                "pf_pretrained_weights=None",
                #"pf_pretrained_weights=,
                
                "load_essential_params_only=True",
                "pf_pre_parameters=None",
                #"pf_pre_parameters=,
                
                  "print_results=True",                          f"print_interval={print_plot_interval}",
                f"plot_weight_ims={plot_and_save_ims}",            f"plot_interval={print_plot_interval}",
                f"save_weight_ims={plot_and_save_ims}", f"save_weight_ims_interval={print_plot_interval}", 
                   "save_weights=True",                    f"save_weights_interval={save_weights_interval}",
                
                f"K={K_f}",
                
                f"ms={ms}",
                f"hertz_o={hertz_f}",
                f"t_f={t_f}", f"t_s={t_s}",
                
                f"topdown_enabled={topdown_enabled}",
                f"topdown_stdp_mult={topdown_stdp_mult}",
                f"params_td_curve={params_td_curve}",
                
                f"seed={seed}",
                
                f"pd_results={pd_results}",
                ]
            
            #train_and_test_integration((argv, argv_a, argv_b))
            time.sleep(0.5)
            p = multiprocessing.Process(target=train_and_test_integration,args=[(argv, argv_a, argv_b)])
            p.start()

if __name__ == "__main__":
    
    try:
        gc.collect() # Clear memory of unused items
        main(sys.argv[1:])
        
# =============================================================================
#         for i in range(0, 30):
#             time.sleep(0.5)
#             load_run_id = f"{i:04d}"
#             p = multiprocessing.Process(target=test_hierarchical,args=[load_run_id])
#             p.start()
# =============================================================================
        
    except KeyboardInterrupt:
        print("\n"+"-"*30+" Keyboard Interrupt "+"-"*30)
        gc.collect() # Clear memory of unused items
    except Exception:
        print("\n\n\n"+"!"*30 + " EXCEPTION " +"!"*30+"\n")
        print(traceback.format_exc())
        print("!"*30 + "!!!!!!!!!!!" +"!"*30+"\n\n\n")
        gc.collect() # Clear memory of unused items