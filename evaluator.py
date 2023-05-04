__author__ = "Otto van der Himst"
__version__ = "1.0"
__email__ = "otto.vanderhimst@ru.nl"

import sys
import util
from util import Parameters, save_img, load, normalize
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
from network import Network
import seaborn as sn
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def load_test_results(pd_run, P):
    
    if P.nd_test_results[:-1] in os.listdir(pd_run):
        pd_test_results = pd_run + P.nd_test_results
            
        try:
            print(f"    Loading <test_params> from <{pd_test_results+P.nf_test_parameters}> ...")
            test_params = util.load(pd_test_results+P.nf_test_parameters)
        except:
            print(f"Could not load <test_params>, terminating.")
            sys.exit()
            
        try:
            print(f"    Loading <test_labels> from <{pd_test_results+test_params.nf_test_labels}> ...")
            test_labels = np.load(pd_test_results+test_params.nf_test_labels)
        except:
            print(f"Could not load <test_labels>, terminating.")
            sys.exit()
            
        try:
            print(f"    Loading <neuron_label_counts> from <{pd_test_results+test_params.nf_neuron_label_counts}> ...")
            neuron_label_counts = np.load(pd_test_results+test_params.nf_neuron_label_counts)
        except:
            print(f"Could not load <neuron_label_counts>, terminating.")
            sys.exit()
            
        try:
            print(f"    Loading <nf_neuron_image_counts from> <{pd_test_results+test_params.nf_neuron_image_counts}> ...")
            neuron_image_counts = np.load(pd_test_results+test_params.nf_neuron_image_counts)
        except:
            print(f"Could not load <neuron_image_counts>, terminating.")
            sys.exit()
    else:
        print(f"Could not find {P.nd_test_results[:-1]}, terminating.")
        sys.exit()
    
    return test_params, test_labels, neuron_label_counts, neuron_image_counts

def evaluate_experiments(pd_experiments, nd_results, n_runs=10, n_nets=3, hierarchical_only=False):

    def evaluate_run(pd_run, P):
        test_params, test_labels, neuron_label_counts, neuron_image_counts = load_test_results(pd_run, P)
        test_results = Network.evaluate_results(None, None, neuron_label_counts, neuron_image_counts, test_labels, plot_cms=False)
        accuracy, cm_classifications, cm_spike_confidence, cm_confidence_error = test_results
        cms = np.transpose(np.dstack((cm_classifications, cm_spike_confidence, cm_confidence_error)), (2, 0, 1))
        
        cm_classifications_norm = normalize(cm_classifications, axis=0)
        mean_digit_accuracies = np.diagonal(cm_classifications_norm)
        mean_overall_accuracy = np.mean(mean_digit_accuracies)
        stdv_overall_accuracy = np.std(mean_digit_accuracies)
        accuracies = np.array(np.append([mean_overall_accuracy, stdv_overall_accuracy], mean_digit_accuracies), dtype=np.float32)
        
        cm_spike_confidence_norm = normalize(cm_spike_confidence, axis=0)
        mean_digit_confidences = np.diagonal(cm_spike_confidence_norm)
        mean_overall_confidence = np.mean(mean_digit_confidences)
        stdv_overall_confidence = np.std(mean_digit_confidences)
        confidences = np.array(np.append([mean_overall_confidence, stdv_overall_confidence], mean_digit_confidences), dtype=np.float32)
        
        mean_digit_errors = np.abs(cm_confidence_error).sum(axis=0)
        mean_overall_error = np.mean(mean_digit_errors)
        stdv_overall_error = np.std(mean_digit_errors)
        errors = np.array(np.append([mean_overall_error, stdv_overall_error], mean_digit_errors), dtype=np.float32)
        
        return cms, accuracies, confidences, errors
    
    P = Parameters()
    P.set_common_default()
    P.set_common_custom(P.params_default, [])
    
    for pd_experiment in pd_experiments:
        
        pd_results = pd_experiment+nd_results
        if not os.path.isdir(pd_results):
            os.mkdir(pd_results)
        
        all_cms = np.zeros((n_runs, n_nets, 3, 10, 10), dtype=np.float32)
        all_accuracies = np.zeros((n_runs, n_nets, 12), dtype=np.float32)
        all_confidences = np.zeros((n_runs, n_nets, 12), dtype=np.float32)
        all_errors = np.zeros((n_runs, n_nets, 12), dtype=np.float32)
        
        ith_run = 0
        nds_nets = [""] if hierarchical_only else ["net_a/", "net_b/", ""]
        for nd_run in os.listdir(pd_experiment):
            
            if hierarchical_only and not "run-hierarchical" in nd_run:
                continue
            elif not hierarchical_only and not "run-integration" in nd_run:
                continue
                
            pd_run = pd_experiment + nd_run + "/"
            print(f"\nProcessing run: <{pd_run}>")
            
            for ith_net, nd_net in enumerate(nds_nets):
                
                cms, accuracies, confidences, errors = evaluate_run(pd_run+nd_net, P)
                all_cms[ith_run, ith_net] = cms
                all_accuracies[ith_run, ith_net] = accuracies
                all_confidences[ith_run, ith_net] = confidences
                all_errors[ith_run, ith_net] = errors
                    
            ith_run += 1
        
        if ith_run != 10:
            print(f"Only found {ith_run} out of {n_runs} runs, terminating.")
            sys.exit()
        
        np.save(pd_results+"confusion_matrices.npy", all_cms)
        np.save(pd_results+"accuracies.npy", all_accuracies)
        np.save(pd_results+"confidences.npy", all_confidences)
        np.save(pd_results+"errors.npy", all_errors)

def save_figures_within(pd_experiments, nd_results, include_cms=True, mean_only=False):
    
    def save_cms(cms, pd_results, id_experiment, ith_run, id_net, id_net_mathcal):
        
        pd_figures_cms = pd_results+"figures_cms/"
        if not os.path.isdir(pd_figures_cms):
            os.mkdir(pd_figures_cms)
        
        xticks = yticks = [digit for digit in range(10)]
        normalized = [True, True, False]
        cm_name = ["cm-classifications_" + id_experiment + "-{:02d}"+f"_{id_net}",
                       "cm-confidences_" + id_experiment + "-{:02d}"+f"_{id_net}",
                            "cm-errors_" + id_experiment + "-{:02d}"+f"_{id_net}"]
        titles = [f"Confusion Matrix Classifications - {id_experiment}"+"-{:02d}"+" - " + id_net_mathcal, 
                      f"Confusion Matrix Confidences - {id_experiment}"+"-{:02d}"+" - " + id_net_mathcal,  
                           f"Confusion Matrix Errors - {id_experiment}"+"-{:02d}"+" - " + id_net_mathcal]
        xlabels = ["Network classification", "Dominant neuron", "Dominant neuron"]
        ylabels = ["True label", "Spike counts", "Spike counts"]
        vmins = [0, 0, -np.std(cms[2])]
        vmaxs = [100, 100, np.std(cms[2])]
        figsize = (10, 7)
        
        for ith_cm, cm in enumerate(cms):
            if normalized[ith_cm]:
                cm = normalize(cm, axis=0)
            
            df_cm = pd.DataFrame(cm, index=yticks, columns=xticks)
            plt.figure(figsize=figsize)
            
            #cmap = LinearSegmentedColormap.from_list('RedGreenRed', ['crimson', 'lime', 'crimson'])
            #s = sn.heatmap(df_cm, annot=True, vmin=vmin, vmax=vmax, cmap=cmap)
            s = sn.heatmap(df_cm, annot=True, vmin=vmins[ith_cm], vmax=vmaxs[ith_cm])
            
            print(titles)
            plt.title(titles[ith_cm].format(ith_cm))
            s.set(xlabel=xlabels[ith_cm], ylabel=ylabels[ith_cm])
            plt.savefig(pd_figures_cms+cm_name[ith_cm].format(ith_run))
            plt.close()
    
    def save_barchart(pd_results, nf_figure,
                      x_ticks, x_labels, x_label, 
                      y_means, y_stdvs, y_outs, y_ticks, y_min, y_max, y_label,
                      bar_width=0.28, figsize=FIGSIZE_WITHIN, n_nets=3):
        
        pd_figures_overall = pd_results+"figures_overall/"
        if not os.path.isdir(pd_figures_overall):
            os.mkdir(pd_figures_overall)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if n_nets == 3:
            bar_a = ax.bar(x_ticks-bar_width, y_means[0], bar_width, yerr=y_stdvs[0], label=r'$H_a$')
            bar_b = ax.bar(          x_ticks, y_means[1], bar_width, yerr=y_stdvs[1], label=r'$H_b$')
            bar_i = ax.bar(x_ticks+bar_width, y_means[2], bar_width, yerr=y_stdvs[2], label=r'$I$')
        else:
            bar = ax.bar(x_ticks, y_means[0], bar_width, yerr=y_stdvs[0], label='net')
        
        ax.set_xlabel(x_label)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.legend(loc="upper left", fontsize=FS_LEGEND)
        
        ax.set_ylabel(y_label)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_ylim([y_min, y_max])
        
        if n_nets == 3:
            ax.bar_label(bar_a, padding=3, fmt='%.2f')
            ax.bar_label(bar_b, padding=3, fmt='%.2f')
            ax.bar_label(bar_i, padding=3, fmt='%.2f')
        else:
            ax.bar_label(bar, padding=3, fmt='%.2f')

        fig.tight_layout()
        plt.show()
        fig.savefig(pd_figures_overall+nf_figure)
        plt.close()
    
    def do_within_experiment_comparison(pd_results, id_experiment,
                                        all_cms, all_accuracies, 
                                        all_confidences, all_errors):

        _, n_nets, _, _, _ = all_cms.shape
        
        if mean_only:
            all_accuracies = all_accuracies[:, :, 0]
            all_confidences = all_confidences[:, :, 0]
            all_errors = all_errors[:, :, 0]
        x_ticks = np.array([i for i in range(1)]) if mean_only else np.arange(len(all_accuracies)+1)
        x_labels= [""] if mean_only else ["Mean", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        x_label= "" if mean_only else "Digit"
        
        y_min_ac = 80
        y_max_ac = 100
        y_min_e = 0
        y_max_e = 20
        
        if mean_only:
            mean_accuracies = np.mean(all_accuracies, axis=0)
            stdv_accuracies = np.std(all_accuracies, axis=0)
        else:
            mean_accuracies = np.delete(np.mean(all_accuracies, axis=0), [1], axis=-1)
            stdv_accuracies = np.delete(np.std(all_accuracies, axis=0), [1], axis=-1)
        save_barchart(pd_results=pd_results, 
                       nf_figure=f"accuracies_{id_experiment}",
                         x_ticks=x_ticks, 
                        x_labels=x_labels,
                         x_label=x_label,
                         y_means=mean_accuracies,
                         y_stdvs=stdv_accuracies,
                          y_outs=None,
                         y_ticks=[i for i in range(0, 101, 10)],
                           y_min=y_min_ac,
                           y_max=y_max_ac,
                         y_label="Accuracy",
                         n_nets=n_nets)
             
        if mean_only:
            mean_confidences = np.mean(all_confidences, axis=0)
            stdv_confidences = np.std(all_confidences, axis=0)
        else:
            mean_confidences = np.delete(np.mean(all_confidences, axis=0), [1], axis=-1)
            stdv_confidences = np.delete(np.std(all_confidences, axis=0), [1], axis=-1)
        save_barchart(pd_results=pd_results, 
                       nf_figure=f"confidences_{id_experiment}",
                         x_ticks=x_ticks, 
                        x_labels=x_labels,
                         x_label=x_label,
                         y_means=mean_confidences,
                         y_stdvs=stdv_confidences,
                          y_outs=None,
                         y_ticks=[i for i in range(0, 101, 10)],
                           y_min=y_min_ac,
                           y_max=y_max_ac,
                         y_label="Confidence",
                         n_nets=n_nets)
             
        if mean_only:
            mean_errors = np.mean(all_errors, axis=0)
            stdv_errors = np.std(all_errors, axis=0)
        else:
            mean_errors = np.delete(np.mean(all_errors, axis=0), [1], axis=-1)
            stdv_errors = np.delete(np.std(all_errors, axis=0), [1], axis=-1)
        save_barchart(pd_results=pd_results, 
                       nf_figure=f"errors_{id_experiment}",
                         x_ticks=x_ticks, 
                        x_labels=x_labels,
                         x_label=x_label,
                         y_means=mean_errors,
                         y_stdvs=stdv_errors,
                          y_outs=None,
                         y_ticks=[i for i in range(0, 51, 10)],
                           y_min=y_min_e,
                           y_max=y_max_e,
                         y_label="Confidence error",
                         n_nets=n_nets)
        
        with open(pd_results+"results_overall.txt", 'w') as f:
            def array_to_string(array):
                return "\n".join(["    "+str(l) for l in array.tolist()])
            f.write(f"Mean accuracies:\n{array_to_string(mean_accuracies)}\n")
            f.write(f"Stdv accuracies:\n{array_to_string(stdv_accuracies)}\n\n")
            f.write(f"Mean confidences:\n{array_to_string(mean_confidences)}\n")
            f.write(f"Stdv confidences:\n{array_to_string(stdv_confidences)}\n\n")
            f.write(f"Mean confidence errors:\n{array_to_string(mean_errors)}\n")
            f.write(f"Stdv confidence errors:\n{array_to_string(stdv_errors)}\n\n")
    
    for pd_experiment in pd_experiments:
        
        id_experiment = f"{pd_experiment.split('/')[-2].split('_')[-1]}"
        pd_results = pd_experiment+nd_results
        
        all_cms = np.load(pd_results+"confusion_matrices.npy")
        all_accuracies = np.load(pd_results+"accuracies.npy")
        all_confidences = np.load(pd_results+"confidences.npy")
        all_errors = np.load(pd_results+"errors.npy")
        
        n_runs, n_nets, _, _, _ = all_cms.shape
        print(f"n_runs={n_runs}")
        print(f"n_nets={n_nets}")
        if include_cms:
            for ith_run in range(n_runs):
                for ith_net, (id_net, id_net_mathcal) in enumerate(
                        [('net-a', r'$H_a$'), ('net-b', r'$H_b$'), ('net-i', r'$I$')] if n_nets == 3 else ['net', r'$NET$']):
                    save_cms(all_cms[ith_run, ith_net], pd_results, id_experiment, ith_run, id_net, id_net_mathcal)
        
        do_within_experiment_comparison(pd_results, id_experiment,
                                        all_cms, all_accuracies, 
                                        all_confidences, all_errors)

def save_figures_between(pd_experiments, nd_results, nd_results_between, ith_net=2, mean_only=False):
    
    def save_barchart(pd_results, nf_figure,
                      x_ticks, x_labels, x_label, 
                      y_means, y_stdvs, y_outs, y_ticks, y_min, y_max, y_label,
                      bar_width=0.28, figsize=FIGSIZE_BETWEEN):
        
        if not os.path.isdir(pd_results):
            os.mkdir(pd_results)
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = []
        for i, (y_mean, y_stdv) in enumerate(zip(y_means, y_stdvs)):
            bar = ax.bar(x_ticks-bar_width*(n_experiments//2)+bar_width*i, 
                         y_mean, bar_width, yerr=y_stdv, label=ids_experiments[i])
            bars.append(bar)
        
        ax.set_xlabel(x_label)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.legend(loc="upper left", fontsize=FS_LEGEND)
        
        ax.set_ylabel(y_label)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_ylim([y_min, y_max])
        
        for bar in bars:
            ax.bar_label(bar, padding=3, fmt='%.2f')
    
        fig.tight_layout()
        plt.show()
        fig.savefig(pd_results+nf_figure)
        plt.close()
    
    n_experiments = len(pd_experiments)
    ids_experiments = []
    int_accuracy_means = []
    int_accuracy_stdvs = []
    int_confidence_means = []
    int_confidence_stdvs = []
    int_error_means = []
    int_error_stdvs = []
    
    for pd_experiment in pd_experiments:
        
        id_experiment = f"{pd_experiment.split('/')[-2].split('_')[-1]}"
        pd_results = pd_experiment+nd_results
        
        ids_experiments.append("Experiment "+id_experiment)
        
        int_accuracy_means.append(np.delete(np.mean(np.load(pd_results+"accuracies.npy"), axis=0)[ith_net], [1]))
        int_accuracy_stdvs.append(np.delete(np.std(np.load(pd_results+"accuracies.npy"), axis=0)[ith_net], [1]))
        
        int_confidence_means.append(np.delete(np.mean(np.load(pd_results+"confidences.npy"), axis=0)[ith_net], [1]))
        int_confidence_stdvs.append(np.delete(np.std(np.load(pd_results+"confidences.npy"), axis=0)[ith_net], [1]))
        
        int_error_means.append(np.delete(np.mean(np.load(pd_results+"errors.npy"), axis=0)[ith_net], [1]))
        int_error_stdvs.append(np.delete(np.std(np.load(pd_results+"errors.npy"), axis=0)[ith_net], [1]))
    
    ids_experiments = [r'$p\_no {-} td$', r'$p\_td {\times} 1$', r'$p\_td {\times} 2$', r'$p\_td {\times} 3$', r'$p\_td {\times} \phi$'] # ***
    
    if mean_only:
        int_accuracy_means = [x[0] for x in int_accuracy_means]
        int_accuracy_stdvs = [x[0] for x in int_accuracy_stdvs]
        int_confidence_means = [x[0] for x in int_confidence_means]
        int_confidence_stdvs = [x[0] for x in int_confidence_stdvs]
        int_error_means = [x[0] for x in int_error_means]
        int_error_stdvs = [x[0] for x in int_error_stdvs]
    x_ticks = np.array([i for i in range(1)]) if mean_only else np.array([i for i in range(len(int_accuracy_means[0]))])
    x_labels= [""] if mean_only else ["Mean", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# =============================================================================
#     x_ticks = np.array([i for i in range(3)])
#     x_labels= [r'$H_a$', r'$H_b$', r'$H_i$']
# =============================================================================
    x_label= "" if mean_only else "Digit"
    
    y_min_ac = 80
    y_max_ac = 100
    y_min_e = 0
    y_max_e = 20
    
    figname_affix = "_"+nd_results_between.split("_")[-1][:-1]
    combined_bar_width = 0.8
    save_barchart(pd_results="./[01LF] results/"+nd_results_between,  
                     nf_figure="between-experiment-accuracy"+figname_affix,
                       x_ticks=x_ticks, 
                      x_labels=x_labels,
                       x_label=x_label,
                       y_means=int_accuracy_means,
                       y_stdvs=int_accuracy_stdvs,
                        y_outs=None,
                       y_ticks=[i for i in range(0, 101, 10)],
                         y_min=y_min_ac,
                         y_max=y_max_ac,
                       y_label="Accuracy",
                       bar_width=combined_bar_width/n_experiments)
    
    save_barchart(pd_results="./[01LF] results/"+nd_results_between, 
                     nf_figure="between-experiment-confidence"+figname_affix,
                       x_ticks=x_ticks, 
                      x_labels=x_labels,
                       x_label=x_label,
                       y_means=int_confidence_means,
                       y_stdvs=int_confidence_stdvs,
                        y_outs=None,
                       y_ticks=[i for i in range(0, 101, 10)],
                         y_min=y_min_ac,
                         y_max=y_max_ac,
                       y_label="Confidence",
                       bar_width=combined_bar_width/n_experiments)
    
    save_barchart(pd_results="./[01LF] results/"+nd_results_between, 
                     nf_figure="between-experiment-error"+figname_affix,
                       x_ticks=x_ticks, 
                      x_labels=x_labels,
                       x_label=x_label,
                       y_means=int_error_means,
                       y_stdvs=int_error_stdvs,
                        y_outs=None,
                       y_ticks=[i for i in range(0, 51, 10)],
                         y_min=y_min_e,
                         y_max=y_max_e,
                       y_label="Confidence error",
                       bar_width=combined_bar_width/n_experiments)

def save_hierachical_comparison(pd_experiments_i, pd_experiments_h, nd_results, nd_results_between):
    
    def save_barchart(pd_results, nf_figure,
                      x_ticks, x_labels, x_label, 
                      y_means, y_stdvs, y_outs, y_ticks, y_min, y_max, y_label,
                      bar_width=0.28, figsize=(10, 5)):
        
        if not os.path.isdir(pd_results):
            os.mkdir(pd_results)
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = []
        
        ids_nets = [r'$H_a$', r'$H_b$', r'$H_x$']
        for i, (y_mean, y_stdv) in enumerate(zip(y_means, y_stdvs)):
            bar = ax.bar(x_ticks-bar_width*(n_experiments//2)+bar_width*(i+1), 
                         y_mean, bar_width, yerr=y_stdv, label=ids_nets[i])
            bars.append(bar)
        
        ax.set_xlabel(x_label)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.legend(loc="upper right", fontsize=FS_LEGEND)
        
        ax.set_ylabel(y_label)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_ylim([y_min, y_max])
        
        for bar in bars:
            ax.bar_label(bar, padding=3, fmt='%.2f')
    
        fig.tight_layout()
        plt.show()
        fig.savefig(pd_results+nf_figure)
        plt.close()
    
    n_experiments = len(pd_experiments_i)
    ids_experiments_i = []
    
    int_accuracy_means_a = []
    int_accuracy_stdvs_a = []
    int_confidence_means_a = []
    int_confidence_stdvs_a = []
    int_error_means_a = []
    int_error_stdvs_a = []
    
    int_accuracy_means_b = []
    int_accuracy_stdvs_b = []
    int_confidence_means_b = []
    int_confidence_stdvs_b = []
    int_error_means_b = []
    int_error_stdvs_b = []
    
    ids_experiments_h = []
    int_accuracy_means_h = []
    int_accuracy_stdvs_h = []
    int_confidence_means_h = []
    int_confidence_stdvs_h = []
    int_error_means_h = []
    int_error_stdvs_h = []
    
    for pd_experiment in pd_experiments_i:
        
        id_experiment = f"{pd_experiment.split('/')[-2].split('_')[-1]}"
        pd_results = pd_experiment+nd_results
        
        ids_experiments_i.append("Experiment "+id_experiment)
        
        int_accuracy_means_a.append(np.delete(np.mean(np.load(pd_results+"accuracies.npy"), axis=0)[0], [1]))
        int_accuracy_stdvs_a.append(np.delete(np.std(np.load(pd_results+"accuracies.npy"), axis=0)[0], [1]))
        int_accuracy_means_b.append(np.delete(np.mean(np.load(pd_results+"accuracies.npy"), axis=0)[1], [1]))
        int_accuracy_stdvs_b.append(np.delete(np.std(np.load(pd_results+"accuracies.npy"), axis=0)[1], [1]))
        
        int_confidence_means_a.append(np.delete(np.mean(np.load(pd_results+"confidences.npy"), axis=0)[0], [1]))
        int_confidence_stdvs_a.append(np.delete(np.std(np.load(pd_results+"confidences.npy"), axis=0)[0], [1]))
        int_confidence_means_b.append(np.delete(np.mean(np.load(pd_results+"confidences.npy"), axis=0)[1], [1]))
        int_confidence_stdvs_b.append(np.delete(np.std(np.load(pd_results+"confidences.npy"), axis=0)[1], [1]))
        
        int_error_means_a.append(np.delete(np.mean(np.load(pd_results+"errors.npy"), axis=0)[0], [1]))
        int_error_stdvs_a.append(np.delete(np.std(np.load(pd_results+"errors.npy"), axis=0)[0], [1]))
        int_error_means_b.append(np.delete(np.mean(np.load(pd_results+"errors.npy"), axis=0)[1], [1]))
        int_error_stdvs_b.append(np.delete(np.std(np.load(pd_results+"errors.npy"), axis=0)[1], [1]))
        
    for pd_experiment in pd_experiments_h:
        
        id_experiment = f"{pd_experiment.split('/')[-2].split('_')[-1]}"
        pd_results = pd_experiment+nd_results
        
        ids_experiments_h.append("Experiment "+id_experiment)
        
        int_accuracy_means_h.append(np.delete(np.mean(np.load(pd_results+"accuracies.npy"), axis=0)[0], [1]))
        int_accuracy_stdvs_h.append(np.delete(np.std(np.load(pd_results+"accuracies.npy"), axis=0)[0], [1]))
        
        int_confidence_means_h.append(np.delete(np.mean(np.load(pd_results+"confidences.npy"), axis=0)[0], [1]))
        int_confidence_stdvs_h.append(np.delete(np.std(np.load(pd_results+"confidences.npy"), axis=0)[0], [1]))
        
        int_error_means_h.append(np.delete(np.mean(np.load(pd_results+"errors.npy"), axis=0)[0], [1]))
        int_error_stdvs_h.append(np.delete(np.std(np.load(pd_results+"errors.npy"), axis=0)[0], [1]))
    
    int_accuracy_means_a = [x[0] for x in int_accuracy_means_a]
    int_accuracy_stdvs_a = [x[0] for x in int_accuracy_stdvs_a]
    int_confidence_means_a = [x[0] for x in int_confidence_means_a]
    int_confidence_stdvs_a = [x[0] for x in int_confidence_stdvs_a]
    int_error_means_a = [x[0] for x in int_error_means_a]
    int_error_stdvs_a = [x[0] for x in int_error_stdvs_a]
    
    int_accuracy_means_b = [x[0] for x in int_accuracy_means_b]
    int_accuracy_stdvs_b = [x[0] for x in int_accuracy_stdvs_b]
    int_confidence_means_b = [x[0] for x in int_confidence_means_b]
    int_confidence_stdvs_b = [x[0] for x in int_confidence_stdvs_b]
    int_error_means_b = [x[0] for x in int_error_means_b]
    int_error_stdvs_b = [x[0] for x in int_error_stdvs_b]
        
    int_accuracy_means_h = [x[0] for x in int_accuracy_means_h]
    int_accuracy_stdvs_h = [x[0] for x in int_accuracy_stdvs_h]
    int_confidence_means_h = [x[0] for x in int_confidence_means_h]
    int_confidence_stdvs_h = [x[0] for x in int_confidence_stdvs_h]
    int_error_means_h = [x[0] for x in int_error_means_h]
    int_error_stdvs_h = [x[0] for x in int_error_stdvs_h]
    
    int_accuracy_means = np.stack( (int_accuracy_means_a, int_accuracy_means_b, int_accuracy_means_h) )
    int_accuracy_stdvs = np.stack( (int_accuracy_stdvs_a, int_accuracy_stdvs_b, int_accuracy_stdvs_h) )
    
    int_confidence_means = np.stack( (int_confidence_means_a, int_confidence_means_b, int_confidence_means_h) )
    int_confidence_stdvs = np.stack( (int_confidence_stdvs_a, int_confidence_stdvs_b, int_confidence_stdvs_h) )
    
    int_error_means = np.stack( (int_error_means_a, int_error_means_b, int_error_means_h) )
    int_error_stdvs = np.stack( (int_error_stdvs_a, int_error_stdvs_b, int_error_stdvs_h) )
    
    x_ticks = np.array([i for i in range(n_experiments)])
    #x_labels= ["Experiment x", "Experiment a", "Experiment b", "Experiment c", "Experiment d"]
    x_labels = [r'$p\_no {-} td$', r'$p\_td {\times} 1$', r'$p\_td {\times} 2$', r'$p\_td {\times} 3$', r'$p\_td {\times} \phi$']
    x_label= ""
    
    y_min_ac = 80
    y_max_ac = 100
    y_min_e = 0
    y_max_e = 20
    
    figname_affix = "_"+nd_results_between.split("_")[-1][:-1]
    combined_bar_width = 0.8
    save_barchart(pd_results="./[01LF] results/"+nd_results_between,  
                     nf_figure="between-experiment-accuracy"+figname_affix,
                       x_ticks=x_ticks, 
                      x_labels=x_labels,
                       x_label=x_label,
                       y_means=int_accuracy_means,
                       y_stdvs=int_accuracy_stdvs,
                        y_outs=None,
                       y_ticks=[i for i in range(0, 101, 10)],
                         y_min=y_min_ac,
                         y_max=y_max_ac,
                       y_label="Accuracy",
                       bar_width=combined_bar_width/3)
    
    save_barchart(pd_results="./[01LF] results/"+nd_results_between, 
                     nf_figure="between-experiment-confidence"+figname_affix,
                       x_ticks=x_ticks, 
                      x_labels=x_labels,
                       x_label=x_label,
                       y_means=int_confidence_means,
                       y_stdvs=int_confidence_stdvs,
                        y_outs=None,
                       y_ticks=[i for i in range(0, 101, 10)],
                         y_min=y_min_ac,
                         y_max=y_max_ac,
                       y_label="Confidence",
                       bar_width=combined_bar_width/3)
    
    save_barchart(pd_results="./[01LF] results/"+nd_results_between, 
                     nf_figure="between-experiment-error"+figname_affix,
                       x_ticks=x_ticks, 
                      x_labels=x_labels,
                       x_label=x_label,
                       y_means=int_error_means,
                       y_stdvs=int_error_stdvs,
                        y_outs=None,
                       y_ticks=[i for i in range(0, 51, 10)],
                         y_min=y_min_e,
                         y_max=y_max_e,
                       y_label="Confidence error",
                       bar_width=combined_bar_width/3)
    

if __name__ == "__main__":

    nd_results = "test_results/"
    mean_only = True
    FIGSIZE_WITHIN = (5, 8) if mean_only else (14, 8)
    FIGSIZE_BETWEEN = (5, 8) if mean_only else (23, 6)
    FS_LEGEND = 13
        
    pd_experiments = ["./[01LF] results/experiment_1a/",
                      "./[01LF] results/experiment_2a/",
                      "./[01LF] results/experiment_2b/",
                      "./[01LF] results/experiment_2c/",
                      "./[01LF] results/experiment_2d/",
                      "./[01LF] results/experiment_2e/",
                      "./[01LF] results/experiment_2f/",]
    
    #evaluate_experiments(pd_experiments=pd_experiments, nd_results=nd_results)
    save_figures_within(pd_experiments=pd_experiments, nd_results=nd_results, include_cms=False, mean_only=mean_only)
                              

# =============================================================================
#     pd_experiments = ["./[01LF] results/experiment_1a/",
#                       "./[01LF] results/experiment_2a/",
#                       "./[01LF] results/experiment_2b/",
#                       "./[01LF] results/experiment_2c/",
#                       "./[01LF] results/experiment_2d/",]
#     save_figures_between(pd_experiments=pd_experiments, nd_results=nd_results, nd_results_between="test_results_e2-i/", ith_net=2, mean_only=mean_only)
# =============================================================================


# =============================================================================
#     pd_experiments = [
#                       "./[01LF] results/experiment_3x/",
#                       "./[01LF] results/experiment_3a/",
#                       "./[01LF] results/experiment_3b/",
#                       "./[01LF] results/experiment_3c/",
#                       "./[01LF] results/experiment_3d/",
#                       ]
#     
#     #evaluate_experiments(pd_experiments=pd_experiments, nd_results=nd_results, n_nets=1, hierarchical_only=True)
#     save_figures_within(pd_experiments=pd_experiments, nd_results=nd_results, include_cms=False, mean_only=mean_only)
#     save_figures_between(pd_experiments=pd_experiments, nd_results=nd_results, nd_results_between="test_results_e3/", ith_net=0, mean_only=mean_only)
# =============================================================================
    
    
# =============================================================================
#     pd_experiments = ["./[01LF] results/experiment_1a/",
#                       "./[01LF] results/experiment_2a/",
#                       "./[01LF] results/experiment_2b/",
#                       "./[01LF] results/experiment_2c/",
#                       "./[01LF] results/experiment_2d/",]
#     save_figures_between(pd_experiments=pd_experiments, nd_results=nd_results, nd_results_between="test_results_e2-ha/", ith_net=0, mean_only=mean_only)
#     save_figures_between(pd_experiments=pd_experiments, nd_results=nd_results, nd_results_between="test_results_e2-hb/", ith_net=1, mean_only=mean_only)
# =============================================================================

# =============================================================================
#     pd_experiments_i = [
#                       "./[01LF] results/experiment_1a/",
#                       "./[01LF] results/experiment_2a/",
#                       "./[01LF] results/experiment_2b/",
#                       "./[01LF] results/experiment_2c/",
#                       "./[01LF] results/experiment_2d/",
#                       ]
#     pd_experiments_h = [
#                       "./[01LF] results/experiment_3x/",
#                       "./[01LF] results/experiment_3a/",
#                       "./[01LF] results/experiment_3b/",
#                       "./[01LF] results/experiment_3c/",
#                       "./[01LF] results/experiment_3d/",
#                       ]
#     save_hierachical_comparison(pd_experiments_i, pd_experiments_h, nd_results, nd_results_between="test_results_e-h-comparison/")
# =============================================================================
