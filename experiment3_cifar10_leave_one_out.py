'''
Testing experiment 3 (section 4.3): get average scores for different values of alpha on the trained model.
Here: extract raw data from ood_density_detection.py runs (folders density_ood_vs_id) and get average accuracy, AUROC score, AUPR score, TNR at 95% TPR.
1) iterate over several models' results
2)for each model, deduce threshold values for classification
3)for each model get the list of grad norms
4)from the list of grad norms, create labels OOD and ID and compute AUROC, AUPR,...
5)plot sucessively on same graph
'''
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
import argparse
import os
import stat
import torchvision.transforms as transforms
import utils
import numpy as np
import torch
from torch.nn import functional as F
from flow_ssl.data import make_sup_data_loaders
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')


def compute_tnr_at_95_tpr(targets, scores):
    '''
    Computes the TNR @ 95% TPR (true negative rate at a fixed 95% true positive rate).
    -----------------------------------------------------------------------------
    Inputs: -targets: target labels to predict (array-like)
            -preds: predictions output by the classifier (array-like)
    -----------------------------------------------------------------------------
    Outputs: TNR @ 95% TPR
    '''
    fpr, tpr, _ = roc_curve(targets, scores, pos_label=1) #get false positive rae for varying levels of true positive rate
    tnr = 1-fpr #compute true negative rate
    return np.interp(0.95, tpr, tnr) #interpolate value of true negative rate at the true positive rate level of 0.95

parser = argparse.ArgumentParser(description='RealNVP')
parser.add_argument('--input_folder', type=str, default='/local2/is148265/sc264857/sc264857/flows_ood/experiments/density_ood_vs_id_alpha', required=False, metavar='PATH',
                help='path to results (with in-distribution results) (default: /local2/is148265/sc264857/sc264857/flows_ood/experiments/plots_xp3/density_ood_vs_id_alpha) (default: None)')
parser.add_argument('--output_folder', type=str, default='/local2/is148265/sc264857/sc264857/flows_ood/experiments/plots_xp3', required=False, metavar='PATH',
                help='path to results (e.g.: /local2/is148265/sc264857/sc264857/flows_ood/experiments/plots_xp3) (default: None)')
#parser.add_argument('--quantile', type=float, default=0.99, required=False, metavar='PATH',
#                help='Quantile order for the computation of the threshold value of the gradient (a fraction args.quantile is under this value in the training set). Used during classification.')
parser.add_argument('--id_class', type=str, default='dog', required=False, help='In-distribution class in the CIFAR-10 datastet. By default the dog class is considered ID.')

args = parser.parse_args()

path_norm_train_id = os.path.join(args.input_folder, 'results_norm_'+args.id_class+'_train_id.txt')
path_norm_test_id = os.path.join(args.input_folder, 'results_norm_'+args.id_class+'_test_id.txt')
path_norm_ood = os.path.join(args.input_folder, 'results_norm_not_'+args.id_class+'_ood.txt')

max_grad = 0 #find the maximum value of density score the model outputs on the train set
number_lines = 0
train_grad_norm_list = []
with open(path_norm_train_id, 'r') as file:
    for line in file.readlines():
        train_grad_norm_list.append(float(line))

###############MEASURE FROM THE RAW DATA ON BOTH ID AND OOD DATA THE AUROC SCORE###############
ood_grad_norm_list = []
id_grad_norm_list = []

with open(path_norm_test_id, 'r') as file:
    for line in file.readlines():
        id_grad_norm_list.append(float(line))

with open(path_norm_ood, 'r') as file:
    for line in file.readlines():
        ood_grad_norm_list.append(float(line))

nb_overlap = 0
nb_elements = 0
min_ood = np.min(np.array(ood_grad_norm_list))
max_id = np.max(np.array(ood_grad_norm_list))
for grad in id_grad_norm_list:
    if grad > min_ood:
        nb_overlap += 1
    nb_elements += 1
for grad in ood_grad_norm_list:
    if grad < max_id:
        nb_overlap += 1
    nb_elements += 1
overlap = nb_overlap/nb_elements

n_ood, n_test = len(ood_grad_norm_list), len(id_grad_norm_list)
max_kappa = -1

scores = np.hstack([np.array(ood_grad_norm_list), np.array(id_grad_norm_list)])
targets = np.ones((n_ood + n_test,), dtype=int)
targets[:n_ood] = 0
auroc_score = roc_auc_score(targets, -scores)
precision, recall, thresholds = precision_recall_curve(targets, -scores)
aupr_score = auc(recall, precision)
res_path = os.path.join(args.output_folder, 'scores_'+args.id_class+'_out.txt')
tnr95tpr = compute_tnr_at_95_tpr(targets, -scores)
with open(res_path,'a') as file:
    file.write('Results on CIFAR-10 for model trained on '+args.id_class+': \nauroc = {}\naupr = {}\ntrue negative rate at 95% true positive rate = {}\n'.format(auroc_score, aupr_score, tnr95tpr))

prec_path = os.path.join(args.output_folder, "precision.txt")
with open(prec_path, 'w') as file:
    np.savetxt(file, np.array(precision))
recall_path = os.path.join(args.output_folder, "recall.txt")
with open(recall_path, 'w') as file:
    np.savetxt(file, np.array(recall))

for beta in np.arange(0.55, 1, 0.01):
    max_grad = np.quantile(train_grad_norm_list, q=beta)
    id_grad_norm_list = torch.tensor(id_grad_norm_list)
    ood_grad_norm_list = torch.tensor(ood_grad_norm_list)
    threshold_tensor_ood = torch.ones_like(ood_grad_norm_list) * max_grad
    threshold_tensor_id = torch.ones_like(id_grad_norm_list) * max_grad


    # SCORES
    preds = np.hstack([torch.le(ood_grad_norm_list, threshold_tensor_ood).long(), torch.le(id_grad_norm_list, threshold_tensor_id).long()])
    acc = accuracy_score(targets, preds)
    kappa_accuracy = kappa(targets, preds)
    f1 = f1_score(targets, preds)

    with open(res_path,'a') as file:
        file.write('Results on CIFAR-10 for model trained on '+args.id_class+':\nGradient threshold value = {} at {} quantile\naccuracy = {}\nkappa accuracy = {}\nF1 score = {}\nOverlap = {}\n'
        .format(max_grad, beta, acc, kappa_accuracy, f1, overlap))
    
    if max_kappa < kappa_accuracy:
        best_beta = beta
        max_kappa = kappa_accuracy

print("Best kappa accuracy returned {} for beta = {}".format(max_kappa, best_beta))
