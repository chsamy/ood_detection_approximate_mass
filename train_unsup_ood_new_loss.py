"""
Code adapted from https://github.com/chrischute/real-nvp, which is in turn
adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import os
import sys
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist
from torch.nn import functional as F

import flow_ssl
from flow_ssl import FlowLoss
from tqdm import tqdm
from torch import distributions
import torch.nn as nn

#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from flow_ssl.data import make_sup_data_loaders
from sklearn.metrics import roc_auc_score


def linear_rampup(final_value, epoch, num_epochs, start_epoch=0):
    t = (epoch - start_epoch + 1) / num_epochs
    if t > 1:
        t = 1.
    return t * final_value


def draw_samples(net, writer, loss_fn, num_samples, device, img_shape, iter):
    images = utils.sample(net, loss_fn.prior, num_samples,
                          cls=None, device=device, sample_shape=img_shape)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    writer.add_image("samples/unsup", images_concat, iter)
    return images_concat

                    
def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm, writer, results_path, norm_ord,
        num_samples=10, sampling=True, tb_freq=100):
    #print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = utils.AverageMeter()
    iter_count = 0
    batch_count = 0
    for x, _ in trainloader:
        #torch.cuda.empty_cache()
        iter_count += 1
        batch_count += x.size(0)
        x = x.to(device)
        optimizer.zero_grad()
        loss = loss_fn(x, net, results_path, norm_ord)
        loss.backward()
        utils.clip_grad_norm(optimizer, max_grad_norm)
        optimizer.step()

        loss_meter.update(loss.item(), x.size(0))

        if iter_count % tb_freq == 0 or batch_count == len(trainloader.dataset):
            tb_step = epoch*(len(trainloader.dataset))+batch_count
            writer.add_scalar("train/loss", loss_meter.avg, tb_step)
            writer.add_scalar("train/bpd", utils.bits_per_dim(x, loss_meter.avg), tb_step)
            if sampling:
                net.eval()
                draw_samples(net, writer, loss_fn, num_samples, device, tuple(x[0].shape), tb_step)
                net.train()

   
def test(epoch, net, testloader, device, loss_fn, writer, results_path, ood=False, 
        tb_name="test"):
    torch.cuda.empty_cache()
    net.eval()
    loss_meter = utils.AverageMeter()
    loss_std_meter = utils.AverageMeter()
    grad_norm_meter = utils.AverageMeter()
    loss_list = []
    grad_norm_list = []

    for x, _ in testloader:
        x = x.to(device).requires_grad_(True)
        z = net(x).requires_grad_(True)
        sldj = net.module.logdet().requires_grad_(True)
        losses = loss_fn.likelihood(z, sldj=sldj, mean=False).requires_grad_(True)
        loss_list.extend([loss.item() for loss in losses])
        del z
        del sldj

        loss = losses.mean()
        loss_std = losses.std(unbiased=True)
        loss_meter.update(loss.item(), x.size(0))
        loss_std_meter.update(loss_std.item(), x.size(0))

        with torch.no_grad():
            if len(x.shape) == 4: #RGB image, batch*C*H*W
                '''
                #grad, flatten, average over batch, sum over channels
                gradient = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
                flattened = torch.flatten(gradient, start_dim=-2, end_dim=-1)
                avg_norm = torch.linalg.norm(flattened, ord=2, dim=-1).mean(dim=0)
                norm = avg_norm.sum(dim=0)
                del gradient
                del flattened
                '''

                #grad, flatten over channel and height and width, norm, average over batch
                gradient = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
                flattened = torch.flatten(gradient, start_dim=-2, end_dim=-1)
                del gradient
                flattened = torch.flatten(flattened, start_dim=-2, end_dim=-1)
                norm = torch.linalg.norm(flattened, ord=2, dim=-1).mean(dim=0)
                del flattened

                '''
                #grad, flatten, sum over channels, norm, average over batch
                gradient = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
                flattened = torch.flatten(gradient, start_dim=-2, end_dim=-1)
                del gradient
                summed_over_channels = flattened.sum(dim=1)
                del flattened
                norm = torch.linalg.norm(summed_over_channels, ord=2, dim=-1).mean(dim=0)
                '''
            elif len(x.shape) == 3: #grey images, batch*H*W
                '''
                #grad, flatten, average over batch, sum over channels
                gradient = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
                flattened = torch.flatten(gradient, start_dim=-2, end_dim=-1)
                avg_norm = torch.linalg.norm(flattened, ord=2, dim=-1).mean(dim=0)
                norm = avg_norm.sum(dim=0)
                del gradient
                del flattened
                '''
                #grad, flatten, norm, average over batch
                gradient = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
                flattened = torch.flatten(gradient, start_dim=-2, end_dim=-1)
                del gradient
                norm = torch.linalg.norm(flattened, ord=2, dim=-1).mean(dim=0)

            grad_norm_list.append(norm)
            grad_norm_meter.update(norm, x.size(0))
    
    with torch.no_grad():
        likelihoods = -torch.from_numpy(np.array(loss_list)).float()
        grad_norms = torch.Tensor(grad_norm_list).float()

    if writer is not None:
        writer.add_scalar("{}/loss".format(tb_name), loss_meter.avg, epoch)
        writer.add_scalar("{}/std on bpd".format(tb_name), loss_std_meter.avg, epoch)
        writer.add_scalar("{}/bpd".format(tb_name), utils.bits_per_dim(x, loss_meter.avg), epoch)
        writer.add_scalar("{}/gradient norm".format(tb_name), grad_norm_meter.avg, epoch)
        writer.add_histogram('{}/likelihoods'.format(tb_name), likelihoods, epoch)
    
    if ood:
        s = "out of distribution"
    else:
        s = "in distribution"
    path = os.path.join(results_path, "results.txt")
    file = open(path, "a")
    file.write("Epoch {} on {} data: bits-per-dimension={} loss={} std-on-loss={} grad norm = {}\n".format(epoch, s, utils.bits_per_dim(x, loss_meter.avg), loss_meter.avg, utils.bits_per_dim(x,loss_std_meter.avg), grad_norm_meter.avg))
    #file.write("Epoch {} on {} data: bits-per-dimension={} loss={} std-on-loss={} grad norm = {}\n".format(epoch, s, utils.bits_per_dim(x, loss_meter.avg), loss_meter.avg, utils.bits_per_dim(x,loss_std_meter.avg), 0))
    file.close()
    torch.cuda.empty_cache()
    
    return likelihoods, grad_norms
    #return likelihoods, []

def get_percentile(arr, p=0.05):
    percentile_idx = int(len(arr) * 0.05)
    percentile = torch.sort(arr)[0][percentile_idx].item()
    return percentile

def get_distribution_nll(net, testloader, ood_testloader, device, loss_fn):
    net.eval()
    ood_bpd_list = []
    id_bpd_list = []
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)
            z = net(x)
            sldj = net.module.logdet()
            nlls = loss_fn.likelihood(z, sldj, mean=False)
            for sample_nll in nlls:
                id_bpd_list.append(sample_nll)
        
        for x, _ in ood_testloader:
            x = x.to(device)
            z = net(x)
            sldj = net.module.logdet()
            nlls = loss_fn.likelihood(z, sldj, mean=False)
            for sample_nll in nlls:
                ood_bpd_list.append(sample_nll)
    return id_bpd_list, ood_bpd_list

def draw_ood_id_histogram(net, testloader, ood_testloader, device, loss_fn, results_path):
    import pandas as pd
    id_bpd_list, ood_bpd_list = get_distribution_nll(net, testloader, ood_testloader, device, loss_fn)
    id_df = pd.DataFrame(id_bpd_list, columns="nll")
    id_df["type"] = "in-distribution"
    ood_df = pd.DataFrame(ood_bpd_list, columns="nll")
    ood_df["type"] = "out-of-distribution"
    id_vs_ood = pd.concat([id_df, ood_df])
    
    title = results_path + "histogram_id_vs_ood.png"
    sns.histplot(data=id_vs_ood, x="nll", hue="type", binwidth=50)
    plt.savefig(title, bbox_inches="tight")

def get_average_gradient(net, testloader, device, loss_fn):
    grad_norm_list = []
    torch.cuda.empty_cache()
    for x, _ in testloader:
        x = x.requires_grad_(True).to(device)
        z = net(x)
        sldj = net.module.logdet()
        nll = loss_fn.likelihood(z, sldj, y=None, mean=True)

        gradient = torch.autograd.grad(nll, x, create_graph=True, retain_graph=True)[0]
        averaged_gradient = gradient.mean(dim=0) #collapse the penalization on the batch dimension by averaging values of gradient
        
        flattened = torch.flatten(averaged_gradient, start_dim=-2, end_dim=-1).requires_grad_(True)
        prev_norm = torch.linalg.norm(flattened, ord=2, dim=1).requires_grad_(True)
        norm = prev_norm.sum(dim=0)
        grad_norm_list.append(norm)
    torch.cuda.empty_cache()
    return grad_norm_list

def draw_grad_histogram(net, testloader, device, loss_fn, results_path):
    torch.cuda.empty_cache()
    grad_norm_list = get_average_gradient(net, testloader, device, loss_fn)
    torch.cuda.empty_cache()
    title = results_path + "histogram_model_smoothness.png"
    sns.histplot(data=grad_norm_list, bins=20)
    plt.savefig(title, bbox_inches="tight")

def draw_grad_histogram_ID_vs_OOD(net, testloader, ood_testloader, device, loss_fn, results_path):
    '''
    Draws likelihood density histogram for OOD set and ID set
    OOD samples should have lower density compared to ID samples (cf.: JEM paper on th method of OOD detection and Nalisnick OOD detection with typical set)
    '''
    torch.cuda.empty_cache()
    grad_norm_list_ID = get_average_gradient(net, testloader, device, loss_fn)
    grad_norm_list_OOD = get_average_gradient(net, ood_testloader, device, loss_fn)
    torch.cuda.empty_cache()
    
    fig, ax = plt.subplots()
    for a in [grad_norm_list_ID, grad_norm_list_OOD]:
        sns.distplot(a, bins=range(1, 110, 10), ax=ax, kde=False)
    ax.set_xlim([0, 100])
    title = results_path + "histogram_OOD_vs_ID_density.png"
    plt.savefig(title, bbox_inches="tight")



parser = argparse.ArgumentParser(description='RealNVP')

parser.add_argument('--dataset', type=str, default="CIFAR10", required=True, metavar='DATA',
                help='Dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                help='path to datasets location (default: None)')
parser.add_argument('--alpha', type=float, default=0.1,
                help='Coefficient of penalization in the loss function.')
'''
parser.add_argument('--xi', type=float, default=0.1,
                help='Discretization step in the finite difference scheme.')
parser.add_argument('--epsilon', type=float, default=0.5,
                help='Radius of the attack of the perturbation in the divergence term in the penalization.')
parser.add_argument('--nb_iter', type=float, default=3,
                help='Number of iterations in the power iteration algorithm.')
'''
parser.add_argument('--prior', type=str, default='normal',
                help='Prior distribution of the latent space.')
parser.add_argument('--results_path', type=str, default=None, required=True,
                help='Path where results are written into.')
parser.add_argument('--ood_dataset', type=str, default=None, required=False, metavar='DATA',
                help='OOD dataset name (default: None)')
parser.add_argument('--ood_data_path', type=str, default=None, required=False, metavar='PATH',
                help='path to ood datasets location (default: None)')

parser.add_argument('--logdir', type=str, default=None, required=True, metavar='PATH',
                help='path to log directory (default: None)')
parser.add_argument('--norm_ord', default=2,
                help='type of norm computed (cf.:pytorch linalg.norm documentation.')

parser.add_argument('--ckptdir', type=str, default=None, required=True, metavar='PATH',
                help='path to ckpt directory (default: None)')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
parser.add_argument('--num_samples', default=20, type=int, help='Number of samples at test time')
parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
parser.add_argument('--resume',  type=str, default=None, metavar='PATH', help='path to ckpt')
parser.add_argument('--weight_decay', default=5e-5, type=float,
                    help='L2 regularization (only applied to the weight norm scale factors)')
parser.add_argument('--use_validation', action='store_true', help='Use trainable validation set')
parser.add_argument('--save_freq', default=25, type=int, 
                    help='frequency of saving ckpts')
parser.add_argument('--lr_anneal', action='store_true')


parser.add_argument('--flow', choices=['RealNVP', 'Glow', 'RealNVPNewMask', 'RealNVPNewMask2', 'RealNVPSmall'], default="RealNVP", help='Flow model to use (default: RealNVP)')
#parser.add_argument('--flow', type=str, default="RealNVP", help='Flow model to use (default: RealNVP)')
parser.add_argument('--num_blocks', default=8, type=int, help='number of blocks in ResNet or number of layers in st-net in other models')
parser.add_argument('--num_scales', default=2, type=int, help='number of scales in multi-layer architecture')
parser.add_argument('--num_mid_channels', default=64, type=int, help='number of channels in coupling layer parametrizing network')
parser.add_argument('--st_type', choices=['highway', 'resnet', 'convnet', 'autoencoder_old', 'autoencoder', 'resnet_ae'], default='resnet')
parser.add_argument('--latent_dim', default=100, type=int, help='dim of bottleneck in autoencoder st-network')
parser.add_argument('--batchnorm', action='store_true')
parser.add_argument('--skip', action='store_true')
parser.add_argument('--aug', action='store_true')
parser.add_argument('--init_zeros', action='store_true')
parser.add_argument('--optim', choices=['Adam', 'RMSprop', 'AdamW'], default='Adam')

# Glow parameters
parser.add_argument('--num_coupling_layers_per_scale', default=8, type=int, help='number of coupling layers in one scale')
parser.add_argument('--no_multi_scale', action='store_true')

# for RealNVPTabular model
parser.add_argument('--no_sampling', action='store_true')
parser.add_argument('--dropout', action='store_true')


args = parser.parse_args()

os.makedirs(args.results_path, exist_ok=True)
os.makedirs(args.logdir, exist_ok=True)
os.makedirs(args.ckptdir, exist_ok=True)

def schedule(epoch):
    t = (epoch) / args.num_epochs
    if t <= 0.8:
        factor = 1.0
    elif t <=0.9:
        factor = 0.5
    else:
        factor = 0.5 ** 2
    return args.lr * factor

with open(os.path.join(args.ckptdir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

writer = SummaryWriter(log_dir=args.logdir)

device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
#device = torch.device('cuda', 2) if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
#print("\ndevice: {}\n".format(device))
start_epoch = 0

if args.dataset.lower() == "mnist" or args.dataset.lower() == "fashionmnist":
    img_shape = (1, 28, 28)
    if args.aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])

elif args.dataset.lower() in ["cifar10", "svhn", "celeba"]:  # celeba has its own train_transform in make_sup_data_loaders
    img_shape = (3, 32, 32)
    if args.aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])

elif args.dataset.lower() == "imagenet":
    img_shape = (3, 32, 32)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])

elif args.dataset.lower() == 'numpy':
    transform_train = None

elif args.dataset.lower() in ["cifar10_transfer", "svhn_transfer", "celeba_transfer"]:  # features from pretrained EfficientNet
    feature_dim = 1792
    transform_train = transforms.Compose([transforms.ToTensor()])

else:
    raise ValueError("Unsupported dataset "+args.dataset)

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainloader, testloader, _ = make_sup_data_loaders(
        args.data_path, 
        args.batch_size, 
        args.num_workers, 
        transform_train, 
        transform_test, 
        use_validation=args.use_validation,
        shuffle_train=True,
        dataset=args.dataset.lower())

if args.ood_dataset:
    _, ood_testloader, _ = make_sup_data_loaders(
            args.ood_data_path, 
            args.batch_size, 
            args.num_workers, 
            transform_train, 
            transform_test, 
            use_validation=args.use_validation,
            shuffle_train=True,
            dataset=args.ood_dataset.lower())

if args.dataset.lower() == 'numpy':
    img_shape = testloader.dataset[0][0].shape
    print(img_shape)

# Model
print('Building {} model...'.format(args.flow))
model_cfg = getattr(flow_ssl, args.flow)
if args.flow == 'RealNVPTabular':
    net = model_cfg(in_dim=feature_dim, hidden_dim=args.num_mid_channels, num_layers=args.num_blocks,
                    num_coupling_layers=args.num_coupling_layers_per_scale, init_zeros=args.init_zeros, dropout=args.dropout)

elif 'RealNVP' in args.flow:
    net = model_cfg(in_channels=img_shape[0], init_zeros=args.init_zeros, mid_channels=args.num_mid_channels,
        num_blocks=args.num_blocks, num_scales=args.num_scales, st_type=args.st_type,
        use_batch_norm=args.batchnorm, img_shape=img_shape, skip=args.skip, latent_dim=img_shape[0])
        #use_batch_norm=args.batchnorm, img_shape=img_shape, skip=args.skip, latent_dim=args.latent_dim)

elif args.flow == 'Glow':
    net = model_cfg(image_shape=img_shape, mid_channels=args.num_mid_channels, num_scales=args.num_scales,
        num_coupling_layers_per_scale=args.num_coupling_layers_per_scale, num_layers=3,
        multi_scale=not args.no_multi_scale, st_type=args.st_type)
print("Model contains {} parameters".format(sum([p.numel() for p in net.parameters()])))


if device == 'cuda':
#if device != 'cpu':
    net = torch.nn.DataParallel(net, args.gpu_ids)
    cudnn.benchmark = True #args.benchmark

if args.resume is not None:
    # Load checkpoint.
    print('Resuming from checkpoint at ckpts/best.pth.tar...')
    #assert os.path.isdir('ckpt'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']

D = np.prod(img_shape) if args.flow != 'RealNVPTabular' else feature_dim
D = int(D)

if args.prior == 'normal':
    prior = distributions.MultivariateNormal(torch.zeros(D).to(device),
                                                 torch.eye(D).to(device))
elif args.prior == 'student':
    typ = torch.float
    df = torch.tensor([2.0]).type(typ).to(device)
    prior = torch.distributions.StudentT(df)

loss_fn = FlowLoss(prior, args.alpha, device, D) #prior, alpha, device, img_shape, epsilon (radius of attack), xi (discretization step), nb_iter (in the power iteration algo)

if 'RealNVP' in args.flow and args.flow != 'RealNVPTabular':
    # We need this to make sure that weight decay is only applied to g -- norm parameter in Weight Normalization
    param_groups = utils.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    if args.optim == 'Adam':
        optimizer = optim.Adam(param_groups, lr=args.lr)
    else:
        optimizer = optim.RMSprop(param_groups, lr=args.lr)

elif args.flow == 'Glow' or args.flow == 'RealNVPTabular':
    if args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.num_epochs, eta_min=1e-6)

for epoch in range(start_epoch, start_epoch + args.num_epochs + 1):
    if args.lr_anneal:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)

    train(epoch=epoch, net=net, trainloader=trainloader, device=device, optimizer=optimizer, loss_fn=loss_fn, max_grad_norm=args.max_grad_norm, writer=writer, results_path=args.results_path, norm_ord=args.norm_ord, num_samples=args.num_samples)
    #scheduler.step()
    test_ll, test_grad_norm_list = test(epoch, net, testloader, device, loss_fn, writer, results_path=args.results_path)

    test_ll_percentile = get_percentile(test_ll)
    test_ll = test_ll.cpu().detach().numpy()
    
    test_grad_norm_list_percentile = get_percentile(test_grad_norm_list)
    test_grad_norm_list = test_grad_norm_list.cpu().detach().numpy()
    

    if args.ood_dataset:
        #print("\n ood set")
        ood_ll, ood_grad_norm_list = test(epoch, net, ood_testloader, device, loss_fn, writer, results_path=args.results_path, ood=True, tb_name="ood")
        ood_ll_percentile = get_percentile(ood_ll)
        ood_ll = ood_ll.cpu().detach().numpy()
        
        ood_grad_norm_list_percentile = get_percentile(ood_grad_norm_list)
        ood_grad_norm_list = ood_grad_norm_list.cpu().detach().numpy()
        

        # AUC-ROC
        n_ood, n_test = len(ood_ll), len(test_ll)
        lls = np.hstack([ood_ll, test_ll])
        targets = np.ones((n_ood + n_test,), dtype=int)
        targets[:n_ood] = 0
        score = roc_auc_score(targets, lls)
        writer.add_scalar("ood/roc_auc", score, epoch)

    # plotting likelihood hists
    fig, ax = plt.subplots()
    plt.hist(test_ll[test_ll > test_ll_percentile], color='r', label='In-distribution', alpha=0.5)    
    plt.hist(ood_ll[ood_ll > ood_ll_percentile], color='b', label='Out-of-distribution', alpha=0.5)
    title = args.results_path+"/likelihood_histogram_"+str(epoch)+".png"
    plt.savefig(title, bbox_inches="tight")
    hist_img = torch.tensor(np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep=''))
    hist_img = torch.tensor(hist_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))).transpose(0, 2).transpose(1, 2)
    writer.add_image("ll_hist", hist_img, epoch)

    # plotting norm of the likelihood gradient histograms
    
    fig, ax = plt.subplots()
    plt.hist(test_grad_norm_list[test_grad_norm_list > test_grad_norm_list_percentile], color='r', label='In-distribution', alpha=0.5, bins=50)    
    plt.hist(ood_grad_norm_list[ood_grad_norm_list > ood_grad_norm_list_percentile], color='b', label='Out-of-distribution', alpha=0.5, bins=50)
    title = args.results_path+"/likelihood_grad_norm_histogram_"+str(epoch)+".png"
    plt.savefig(title, bbox_inches="tight")
    grad_hist_img = torch.tensor(np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep=''))
    grad_hist_img = torch.tensor(grad_hist_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))).transpose(0, 2).transpose(1, 2)
    writer.add_image("ll_grad_norm_hist", grad_hist_img, epoch)


    # Save checkpoint
    if (epoch % args.save_freq == 0):
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
        os.makedirs(args.ckptdir, exist_ok=True)
        torch.save(state, os.path.join(args.ckptdir, str(epoch)+'.pt'))

    if not args.no_sampling:
        # Save samples and data
        os.makedirs(os.path.join(args.ckptdir, 'samples'), exist_ok=True)
        images_concat = draw_samples(net, writer, loss_fn, args.num_samples, device, img_shape, epoch*len(trainloader.dataset))
        os.makedirs(args.ckptdir, exist_ok=True)
        torchvision.utils.save_image(images_concat,
                                     os.path.join(args.ckptdir, 'samples/epoch_{}.png'.format(epoch)))