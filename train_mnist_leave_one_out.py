from torch.utils.data import ConcatDataset
from torchvision import datasets
from torch.utils.data import Subset
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
from matplotlib.lines import Line2D
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
from torch.utils.tensorboard import SummaryWriter
from flow_ssl.data import make_sup_data_loaders
from sklearn.metrics import roc_auc_score

#############################FUNCTION DEFINITION#############################
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

def schedule(epoch):
    t = (epoch) / args.num_epochs
    if t <= 0.8:
        factor = 1.0
    elif t <=0.9:
        factor = 0.5
    else:
        factor = 0.5 ** 2
    return args.lr * factor

def get_percentile(arr, p=0.05):
    percentile_idx = int(len(arr) * 0.05)
    percentile = torch.sort(arr)[0][percentile_idx].item()
    return percentile

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig("gradient_flow_bird.png")
                    
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
        losses = loss_fn.likelihood(z, sldj=sldj, mean=False).requires_grad_(True) #the loss is just the likelihood here to compute the bpd metric.
        loss_list.extend([loss.item() for loss in losses])
        del z
        del sldj

        loss = losses.mean()
        loss_std = losses.std(unbiased=True)
        loss_meter.update(loss.item(), x.size(0))
        loss_std_meter.update(loss_std.item(), x.size(0))

        gradient = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
        with torch.no_grad():
            '''
            #grad, flatten, average over batch, sum over channels
            gradient = torch.autograd.grad(nll, x, create_graph=True, retain_graph=True)[0]
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
        likelihoods = -torch.Tensor(loss_list).float()
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
    file.close()
    torch.cuda.empty_cache()
    
    return likelihoods, grad_norms


#############################ARGUMENT PARSING#############################
parser = argparse.ArgumentParser(description='RealNVP')
parser.add_argument('--dataset', type=str, default="FashionMNIST", required=True, metavar='DATA',
                help='Dataset name (default: FashionMNIST)')
parser.add_argument('--alpha', type=float, default=0.1,
                help='Coefficient of penalization in the loss function.')
parser.add_argument('--id_class', type=str, default='Pullover',
                help='In-distribution class in the FashionMNIST datastet. By default the pullover class is considered ID.')
parser.add_argument('--results_path', type=str, default=None, required=True,
                help='Path where results are written into.')
parser.add_argument('--logdir', type=str, default=None, required=True, metavar='PATH',
                help='path to log directory (default: None)')
parser.add_argument('--ckptdir', type=str, default=None, required=True, metavar='PATH',
                help='path to ckpt directory (default: None)')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
parser.add_argument('--lr', default=5e-5, type=float, help='Learning rate')
parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
parser.add_argument('--num_epochs', default=80, type=int, help='Number of epochs to train')
parser.add_argument('--num_samples', default=20, type=int, help='Number of samples at test time')
parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
parser.add_argument('--resume',  type=str, default=None, metavar='PATH', help='path to ckpt')
parser.add_argument('--weight_decay', default=5e-5, type=float,
                    help='L2 regularization (only applied to the weight norm scale factors)')
parser.add_argument('--use_validation', action='store_true', help='Use trainable validation set')
parser.add_argument('--save_freq', default=20, type=int, 
                    help='frequency of saving ckpts')
parser.add_argument('--lr_anneal', action='store_true')


parser.add_argument('--flow', choices=['RealNVP', 'Glow', 'RealNVPNewMask', 'RealNVPNewMask2', 'RealNVPSmall'], default="RealNVP", help='Flow model to use (default: RealNVP)')
parser.add_argument('--num_blocks', default=6, type=int, help='number of blocks in ResNet or number of layers in st-net in other models')
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

with open(os.path.join(args.ckptdir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

#############################DATA PROCESSING AND LOADING#############################
img_shape = (1, 28, 28)

if args.dataset.lower() == 'fashionmnist':
    dataset = 'FashionMNIST'
    data_path = '/local2/is148265/sc264857/sc264857/torch/data/FashionMNIST/FashionMNIST/processed/'

    transform_train = transforms.Compose([transforms.RandomCrop(28, padding=4),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=45),
                                        transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.FashionMNIST(root=data_path, train=True, transform=transform_train, download=True)
    test_set = datasets.FashionMNIST(root=data_path, train=False, transform=transform_test, download=True)

    train_ood_idx = train_set.class_to_idx[args.id_class]
    test_ood_idx = test_set.class_to_idx[args.id_class]

    train_indices = []
    test_indices = []
    ood_train_indices = []
    ood_test_indices = []

    for i in range(len(train_set)):
        current_class = train_set[i][1]
        if current_class == train_ood_idx:
            train_indices.append(i)
        else:
            ood_train_indices.append(i)

    for i in range(len(test_set)):
        current_class = test_set[i][1]
        if current_class == test_ood_idx:
            test_indices.append(i)
        else:
            ood_test_indices.append(i)

elif args.dataset.lower()=='mnist':
    dataset = 'MNIST'
    data_path = '/local2/is148265/sc264857/sc264857/torch/data/MNIST/MNIST/processed/'

    transform_train = transforms.Compose([transforms.RandomCrop(28, padding=4),
                                        transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(root=data_path, train=True, transform=transform_train, download=False)
    test_set = datasets.MNIST(root=data_path, train=False, transform=transform_test, download=False)

    train_indices = []
    test_indices = []
    ood_train_indices = []
    ood_test_indices = []

    for i in range(len(train_set)):
        current_class = train_set[i][1]
        if current_class == int(args.id_class):
            train_indices.append(i)
        else:
            ood_train_indices.append(i)

    for i in range(len(test_set)):
        current_class = test_set[i][1]
        if current_class == int(args.id_class):
            test_indices.append(i)
        else:
            ood_test_indices.append(i)

id_train_dataset = Subset(train_set, train_indices)
id_test_dataset = Subset(test_set, test_indices)
ood_dataset = ConcatDataset([Subset(train_set, ood_train_indices), Subset(test_set, ood_test_indices)])
print("Length of training dataset: {}".format(len(id_train_dataset)))

train_loader = torch.utils.data.DataLoader(
                id_train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True
        )
test_loader = torch.utils.data.DataLoader(
                id_test_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True
        )

ood_loader = torch.utils.data.DataLoader(
                ood_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True
        )

writer = SummaryWriter(log_dir=args.logdir)


#############################MODEL DEFINITION#############################
device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
start_epoch = 0

print('Building {} model...'.format(args.flow))
model_cfg = getattr(flow_ssl, args.flow)
if 'RealNVP' in args.flow:
    net = model_cfg(in_channels=img_shape[0], init_zeros=args.init_zeros, mid_channels=args.num_mid_channels,
        num_blocks=args.num_blocks, num_scales=args.num_scales, st_type=args.st_type,
        use_batch_norm=args.batchnorm, img_shape=img_shape, skip=args.skip, latent_dim=img_shape[0])

elif args.flow == 'Glow':
    net = model_cfg(image_shape=img_shape, mid_channels=args.num_mid_channels, num_scales=args.num_scales,
        num_coupling_layers_per_scale=args.num_coupling_layers_per_scale, num_layers=3,
        multi_scale=not args.no_multi_scale, st_type=args.st_type)
print("Model contains {} parameters".format(sum([p.numel() for p in net.parameters()])))


if device == 'cuda':
    net = torch.nn.DataParallel(net, args.gpu_ids)
    cudnn.benchmark = True #args.benchmark

if args.resume is not None:
    # Load checkpoint.
    print('Resuming from checkpoint at ckpts/best.pth.tar...')
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']

D = np.prod(img_shape)
D = int(D)

#############################LOSS DEFINITION#############################
prior = distributions.MultivariateNormal(torch.zeros(D).to(device), torch.eye(D).to(device))
loss_fn = FlowLoss(prior, args.alpha, device, D)

#############################OPTIMIZER INITIALIZATION#############################
if 'RealNVP' in args.flow and args.flow != 'RealNVPTabular':
    # We need this to make sure that weight decay is only applied to g -- norm parameter in Weight Normalization
    param_groups = utils.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    if args.optim == 'Adam':
        optimizer = optim.Adam(param_groups, lr=args.lr)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(param_groups, lr=args.lr)
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


#############################TRAINING LOOP#############################
for epoch in range(start_epoch, start_epoch + args.num_epochs + 1):
    
    if args.lr_anneal:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    

    train(epoch=epoch, net=net, trainloader=train_loader, device=device, optimizer=optimizer, loss_fn=loss_fn, max_grad_norm=args.max_grad_norm, writer=writer, results_path=args.results_path, norm_ord=2, num_samples=args.num_samples)
    test_ll, test_grad_norm_list = test(epoch, net, test_loader, device, loss_fn, writer, results_path=args.results_path)

    test_ll_percentile = get_percentile(test_ll)
    test_ll = test_ll.cpu().detach().numpy()
    test_grad_norm_list_percentile = get_percentile(test_grad_norm_list)
    test_grad_norm_list = test_grad_norm_list.cpu().detach().numpy()

    #print("\n ood set")
    ood_ll, ood_grad_norm_list = test(epoch, net, ood_loader, device, loss_fn, writer, results_path=args.results_path, ood=True, tb_name="ood")
    ood_ll_percentile = get_percentile(ood_ll)
    ood_ll = ood_ll.cpu().detach().numpy()
    ood_grad_norm_list_percentile = get_percentile(ood_grad_norm_list)
    ood_grad_norm_list = ood_grad_norm_list.cpu().detach().numpy()

    # plotting likelihood hists
    fig, ax = plt.subplots()
    plt.hist(test_ll[test_ll > test_ll_percentile], color='r', label='In-distribution', alpha=0.5, density=True, log=True)    
    plt.hist(ood_ll[ood_ll > ood_ll_percentile], color='b', label='Out-of-distribution', alpha=0.5, density=True, log=True)
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
        images_concat = draw_samples(net, writer, loss_fn, args.num_samples, device, img_shape, epoch*len(train_loader.dataset))
        os.makedirs(args.ckptdir, exist_ok=True)
        torchvision.utils.save_image(images_concat,
                                     os.path.join(args.ckptdir, 'samples/epoch_{}.png'.format(epoch)))