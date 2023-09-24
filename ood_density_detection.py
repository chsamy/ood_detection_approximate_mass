import argparse
import os
import sys
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch import distributions
from flow_ssl import FlowLoss
import utils
import matplotlib
matplotlib.use('Agg')
from torch.nn import functional as F
import numpy as np
import flow_ssl
from flow_ssl.data import make_sup_data_loaders


parser = argparse.ArgumentParser(description='RealNVP')
parser.add_argument('--model_path', type=str, default=None, required=True,
                help='Path to trained model.')
parser.add_argument('--data_path', type=str, default=None, required=True,
                help='Path to dataset.')
parser.add_argument('--ood_data_path', type=str, default=None, required=True,
                help='Path to dataset.')
parser.add_argument('--results_path', type=str, default=None, required=True,
                help='Path where results are written into.')
parser.add_argument('--batch_size', type=int, default=32, required=False,
                help='Batch size.')
parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
parser.add_argument('--flow', choices=['RealNVP', 'Glow', 'RealNVPNewMask', 'RealNVPNewMask2', 'RealNVPSmall'], default="RealNVP", help='Flow model to use (default: RealNVP)')
parser.add_argument('--dataset', type=str, default="CIFAR10", required=False, metavar='DATA',
                help='Dataset name (default: CIFAR10)')
parser.add_argument('--ood_dataset', type=str, default='SVHN', required=False, metavar='DATA',
                help='OOD dataset name (default: SVHN)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'

with open(os.path.join(args.results_path, 'command_density_detection.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')


################################## DATA ##################################
transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])

if args.dataset.lower() == "mnist" or args.dataset.lower() == "fashionmnist":
    img_shape = (1, 28, 28)
    num_blocks = 6
    num_scales = 2
elif args.dataset.lower() in ["cifar10", "svhn", "celeba"]:
    img_shape = (3, 32, 32)
    num_blocks = 8
    num_scales = 3

if args.ood_dataset.lower() not in ['sun', 'inaturalist', 'sun', 'places365', 'dtd']:
    trainloader, testloader, _ = make_sup_data_loaders(
            args.data_path, 
            args.batch_size, 
            8, 
            transform_train, 
            transform_test, 
            use_validation=False,
            shuffle_train=True,
            dataset=args.dataset.lower())

    ood_trainloader, ood_testloader, _ = make_sup_data_loaders(
            args.ood_data_path, 
            args.batch_size, 
            8, 
            transform_train, 
            transform_test, 
            use_validation=False,
            shuffle_train=True,
            dataset=args.ood_dataset.lower())
elif args.ood_dataset.lower() == 'inaturalist':
    train_set = torchvision.datasets.INaturalist(root=args.ood_data_path, transform=transform_train, version='2021_train', download=True)
    test_set = torchvision.datasets.INaturalist(root=args.ood_data_path, transform=transform_test, version='2021_val', download=True)
    ood_train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True
        )
    ood_test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True
        )
elif args.ood_dataset.lower() == 'sun':
    train_set = torchvision.datasets.SUN397(root=args.ood_data_path, transform=transform_train, download=True)
    test_set = torchvision.datasets.SUN397(root=args.ood_data_path, transform=transform_test, download=True)
    ood_train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True
        )
    ood_test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True
        )
elif args.ood_dataset.lower() == 'places365':
    train_set = torchvision.datasets.Places365(root=args.ood_data_path, transform=transform_train, split='train-standard', download=True)
    test_set = torchvision.datasets.Places365(root=args.ood_data_path, transform=transform_test, split='val', download=True)
    ood_train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True
        )
    ood_test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True
        )
elif args.ood_dataset.lower() == 'dtd':
    train_set = torchvision.datasets.DTD(root=args.ood_data_path, transform=transform_train, version='train_standard', download=True)
    test_set = torchvision.datasets.DTD(root=args.ood_data_path, transform=transform_test, version='val', download=True)
    ood_train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True
        )
    ood_test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True
        )

D = int(np.prod(img_shape))

################################## MODEL ##################################
init_zeros = True
num_mid_channels = 64
st_type = 'resnet'
batchnorm = True
skip = True
num_coupling_layers_per_scale = 8
no_multi_scale = True

print('Building {} model...'.format(args.flow))
model_cfg = getattr(flow_ssl, args.flow)
if 'RealNVP' in args.flow:
    net = model_cfg(in_channels=img_shape[0], init_zeros=init_zeros, mid_channels=num_mid_channels,
        num_blocks=num_blocks, num_scales=num_scales, st_type=st_type,
        use_batch_norm=batchnorm, img_shape=img_shape, skip=skip, latent_dim=img_shape[0])
elif args.flow == 'Glow':
    net = model_cfg(image_shape=img_shape, mid_channels=num_mid_channels, num_scales=num_scales,
        num_coupling_layers_per_scale=num_coupling_layers_per_scale, num_layers=3,
        multi_scale=not no_multi_scale, st_type=st_type)
print("Model contains {} parameters".format(sum([p.numel() for p in net.parameters()])))


net = torch.nn.DataParallel(net, args.gpu_ids)
cudnn.benchmark = True

# Load checkpoint.
print('Resuming from checkpoint at '+args.model_path+'...')
checkpoint = torch.load(args.model_path)
net.load_state_dict(checkpoint['net'], strict=True)

prior = distributions.MultivariateNormal(torch.zeros(D).to(device),torch.eye(D).to(device))
loss_fn = FlowLoss(prior, 0, device, D)


################################## TESTING LOOP ##################################
for x, _ in trainloader:
    x = x.to(device).requires_grad_(True)
    z = net(x).requires_grad_(True)
    sldj = net.module.logdet().requires_grad_(True)
    losses = loss_fn.likelihood(z, sldj=sldj, mean=False).requires_grad_(True) #the loss is just the likelihood here to compute the bpd metric.
    del z
    del sldj
    loss = losses.mean()
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
    results = os.path.join(args.results_path, 'results_llh_'+args.dataset+'_train_id.txt')
    with open(results, 'a') as file:
        file.write(str(loss.item())+'\n')
    results = os.path.join(args.results_path, 'results_norm_'+args.dataset+'_train_id.txt')
    with open(results, 'a') as file:
        file.write(str(norm.item())+'\n')
    torch.cuda.empty_cache()

for x, _ in testloader:
    x = x.to(device).requires_grad_(True)
    z = net(x).requires_grad_(True)
    sldj = net.module.logdet().requires_grad_(True)
    losses = loss_fn.likelihood(z, sldj=sldj, mean=False).requires_grad_(True) #the loss is just the likelihood here to compute the bpd metric.
    del z
    del sldj
    loss = losses.mean()
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
    results = os.path.join(args.results_path, 'results_llh_'+args.dataset+'_test_id.txt')
    with open(results, 'a') as file:
        file.write(str(loss.item())+'\n')
    results = os.path.join(args.results_path, 'results_norm_'+args.dataset+'_test_id.txt')
    with open(results, 'a') as file:
        file.write(str(norm.item())+'\n')
    torch.cuda.empty_cache()

for x, _ in ood_trainloader:
    x = x.to(device).requires_grad_(True)
    z = net(x).requires_grad_(True)
    sldj = net.module.logdet().requires_grad_(True)
    losses = loss_fn.likelihood(z, sldj=sldj, mean=False).requires_grad_(True) #the loss is just the likelihood here to compute the bpd metric.
    del z
    del sldj
    loss = losses.mean()
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
    results = os.path.join(args.results_path, 'results_llh_not_'+args.dataset+'_ood.txt')
    with open(results, 'a') as file:
        file.write(str(loss.item())+'\n')
    results = os.path.join(args.results_path, 'results_norm_not_'+args.dataset+'_ood.txt')
    with open(results, 'a') as file:
        file.write(str(norm.item())+'\n')
    torch.cuda.empty_cache()

for x, _ in ood_testloader:
    x = x.to(device).requires_grad_(True)
    z = net(x).requires_grad_(True)
    sldj = net.module.logdet().requires_grad_(True)
    losses = loss_fn.likelihood(z, sldj=sldj, mean=False).requires_grad_(True) #the loss is just the likelihood here to compute the bpd metric.
    del z
    del sldj
    loss = losses.mean()
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
    results = os.path.join(args.results_path, 'results_llh_not_'+args.dataset+'_ood.txt')
    with open(results, 'a') as file:
        file.write(str(loss.item())+'\n')
    results = os.path.join(args.results_path, 'results_norm_not_'+args.dataset+'_ood.txt')
    with open(results, 'a') as file:
        file.write(str(norm.item())+'\n')
    torch.cuda.empty_cache()
