import argparse
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets
from torch.utils.data import Subset
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
parser.add_argument('--results_path', type=str, default=None, required=True,
                help='Path where results are written into.')
parser.add_argument('--id_class', type=str, default='Trouser',
                help='In-distribution class in the CIFAR-10 datastet. By default the dog class is considered ID.')
parser.add_argument('--ood_class', type=str, default='Pants',
                help='Name of the class in OOD dataset corresponding to the id_class argument.')
parser.add_argument('--batch_size', type=int, default=32, required=False,
                help='Batch size.')
parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
parser.add_argument('--flow', choices=['RealNVP', 'Glow', 'RealNVPNewMask', 'RealNVPNewMask2', 'RealNVPSmall'], default="RealNVP", help='Flow model to use (default: RealNVP)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'

with open(os.path.join(args.results_path, 'command_density_detection.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')


################################## DATA ##################################
img_shape = (1, 28, 28)
D = int(np.prod(img_shape))
transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.FashionMNIST(root=args.data_path, train=True, transform=transform, download=True)
test_set = datasets.FashionMNIST(root=args.data_path, train=False, transform=transform, download=True)

train_indices = []
test_indices = []
id_train_indices = []
id_test_indices = []
train_id_idx = train_set.class_to_idx[args.id_class]
test_id_idx = test_set.class_to_idx[args.id_class]

for i in range(len(train_set)):
    current_class = train_set[i][1]
    if current_class != train_id_idx:
        train_indices.append(i)
    else:
        id_train_indices.append(i)

for i in range(len(test_set)):
    current_class = test_set[i][1]
    if current_class != test_id_idx:
        test_indices.append(i)
    else:
        id_test_indices.append(i)

id_train_dataset = Subset(train_set, id_train_indices)
id_test_dataset = Subset(test_set, id_test_indices)
ood_dataset = ConcatDataset([Subset(train_set, train_indices), Subset(test_set, test_indices)])
print("Length of training dataset: {}".format(len(id_train_dataset)))
print("Length of test dataset: {}".format(len(id_test_dataset)))

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

################################## MODEL ##################################
init_zeros = True
num_mid_channels = 64
num_blocks = 8
num_scales = 3
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
for x, _ in train_loader:
    x = x.to(device).requires_grad_(True)
    z = net(x).requires_grad_(True)
    sldj = net.module.logdet().requires_grad_(True)
    losses = loss_fn.likelihood(z, sldj=sldj, mean=False).requires_grad_(True) #the loss is just the likelihood here to compute the bpd metric.
    del z
    del sldj
    loss = losses.mean()
    gradient = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
    with torch.no_grad():
        flattened = torch.flatten(gradient.mean(dim=0), start_dim=-2, end_dim=-1)
        norm = torch.linalg.norm(flattened, ord=2, dim=1).sum(dim=0)
        del gradient
        del flattened
    results = os.path.join(args.results_path, 'results_llh_'+args.id_class+'_train_id.txt')
    with open(results, 'a') as file:
        file.write(str(loss.item())+'\n')
    results = os.path.join(args.results_path, 'results_norm_'+args.id_class+'_train_id.txt')
    with open(results, 'a') as file:
        file.write(str(norm.item())+'\n')
    torch.cuda.empty_cache()

for x, _ in test_loader:
    x = x.to(device).requires_grad_(True)
    z = net(x).requires_grad_(True)
    sldj = net.module.logdet().requires_grad_(True)
    losses = loss_fn.likelihood(z, sldj=sldj, mean=False).requires_grad_(True) #the loss is just the likelihood here to compute the bpd metric.
    del z
    del sldj
    loss = losses.mean()
    gradient = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
    with torch.no_grad():
        flattened = torch.flatten(gradient.mean(dim=0), start_dim=-2, end_dim=-1)
        norm = torch.linalg.norm(flattened, ord=2, dim=1).sum(dim=0)
        del gradient
        del flattened
    results = os.path.join(args.results_path, 'results_llh_'+args.id_class+'_test_id.txt')
    with open(results, 'a') as file:
        file.write(str(loss.item())+'\n')
    results = os.path.join(args.results_path, 'results_norm_'+args.id_class+'_test_id.txt')
    with open(results, 'a') as file:
        file.write(str(norm.item())+'\n')
    torch.cuda.empty_cache()

for x, _ in ood_loader:
    x = x.to(device).requires_grad_(True)
    z = net(x).requires_grad_(True)
    sldj = net.module.logdet().requires_grad_(True)
    losses = loss_fn.likelihood(z, sldj=sldj, mean=False).requires_grad_(True) #the loss is just the likelihood here to compute the bpd metric.
    del z
    del sldj
    loss = losses.mean()
    gradient = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
    with torch.no_grad():
        flattened = torch.flatten(gradient.mean(dim=0), start_dim=-2, end_dim=-1)
        norm = torch.linalg.norm(flattened, ord=2, dim=1).sum(dim=0)
        del gradient
        del flattened
    results = os.path.join(args.results_path, 'results_llh_not_'+args.id_class+'_ood.txt')
    with open(results, 'a') as file:
        file.write(str(loss.item())+'\n')
    results = os.path.join(args.results_path, 'results_norm_not_'+args.id_class+'_ood.txt')
    with open(results, 'a') as file:
        file.write(str(norm.item())+'\n')
    torch.cuda.empty_cache()
