import os
import numpy as np
import torch.nn as nn
import torch


class FlowLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """

    def __init__(self, prior, alpha, img_shape, k=256):
        super().__init__()
        self.k = k
        self.prior = prior
        self.alpha = torch.tensor(alpha)
        self.img_shape = img_shape #shape of the input image
    
    def likelihood(self, z, sldj, y=None, mean=True):
        z = z.reshape((z.shape[0], -1))
        if y is not None:
            prior_ll = self.prior.log_prob(z, y)
        else:
            prior_ll = self.prior.log_prob(z)
        corrected_prior_ll = prior_ll - np.log(self.k) * np.prod(z.size()[1:]) 
        corrected_prior_ll = corrected_prior_ll

        ll = corrected_prior_ll + sldj
        nll = -ll.mean() if mean else -ll

        return nll
    
    def forward(self, x, net, results_path, norm_ord=2, y=None, mean=True):
        x = x.requires_grad_(True)
        z = net(x)
        sldj = net.module.logdet()
        nll = self.likelihood(z, sldj, y=None, mean=True)
        
        if len(x.shape) == 4: #RGB image, batch*C*H*W
        
            '''
            #grad, flatten, average over batch, sum over channels
            gradient = torch.autograd.grad(nll, x, create_graph=True, retain_graph=True)[0]
            flattened = torch.flatten(gradient, start_dim=-2, end_dim=-1)
            avg_norm = torch.linalg.norm(flattened, ord=2, dim=-1).mean(dim=0)
            norm = avg_norm.sum(dim=0)
            del gradient
            del flattened
            '''

            #grad, flatten over channel and height and width, norm, average over batch
            gradient = torch.autograd.grad(nll, x, create_graph=True, retain_graph=True)[0]
            flattened = torch.flatten(gradient, start_dim=-2, end_dim=-1)
            del gradient
            flattened = torch.flatten(flattened, start_dim=-2, end_dim=-1)
            norm = torch.linalg.norm(flattened, ord=2, dim=-1).mean(dim=0)
            del flattened

            '''
            #grad, flatten, sum over channels, norm, average over batch
            gradient = torch.autograd.grad(nll, x, create_graph=True, retain_graph=True)[0]
            flattened = torch.flatten(gradient, start_dim=-2, end_dim=-1)
            del gradient
            summed_over_channels = flattened.sum(dim=1)
            del flattened
            norm = torch.linalg.norm(summed_over_channels, ord=2, dim=-1).mean(dim=0)
            '''
        elif len(x.shape) == 3: #grey images, batch*H*W
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
            gradient = torch.autograd.grad(nll, x, create_graph=True, retain_graph=True)[0]
            flattened = torch.flatten(gradient, start_dim=-2, end_dim=-1)
            del gradient
            norm = torch.linalg.norm(flattened, ord=2, dim=-1).mean(dim=0)
        
        loss = nll + self.alpha*norm
        
        path = os.path.join(results_path, "intermediary_check.txt")
        with open(path, "a") as file:
            file.write("nll={} norm={}\n".format(nll, norm))
        
        return loss

'''    
    def div_KL(self, net, r, x):
        
        Computes the KL divergence between p(x) and p(x+r), p=learned distribution.
        -----------------------------INPUT-----------------------------
        -net: the INN model that computes the likelihood
        -x: the input batch data
        -r: the perturbation added to the term x
        -----------------------------OUTPUT-----------------------------
        -KL divergence between p(x) and p(x+r).
        
        z = net(x)
        sldj = net.module.logdet()
        nll_x = self.likelihood(z, sldj)
        nll_x_scaled = nll_x/self.img_shape
        p_x = torch.exp(-nll_x_scaled)
        print(" fake p_x = {} \n".format(p_x))
        p_x = torch.pow(p_x, self.img_shape)
        print(" p_x = {} ".format(p_x))

        y = x+r
        print(" y = {} ".format(y))
        z_pert = net(y)
        sldj_pert = net.module.logdet()
        nll_xr = self.likelihood(z_pert, sldj_pert)
        print("\n nll_xr = {} ".format(nll_xr))
        
        diff = nll_xr - nll_x
        D = torch.prod(p_x, -diff)

        return D.mean(dim=0)
    
    def power_iteration(self, net, x):
        
        Performs power iteration to find the best r that perturbs the network the most.
        -----------------------------INPUT-----------------------------
        -x: input batch data
        -net: network to perturb
        -K: number of iteration (default 1 because seemed to perform best in Virtual Adversarial Training)
        -----------------------------OUTPUT-----------------------------
        -r: best perturbation in radius epsilon around input data x that maximizes KL divergence between likelihood of the network at x and at x+r.

        r = torch.randn_like(x)
        print("shape of r = {} \n".format(r.shape))
        for i in range(self.nb_iter):
            D = self.div_KL(net, r, x)
            g = torch.grad.autograd(D, r, retain_graph=True)[0]
            print("\nshape of g = {} \n".format(g.shape))
            H = torch.grad.autograd(g, r, retain_graph=True)[0]
            print("\nshape of H = {} \n".format(H.shape))
            r = torch.prod(H, r)
            r = r/torch.linalg.norm(r, ord=2, dim=(-2,-1)).requires_grad_(True)
        return r

    def forward(self, x, net, results_path, y=None, norm_ord=None, mean=True):
        x = x.requires_grad_(True)
        print("\n x = {} \n".format(x))
        z = net(x)
        sldj = net.module.logdet()
        nll = self.likelihood(z, sldj, y=None, mean=True)

        r_best = self.power_iteration(net, x)
        D_KL = self.div_KL(net, r_best, x)
        
        loss = nll + self.alpha*D_KL
        #print("norm = {}\n".format(norm))
        #print("negative LLH = {}\n".format(nll))
        
        file = open(results_path+"intermediary_check.txt", "a")
        file.write("nll={} loss={}\n".format(nll, loss))
        file.close()
        
        return loss
'''