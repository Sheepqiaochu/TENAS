import pdb

import numpy as np
import torch


def recal_bn(network, xloader, recalbn, device):
    for m in network.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.fill_(0)
            m.running_var.data.fill_(0)
            m.num_batches_tracked.data.zero_()
            m.momentum = None
    network.train()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(xloader):
            if i >= recalbn: break
            inputs = inputs.cuda(device=device, non_blocking=True)
            _, _ = network(inputs)
    return network


def get_ntk_n(xloader, networks, UAP, recalbn=0, train_mode=False, num_batch=-1):
    device = torch.cuda.current_device()
    # if recalbn > 0:
    #     network = recal_bn(network, xloader, recalbn, device)
    #     if network_2 is not None:
    #         network_2 = recal_bn(network_2, xloader, recalbn, device)
    ntks = []
    ntks_p = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads = [[] for _ in range(len(networks))]
    grads_p = [[] for _ in range(len(networks))]

    for i, (inputs, targets) in enumerate(xloader):
        if 0 < num_batch <= i: break
        inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs_)):
                logit[_idx:_idx + 1].backward(torch.ones_like(logit[_idx:_idx + 1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                torch.cuda.empty_cache()

        # pdb.set_trace()
        # inputs_p = inputs*UAP.std_tensor+UAP.mean_tensor
        # inputs_p = inputs_p + UAP.uap
        # inputs_p = ((inputs_p-UAP.mean_tensor)/UAP.std_tensor).cuda(device=device, non_blocking=True)
        inputs_p = UAP(inputs)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            inputsp_ = inputs_p.clone().cuda(device=device, non_blocking=True)
            logit_p = network(inputsp_)
            if isinstance(logit_p, tuple):
                logit_p = logit_p[1]  # 201 networks: return features and logits
            for _idx in range(len(inputsp_)):
                logit[_idx:_idx + 1].backward(torch.ones_like(logit_p[_idx:_idx + 1]), retain_graph=True)
                grad_p = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad_p.append(W.grad.view(-1).detach())
                grads_p[net_idx].append(torch.cat(grad_p, -1))
                network.zero_grad()
                torch.cuda.empty_cache()


    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))


    grads_p = [torch.stack(_grads, 0) for _grads in grads_p]
    ntks_p = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads_p]     
    conds_p = []
    for ntk in ntks_p:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        conds_p.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
    

    # return [a-b for a,b in zip(conds,conds_p)]
    return np.sum([conds,conds_p],axis=0)