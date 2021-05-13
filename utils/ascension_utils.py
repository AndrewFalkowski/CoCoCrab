# ascension hub utils

import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import pandas as pd
from crabnet.neokingcrab import CrabNet
from crabnet.neomodel import Model
from utils.get_compute_device import get_compute_device

compute_device = get_compute_device(prefer_last=True)

class AscendedCrab():
    def __init__(self, src, prop0, prop1, saving=False, ensemble=False, compute_device=compute_device):
        self.src = src
        self.prop0 = prop0
        self.prop1 = prop1
        self.saving = saving
        self.ensemble=False
        self.compute_device = compute_device


    def ascend(self, epochs=100):
        data = rf'data/materials_data/{self.prop0}/test.csv'
        model_0_name = self.prop0
        model_0 = Model(CrabNet(compute_device=compute_device).to(compute_device), 
                        model_name=f'{model_0_name}', verbose=False)
        model_0.load_network(f'{self.prop0}.pth')
        model_0.load_data(data, batch_size=2**9, train=False)
        
        if not self.prop1 == 'Loss':
            model_1_name = self.prop1
            model_1 = Model(CrabNet(compute_device=compute_device).to(compute_device), 
                        model_name=f'{model_1_name}', verbose=False)
            model_1.load_network(f'{self.prop1}.pth')
            model_1.load_data(data, batch_size=2**9, train=False)

        num_elems = int(self.src.shape[1])
        frac = torch.ones(num_elems).view(1,-1)
        
        delim = '-'
        print(f'\n\nOptimizing {delim.join(elem_lookup(self.src))} System over {epochs} Epochs... \n'.title())

        losses = []
        srcs = []
        fracs = []

        prop0_preds = []
        prop0_uncs = []
        if not self.prop1 == 'Loss':
            prop1_preds = []
            prop1_uncs = []

        self.src = self.src.to(compute_device,
                dtype=torch.long,
                non_blocking=True)
        
        frac_mask = torch.where(self.src != 0, 1, 0)
        frac_mask = frac_mask.to(compute_device, 
                              dtype=torch.float,
                              non_blocking=True)
        
        frac = frac.to(compute_device,
                dtype=torch.float,
                non_blocking=True)
        
        optim_lr = 0.025
        optimizer = optim.Adam([frac.requires_grad_()], lr=optim_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [epochs-20], gamma=0.1, last_epoch=-1, verbose=False)
        criterion = nn.L1Loss()
        criterion = criterion.to(compute_device)
        
        for epoch in tqdm(range(epochs)):
            # print(f'frac: {frac}')
            soft_frac = masked_softmax(frac, frac_mask)
            
            optimizer.zero_grad()
            prop0_pred, prop0_unc = model_0.predict(self.src, soft_frac)
            if not self.prop1 =='Loss':
                prop1_pred, prop1_unc = model_1.predict(self.src, soft_frac)
            
            # factor = torch.tensor(bpred+epred).to(compute_device)
            loss = criterion(prop0_pred, torch.tensor([[100000.0]]).to(compute_device))
            # loss = loss + criterion(epred, torch.tensor([10000.0]).to(compute_device))
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
            losses.append(loss.item())
            srcs.append(self.src)
            fracs.append(soft_frac)
            prop0_preds.append(prop0_pred.item())
            prop0_uncs.append(prop0_unc.item())
            if not self.prop1 == 'Loss':
                prop1_preds.append(prop1_pred.item())
                prop1_uncs.append(prop1_unc.item())
                
        srcs = [elem_lookup(item.detach().numpy().reshape(-1)) for item in srcs]
        fracs = [item.detach().numpy().reshape(-1) for item in fracs]
        
        optimized_frac_df = pd.DataFrame(
            {'Elements': srcs,
             'Fractions': fracs,
               'Prop 0': prop0_preds,
               'Prop 0 UNC': prop0_uncs,
              'Loss': losses
            })
        
        print('\n-----------------------------------------------------')
        print('\nOptimized Fractional Composition:\n'.title())
        print(optimized_frac_df.tail(1).iloc[:,0:2].to_markdown(index=False, tablefmt="simple"))
        print('\n')
        print(optimized_frac_df.tail(1).iloc[:,2:].to_markdown(index=False, tablefmt="rst"))

        return optimized_frac_df



def masked_softmax(vec, mask, dim=1, epsilon=1e-5):
    exps = torch.exp(vec)
    masked_exps = exps * mask
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps/masked_sums)


def elem_lookup(src):
    try:
        numpy_src = src.numpy().reshape(-1)
    except:
        numpy_src = src
    all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                   'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                   'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                   'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                   'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                   'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                   'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                   'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                   'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                   'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                   'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    
    elem_names = [all_symbols[i] for i in numpy_src]
    return elem_names

#%%
if __name__ == '__main__':
    pass