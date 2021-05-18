# ascension hub utils
import os, sys
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import pandas as pd
from crabnet.neokingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device

compute_device = get_compute_device(prefer_last=True)

class AscendedCrab():
    def __init__(self, src, prop0, prop1, prop0_target, prop1_target,
                 alpha=0.5, saving=False, ensemble=False,
                 lr=0.025, compute_device=compute_device):
        
        self.src = src.to(compute_device, dtype=torch.long, non_blocking=True)
        self.prop0 = prop0
        self.prop0_target = 1 if prop0_target == 'max' else -1
        self.prop1 = prop1
        self.prop1_target = 1 if prop1_target == 'max' else -1
        self.alpha = alpha
        self.saving = saving
        self.ensemble=False
        self.lr = lr
        self.compute_device = compute_device
        self.model_0 = load_model(self.prop0)
        if not self.prop1 == 'Loss':
            self.model_1 = load_model(self.prop1)


    def ascend(self, epochs=100):

        delim = '-'
        print(f'\n\nOptimizing {delim.join(elem_lookup(self.src))} System...\n'.title())

        loss0s = []
        srcs = []
        fracs = []

        prop0_preds = []
        prop0_uncs = []
        if not self.prop1 == 'Loss':
            loss1s = []
            prop1_preds = []
            prop1_uncs = []


        frac = torch.rand(int(self.src.shape[1])).view(1,-1) \
                .to(compute_device,dtype=torch.float,non_blocking=True)\


        frac_mask = torch.where(self.src != 0, 1, 0)
        frac_mask = frac_mask.to(compute_device,
                              dtype=torch.float,
                              non_blocking=True)


        optimizer = optim.Adam([frac.requires_grad_()], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [epochs-5], gamma=0.1, last_epoch=-1, verbose=False)
        criterion = nn.L1Loss()
        # criterion = nn.MSELoss()
        criterion = criterion.to(compute_device)
        torch.autograd.set_detect_anomaly(True)
        for epoch in tqdm(range(epochs)):

            soft_frac = masked_softmax(frac, frac_mask)

            optimizer.zero_grad()
            if not self.prop1 == 'Loss':
                # prop0_pred, prop0_unc = self.model_0.single_predict(self.src, soft_frac)
                # if epoch == 0:
                #     init_0 = prop0_pred.clone().detach()
                # loss0 = criterion((prop0_pred/init_0), torch.tensor([[100000.0]]).to(compute_device))
                # prop1_pred, prop1_unc = self.model_1.single_predict(self.src, soft_frac)
                # if epoch == 0:
                #     init_1 = prop1_pred.clone().detach()
                # loss1 = criterion((init_1/prop1_pred), torch.tensor([[100000.0]]).to(compute_device))

                scaled_p0, _, prop0_pred, prop0_unc = self.model_0.single_predict(self.src, soft_frac)
                loss0 = criterion(scaled_p0, torch.tensor([[self.prop0_target*100000.0]]).to(compute_device))
                scaled_p1, _, prop1_pred, prop1_unc = self.model_1.single_predict(self.src, soft_frac)
                loss1 = criterion(scaled_p1, torch.tensor([[self.prop1_target*100000.0]]).to(compute_device))

                # loss = (self.alpha * loss0) + ((1-self.alpha)*loss1)
                loss = loss0 + loss1
                loss.backward()

                loss1s.append(loss1.item())
                prop1_preds.append(prop1_pred.item())
                prop1_uncs.append(prop1_unc.item())
            else:
                scaled_p0, _, prop0_pred, prop0_unc = self.model_0.single_predict(self.src, soft_frac)
                loss0 = criterion(scaled_p0, torch.tensor([[self.prop0_target*100000.0]]).to(compute_device))
                loss0.backward()

            optimizer.step()
            scheduler.step()

            loss0s.append(loss0.item())
            srcs.append(self.src)
            fracs.append(soft_frac)
            prop0_preds.append(prop0_pred.item())
            prop0_uncs.append(prop0_unc.item())

        srcs = [elem_lookup(item.detach().numpy().reshape(-1)) for item in srcs]
        fracs = [item.detach().numpy().reshape(-1) for item in fracs]

        if self.prop1 == 'Loss':
            optimized_frac_df = pd.DataFrame(
                {'Elements': srcs,
                 'Fractions': fracs,
                   f'{self.prop0}': prop0_preds,
                   f'{self.prop0} UNC': prop0_uncs,
                  'Loss': loss0s
                })
        else:
            optimized_frac_df = pd.DataFrame(
                {'Elements': srcs,
                 'Fractions': fracs,
                   f'{self.prop0}': prop0_preds,
                   f'{self.prop0} UNC': prop0_uncs,
                   f'Prop0 Loss': loss0s,
                   f'{self.prop1}': prop1_preds,
                   f'{self.prop1} UNC': prop1_uncs,
                  f'Prop1 Loss': loss1s})


        print('\n-----------------------------------------------------')
        print('\nOptimized Fractional Composition:\n'.title())
        print(optimized_frac_df.tail(1).iloc[:,0:2].to_markdown(index=False, tablefmt="simple"))
        print('\n')
        print(optimized_frac_df.tail(1).iloc[:,2:5].to_markdown(index=False, tablefmt="rst"))
        if not self.prop1 == 'Loss':
            print(optimized_frac_df.tail(1).iloc[:,5:].to_markdown(index=False, tablefmt="rst"))
        return optimized_frac_df


def load_model(prop):
    try:
        data = rf'data/materials_data/{prop}/test.csv'
        model_name = prop
        model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                        model_name=f'{model_name}', verbose=False)
        model.load_network(f'{prop}.pth')
        model.load_data(data, batch_size=2**9, train=False)
    except:
        model_list = os.listdir('models/trained_models/')
        print('\nAn error occurred while trying to load the model...\n')
        print('It is likely that you are trying to load a property model that is not available.')
        print(f'\nAvailable models include:')
        for model in model_list:
            print('  - ' + model[:-4])
        sys.exit()
    return model


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

def save_results(optim_frac_df, save_dir=None):
    if save_dir is not None:
        delim = '-'
        file_name = f'{save_dir}/{delim.join(optim_frac_df.iloc[0,0])}_optimization_results.csv'
        os.makedirs(save_dir, exist_ok=True)
        optim_frac_df.to_csv(file_name, index=False)
#%%
# if __name__ == '__main__':
#     pass
