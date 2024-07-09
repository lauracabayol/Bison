import random
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from bison.archive.gadget_bispectrum_archive import GadgetBispectrumArchive

class Bispectrum_Emulator():
    def __init__(self, 
                 ndeg=4, 
                 drop_sim=None,
                 emu_quantity='squeezed',
                 emulator='SR', 
                 nepochs=1000, 
                 batch_size=64,
                 nhidden=4,
                lr=1e-3):
        self.ndeg = ndeg
        self.drop_sim=drop_sim
        self.emu_type=emulator
        self.emu_quantity=emu_quantity
        self.nepochs=nepochs
        
        seed=32
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.archive = GadgetBispectrumArchive()
        self.k_Mpc_bisqueezed = self.archive.data[0]['k_squeezed_Mpc']
        self.k_Mpc_bispread = self.archive.data[0]['k_spread_Mpc']

        if self.emu_quantity=='squeezed':
            self.k_Mpc_training=self.archive.data[0]['k_squeezed_Mpc']
        elif self.emu_quantity=='spread':
            self.k_Mpc_training=self.archive.data[0]['k_spread_Mpc']

        self._set_bispectrums()
        self._set_training_data()

        
        
        if self.emu_type =='SR':
            self._define_symbolic_regressor()
            self._train_symbolic_regressor()
        elif self.emu_type=='NN':
            self.batch_size = batch_size
            self.lr=lr
            self.nhidden=nhidden
            self._define_neural_network()
            self._train_neural_network()

    def _set_bispectrums(self):
        self.squeezed_bispectrums = np.array([self.archive.training_data[ii]['squeezed_bispectrum_Mpc'] for ii in range(len(self.archive.training_data))])
        self.spread_bispectrums = np.array([self.archive.training_data[ii]['spread_bispectrum_Mpc'] for ii in range(len(self.archive.training_data))])

        p1ds = np.array([self.archive.training_data[ii]['p1d_Mpc'] for ii in range(len(self.archive.training_data))])
        k1d = np.array(self.archive.training_data[00]['k1d_Mpc'])

        
        #k1d = k1d[(k1d>0)&(k1d<4)]

        p1ds_squeezed = p1ds[:,1:len(self.k_Mpc_bisqueezed)+1]
        self.reduced_squeezed_bispectrums = self.squeezed_bispectrums / (p1ds_squeezed**2+2*p1ds_squeezed*p1ds_squeezed[:,0:1])

        k1d = k1d[(k1d>0) & (k1d<4)]
        k_index = np.arange(1,len(k1d),1)
        k2_index = 2*k_index

        
        p1ds_spread_k = p1ds[:,k_index]        
        p1ds_spread_2k = p1ds[:,k2_index]
        
        
        self.reduced_spread_bispectrums = self.spread_bispectrums / (p1ds_spread_k**2+2*p1ds_spread_k*p1ds_spread_2k)
            

    def _polynomial_smoothing(self):
        training_bisqueezed = []
        training_bispread = []
        training_coefficients_squeezed = []
        training_coefficients_spread = []

        for ii in range(len(self.archive.training_data)):
            fit_p1d = np.polyfit(np.log(self.k_Mpc_bisqueezed), self.reduced_squeezed_bispectrums[ii], deg=self.ndeg)
            training_coefficients_squeezed.append(fit_p1d)
            fit_p1d = np.poly1d(fit_p1d)
            training_bisqueezed.append(fit_p1d(np.log(self.k_Mpc_bisqueezed)))

            fit_p1d = np.polyfit(np.log(self.k_Mpc_bispread), self.reduced_spread_bispectrums[ii], deg=self.ndeg)
            training_coefficients_spread.append(fit_p1d)
            fit_p1d = np.poly1d(fit_p1d)
            training_bispread.append(fit_p1d(np.log(self.k_Mpc_bispread)))

        self.training_bisqueezed = np.array(training_bisqueezed) 
        self.training_bispread = np.array(training_bispread)
        self.training_coefficients_squeezed = np.array(training_coefficients_squeezed)
        self.training_coefficients_spread = np.array(training_coefficients_spread)
        

    def _Bispectrum_from_polynomial(self, coeffs):
        fit_p1d = np.poly1d(coeffs)
        Bk = fit_p1d(np.log(self.k_Mpc_training))
        return Bk

    def _Bispectrum_from_polynomial_torch(self, coeffs):
        k = torch.tensor(self.k_Mpc_training, dtype=torch.float32)
        log_k = torch.log(k)        
        Bk = coeffs[:, 0, None] * log_k[None,:]**4 + coeffs[:, 1, None] * log_k[None,:]**3 + coeffs[:, 2, None] * log_k[None,:]**2 +  coeffs[:, 3, None] * log_k[None,:]+ coeffs[:, 4, None] 
        return Bk

    def _set_training_data(self):
        training_data = []
        for ii in range(len(self.archive.training_data)):
            training_data_snap = [
                self.archive.data[ii]['Delta2_p'],
                self.archive.data[ii]['n_p'],
                self.archive.data[ii]['mF'],
                self.archive.data[ii]['sigT_Mpc'],
                self.archive.data[ii]['gamma'],
                self.archive.data[ii]['kF_Mpc'],
            ]
            training_data.append(training_data_snap)

        training_data = np.array(training_data)
        min_ = np.min(training_data,0)
        max_ = np.max(training_data,0)
        self.paramLims = np.c_[min_,max_]
        self.training_data = (training_data-max_) / (max_ - min_)
        self._polynomial_smoothing()

    def get_training_data(self):
        return self.training_data, self.training_bisqueezed

    def _define_symbolic_regressor(self):
        default_pysr_params = dict(
            populations=20,
            model_selection="accuracy",
            batching=False,
            bumper=True,
            maxsize = 10,
            maxdepth=6,
            parsimony=0.001,
            #constraints={'^': (-1, 1)}
        )
        self.model = PySRRegressor(
            niterations=500,
            binary_operators=["+", "*","/"],#"-", ,"^"
            unary_operators=["log", "sqrt"],#"sin"
            **default_pysr_params,
        )

    def _train_symbolic_regressor(self):
        if self.emu_quantity=='squeezed':
            ylabel =self.training_coefficients_squeezed
        elif self.emu_quantity=='spread':
            ylabel =self.training_coefficients_spread
        self.model.fit(self.training_data, ylabel)

    def _define_neural_network(self):
        layers = []
        layers.append(nn.Linear(self.training_data.shape[1], 64))  # Input layer
        
        # Hidden layers
        for _ in range(self.nhidden):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(64, 64))
            layers.append(nn.Dropout(0))
        
        # Output layer
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, self.ndeg + 1))
        
        # Define the model as a sequential container
        self.model = nn.Sequential(*layers)
        
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _train_neural_network(self):
        if self.emu_quantity=='squeezed':
            ylabel =self.training_bisqueezed
        elif self.emu_quantity=='spread':
            ylabel =self.training_bispread
            
        dataset = BispectrumDataset(self.training_data, ylabel)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.nepochs):
            for batch_data, batch_bisp in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                bk_pred = self._Bispectrum_from_polynomial_torch(outputs)
                loss = self.loss_function(bk_pred, batch_bisp)
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{self.nepochs}], Loss: {loss.item():.4f}')

    def _evaluate_neural_network(self, test_data):
        test_data_tensor = torch.Tensor(test_data)
        with torch.no_grad():
            predicted_coeffs = self.model(test_data_tensor)
        return predicted_coeffs.detach().cpu().numpy()
    def emulate_Bispec_Mpc(self, test_data):
        test_data_snap = [
            test_data[0]['Delta2_p'],
            test_data[0]['n_p'],
            test_data[0]['mF'],
            test_data[0]['sigT_Mpc'],
            test_data[0]['gamma'],
            test_data[0]['kF_Mpc'],
        ]
        test_data_snap = (np.array(test_data_snap) - emu.paramLims[:,1]) / (emu.paramLims[:,1]-emu.paramLims[:,0])

        ypredict = emu._evaluate_neural_network(test_data_snap.reshape(1,len(test_data_snap)))
        ypredict = emu._Bispectrum_from_polynomial(ypredict[0])

        return ypredict
