import os
import sys
import json
import pickle
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch import autograd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

Xmin, Xmax = 0.0, 1.0
Ymin, Ymax = 0.0, 1.0
Tmin, Tmax = 0.0, 0.25


class VolumeGenerator:
    def __init__(self, Xmin, Xmax, Ymin, Ymax, Tmin, Tmax):
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.Ymin = Ymin
        self.Ymax = Ymax
        self.Tmin = Tmin
        self.Tmax = Tmax

    def generate_volume_points(self, num_points):
        """
        Generates random points inside the volume using PyTorch.
        :param num_points: Number of points to generate inside the volume.
        :return: A tensor of shape (num_points, 3) with points [x, y, t].
        """
        x_points = torch.rand(num_points) * (self.Xmax - self.Xmin) + self.Xmin
        y_points = torch.rand(num_points) * (self.Ymax - self.Ymin) + self.Ymin
        t_points = torch.rand(num_points) * (self.Tmax - self.Tmin) + self.Tmin
        return torch.stack((x_points, y_points, t_points), dim=1)

    def generate_surface_points(self, surface, num_points):
        """
        Generates random points on a specified surface of the volume.
        :param surface: The surface to generate points on ('Xmin', 'Xmax', 'Ymin', 'Ymax', 'Tmin', 'Tmax').
        :param num_points: Number of points to generate on the specified surface.
        :return: A tensor of shape (num_points, 3) with points [x, y, t] on the specified surface.
        """
        if surface == 'Xmin':
            return torch.stack((
                torch.full((num_points,), self.Xmin),
                torch.rand(num_points) * (self.Ymax - self.Ymin) + self.Ymin,
                torch.rand(num_points) * (self.Tmax - self.Tmin) + self.Tmin
            ), dim=1)

        elif surface == 'Xmax':
            return torch.stack((
                torch.full((num_points,), self.Xmax),
                torch.rand(num_points) * (self.Ymax - self.Ymin) + self.Ymin,
                torch.rand(num_points) * (self.Tmax - self.Tmin) + self.Tmin
            ), dim=1)

        elif surface == 'Ymin':
            return torch.stack((
                torch.rand(num_points) * (self.Xmax - self.Xmin) + self.Xmin,
                torch.full((num_points,), self.Ymin),
                torch.rand(num_points) * (self.Tmax - self.Tmin) + self.Tmin
            ), dim=1)

        elif surface == 'Ymax':
            return torch.stack((
                torch.rand(num_points) * (self.Xmax - self.Xmin) + self.Xmin,
                torch.full((num_points,), self.Ymax),
                torch.rand(num_points) * (self.Tmax - self.Tmin) + self.Tmin
            ), dim=1)

        elif surface == 'Tmin':
            return torch.stack((
                torch.rand(num_points) * (self.Xmax - self.Xmin) + self.Xmin,
                torch.rand(num_points) * (self.Ymax - self.Ymin) + self.Ymin,
                torch.full((num_points,), self.Tmin)
            ), dim=1)

        elif surface == 'Tmax':
            return torch.stack((
                torch.rand(num_points) * (self.Xmax - self.Xmin) + self.Xmin,
                torch.rand(num_points) * (self.Ymax - self.Ymin) + self.Ymin,
                torch.full((num_points,), self.Tmax)
            ), dim=1)

        else:
            raise ValueError("Invalid surface name. Choose from 'Xmin', 'Xmax', 'Ymin', 'Ymax', 'Tmin', 'Tmax'.")
        
class EulerViscousTime2D:
    def __init__(self, model, device='cpu', value=-7.0):
        self.model = model
        self.device = device
        self.raw_mu = torch.nn.Parameter(torch.tensor([value], requires_grad=True, device=self.device, dtype=torch.float32))

    @property
    def mu(self):
        return torch.nn.functional.softplus(self.raw_mu)


    def compute_gradients(self, outputs, inputs):
        return autograd.grad(outputs=outputs, inputs=inputs,
                             grad_outputs=torch.ones_like(outputs),
                             create_graph=True, retain_graph=True)[0]
    
    def compute_loss(self, x_train):
        self.x_train = torch.tensor(x_train, device=self.device, dtype=torch.float32)
        self.gamma = torch.tensor(1.4, device=self.device, dtype=torch.float32)  # Specific heat ratio

        self.x_train.requires_grad = True

        output = self.model(self.x_train)
        rho, p, u, v = output[:, 0], output[:, 1], output[:, 2], output[:,3]

        E = p/(self.gamma - 1) + 0.5*rho*(u**2 + v**2)

        U1 = rho
        U2 = rho*u
        U3 = rho*v
        U4 = E

        U1_x = self.compute_gradients(U1, self.x_train)[:,0]
        U2_x = self.compute_gradients(U2, self.x_train)[:,0]
        U3_x = self.compute_gradients(U3, self.x_train)[:,0]
        U4_x = self.compute_gradients(U4, self.x_train)[:,0]
        
        U1_y = self.compute_gradients(U1, self.x_train)[:,1]
        U2_y = self.compute_gradients(U2, self.x_train)[:,1]
        U3_y = self.compute_gradients(U3, self.x_train)[:,1]
        U4_y = self.compute_gradients(U4, self.x_train)[:,1]
        
        U1_xx = self.compute_gradients(U1_x, self.x_train)[:,0]
        U2_xx = self.compute_gradients(U2_x, self.x_train)[:,0]
        U3_xx = self.compute_gradients(U3_x, self.x_train)[:,0]
        U4_xx = self.compute_gradients(U4_x, self.x_train)[:,0]
        
        U1_yy = self.compute_gradients(U1_y, self.x_train)[:,1]
        U2_yy = self.compute_gradients(U2_y, self.x_train)[:,1]
        U3_yy = self.compute_gradients(U3_y, self.x_train)[:,1]
        U4_yy = self.compute_gradients(U4_y, self.x_train)[:,1]
        
        U1_t = self.compute_gradients(U1, self.x_train)[:,2]
        U2_t = self.compute_gradients(U2, self.x_train)[:,2]
        U3_t = self.compute_gradients(U3, self.x_train)[:,2]
        U4_t = self.compute_gradients(U4, self.x_train)[:,2]

        f1 = rho*u
        f2 = rho * u**2 + p
        f3 = rho*u*v
        f4 = u*(E + p)

        f1_x = self.compute_gradients(f1, self.x_train)[:,0]
        f2_x = self.compute_gradients(f2, self.x_train)[:,0]
        f3_x = self.compute_gradients(f3, self.x_train)[:,0]
        f4_x = self.compute_gradients(f4, self.x_train)[:,0]

        g1 = rho*v
        g2 = rho * u *v
        g3 = rho * v**2 + p
        g4 = v*(E + p)
    
        g1_y = self.compute_gradients(g1, self.x_train)[:,1]
        g2_y = self.compute_gradients(g2, self.x_train)[:,1]
        g3_y = self.compute_gradients(g3, self.x_train)[:,1]
        g4_y = self.compute_gradients(g4, self.x_train)[:,1]
    
        r1 = U1_t + f1_x + g1_y - self.mu *(U1_xx + U1_yy) 
        r2 = U2_t + f2_x + g2_y - self.mu *(U2_xx + U2_yy) 
        r3 = U3_t + f3_x + g3_y - self.mu *(U3_xx + U3_yy) 
        r4 = U4_t + f4_x + g4_y - self.mu *(U4_xx + U4_yy) 


        return r1, r2, r3, r4

class FNN(nn.Module):
    def __init__(self, inputs, layers, init_type='xavier', output2='exp'):
        super(FNN, self).__init__()

        # Normalize inputs
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        self.register_buffer('mean', inputs_tensor.mean(dim=0))
        self.register_buffer('std', inputs_tensor.std(dim=0))

        self.layers = layers
        self.init_type = init_type
        self.output2 = output2  # Transformation for the second output

        self.hidden_weights = nn.ParameterList([
            nn.Parameter(torch.empty(layers[i], layers[i + 1]))
            for i in range(len(layers) - 1)
        ])
        self.hidden_biases = nn.ParameterList([
            nn.Parameter(torch.empty(1, layers[i + 1]))
            for i in range(len(layers) - 1)
        ])

        self.initialize_weights()

    def initialize_weights(self):
        init_methods = {
            'xavier': init.xavier_normal_,
            'he': lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')
        }
        initializer = init_methods.get(self.init_type, init.xavier_normal_)

        for weight in self.hidden_weights:
            initializer(weight)
        for bias in self.hidden_biases:
            init.constant_(bias, 0)

    def forward(self, x):
        # Convert NumPy array to PyTorch tensor if necessary
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.mean.device)

        # Normalize input
        x = (x - self.mean) / self.std

        for i, (weight, bias) in enumerate(zip(self.hidden_weights, self.hidden_biases)):
            x = torch.matmul(x, weight) + bias
            if i < len(self.hidden_weights) - 1:  # Apply activation for hidden layers
                x = torch.tanh(x)

        # Apply transformations to outputs
        x[:, 0] = torch.exp(x[:, 0])  # Exponential transformation for the first output
        if self.output2 == 'exp':
            x[:, 1] = torch.exp(x[:, 1])
        elif self.output2 == '10pow':
            x[:, 1] = 10 ** x[:, 1]

        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def save_predictions(self, x_input, filename):
        predictions = self.predict(x_input).cpu().numpy()
        inputs = (
            x_input if isinstance(x_input, np.ndarray) else x_input.cpu().numpy()
        )
        data = np.concatenate((inputs, predictions), axis=1)
        np.save(filename, data)

    def save_parameters(self, filepath):
        params = {
            'hidden_weights': [w.detach().cpu().numpy() for w in self.hidden_weights],
            'hidden_biases': [b.detach().cpu().numpy() for b in self.hidden_biases],
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)

    def print_state_dict(self):
        print("Model's state_dict:")
        for name, param in self.state_dict().items():
            print(name, "\t", param.size())

class ResidualDataset(Dataset):
    def __init__(self, res_pts):
        self.res = res_pts
        self.n_residuals = self.res.shape[0]

    def __getitem__(self, index):
        return self.res[index]

    def __len__(self):
        return self.n_residuals
    
class Loss:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.loss_fn = nn.MSELoss()  

    # def LossPDE(self, coords, pde):
    #     self.pde = pde
    #     self.coords = torch.tensor(coords, dtype=torch.float32, device=self.device).clone().detach().requires_grad_(True)
        
    #     e1, e2, e3, e4= self.pde.compute_loss(self.coords)

    #     loss = (self.loss_fn(e1, torch.zeros_like(e1)),
    #              self.loss_fn(e2, torch.zeros_like(e2)), 
    #              self.loss_fn(e3, torch.zeros_like(e3)),
    #              self.loss_fn(e4, torch.zeros_like(e4)))

    #     mse = (self.loss_fn(e1, torch.zeros_like(e1)) + 
    #            self.loss_fn(e2, torch.zeros_like(e2)) + 
    #            self.loss_fn(e3, torch.zeros_like(e3)) +
    #            self.loss_fn(e4, torch.zeros_like(e4)))

    #     return loss, mse
    def LossPDE(self, coords, pde):
        self.pde = pde
        coords = torch.as_tensor(coords, dtype=torch.float32, device=self.device).requires_grad_(True)

        e1, e2, e3, e4 = self.pde.compute_loss(coords)
        residuals = torch.stack([e1, e2, e3, e4])

        loss = [self.loss_fn(e, torch.zeros_like(e)) for e in residuals]
        mse = sum(loss)

        return loss, mse

    def LossInitial(self, coords, target):
        coords = torch.as_tensor(coords, dtype=torch.float32, device=self.device).requires_grad_(True)
        target = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        output = self.model(coords)
        mse = self.loss_fn(target, output)
        return mse

ic = torch.tensor([[0.5313,0.0,0.0,0.4],[1.0,0.7276,0,1.0],[0.8,0.0,0.0,1.0],[1.0, 0, 0.7276, 1.0]])
print(ic[0,1])


volume_gen = VolumeGenerator(Xmin, Xmax, Ymin, Ymax, Tmin, Tmax)

interior_points = volume_gen.generate_volume_points(40000)

inputs = interior_points.clone().detach().to(dtype=torch.float32, device=device)
layers=[3]+5*[100]+[4]

model = FNN(inputs, layers, init_type='xavier')
model.to(device)


loss = Loss(model, device=device)
ns = EulerViscousTime2D(model, device=device, value = -5.0)

losses=[]

data= {"Epoch":[],
        "Cont":[],
        "Mom_x":[],
        "Mom_y":[],
        "Energy":[],
        "lefttop":[],
        "leftbottom":[],
        "righttop":[],
        "rightbottom":[],
        "mu":[]
        }

filename  = 'loss.json'


def train(epochs=10000, lr = 0.001):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer1 = optim.Adam([ns.raw_mu], lr=0.001)

    lambda1 = lambda epoch: 10**(-(2/50000)*epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    for epoch in range(epochs):
        initial_data = volume_gen.generate_surface_points(surface='Tmin', num_points=20000)
        top_left = initial_data[(initial_data[:, 0] < 0.5) & (initial_data[:, 1] > 0.5)]
        top_right = initial_data[(initial_data[:, 0] > 0.5) & (initial_data[:, 1] > 0.5)]
        bottom_left = initial_data[(initial_data[:, 0] < 0.5) & (initial_data[:, 1] < 0.5)]
        bottom_right = initial_data[(initial_data[:, 0] > 0.5) & (initial_data[:, 1] < 0.5)]


        tl_ic = torch.cat([
            ic[1,0]*torch.ones((top_left.shape[0], 1)),       
            ic[1,3]*torch.ones((top_left.shape[0], 1)),           
            ic[1,1]*  torch.ones((top_left.shape[0], 1)),    
            ic[1,2]* torch.ones((top_left.shape[0], 1))      
        ], dim=1)
        
        tr_ic = torch.cat([
            ic[0,0]*torch.ones((top_right.shape[0], 1)),          
            ic[0,3]*torch.ones((top_right.shape[0], 1)),          
            ic[0,1]*  torch.ones((top_right.shape[0], 1)),    
            ic[0,2]* torch.ones((top_right.shape[0], 1))     
        ], dim=1)

        bl_ic = torch.cat([
            ic[2,0]*torch.ones((bottom_left.shape[0], 1)),       
            ic[2,3]*torch.ones((bottom_left.shape[0], 1)),           
            ic[2,1]*  torch.ones((bottom_left.shape[0], 1)),    
            ic[2,2]* torch.ones((bottom_left.shape[0], 1))      
        ], dim=1)
        
        br_ic = torch.cat([
            ic[3,0]*torch.ones((bottom_right.shape[0], 1)),          
            ic[3,3]*torch.ones((bottom_right.shape[0], 1)),          
            ic[3,1]*  torch.ones((bottom_right.shape[0], 1)),    
            ic[3,2]* torch.ones((bottom_right.shape[0], 1))     
        ], dim=1)
        
        collocation_points = volume_gen.generate_volume_points(160000)


        dataset = ResidualDataset(collocation_points)
        dataloader_ = DataLoader(dataset=dataset, batch_size=4000, shuffle=True, pin_memory=True)
        for residuals in dataloader_:
            
            pde_list, pde_mse = loss.LossPDE(residuals, ns)
            mse_tl = loss.LossInitial(top_left, tl_ic)
            mse_tr = loss.LossInitial(top_right, tr_ic)
            mse_bl = loss.LossInitial(bottom_left, bl_ic)
            mse_br = loss.LossInitial(bottom_right, br_ic)
            total = ((pde_list[0] + pde_list[1] + pde_list[2] + pde_list[3]) + 10*(mse_bl + mse_br + mse_tl + mse_tr))
            
            optimizer.zero_grad()
            optimizer1.zero_grad()
            total.backward()
            optimizer.step()
            optimizer1.step()
            scheduler.step()
            
            loss_ = total.item()        
        if epoch%10==0:
            data["Epoch"].append(epoch)
            data["Cont"].append(pde_list[0].item())
            data["Mom_x"].append(pde_list[1].item())
            data["Mom_y"].append(pde_list[2].item())
            data["Energy"].append(pde_list[3].item())
            data["lefttop"].append(mse_tl.item())
            data["leftbottom"].append(mse_bl.item())
            data["righttop"].append(mse_tr.item())
            data["rightbottom"].append(mse_br.item())
            data["mu"].append(ns.mu.item())
            
            with open(filename, 'w') as f:
                json.dump(data, f)
                
            print(f"Epoch: {epoch}, Loss: {loss_:.4e}, PDE: {pde_mse.item()}, mu : {ns.mu.item()}")

            # Save model state
            torch.save(model.state_dict(), f'./results/model_adam.pth')

            # Define step size and ranges for all three dimensions
            step_size = 0.01
            x = torch.arange(0.0, 1.0 + step_size, step_size)
            y = torch.arange(0.0, 1.0 + step_size, step_size)
            t = torch.arange(0.0, 0.25 + step_size, step_size)

            X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
            flat_X = X.flatten()
            flat_Y = Y.flatten()
            flat_T = T.flatten()

            # Stack the flattened coordinates into a single tensor
            grid = torch.stack([flat_X, flat_Y, flat_T], dim=1)
            grid = grid.to(dtype=torch.float32, device=device)
            # Save predictions for the 3D grid and additional coordinates
            model.save_predictions(grid, './predict/predictions.npy')

            
t0 = time.time()
train(epochs=4001, lr =1e-02)
total_time = (time.time()-t0)/60
print(f'Total: {total_time} min')


def train_lbfgs(epochs=50000, lr=1e-02):
    optimizer = optim.LBFGS(
        list(model.parameters()) + [ns.raw_mu],
        lr=lr,
        max_iter=20,  # Maximum number of iterations per call
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe"  # Line search method
    )

    # optimizer_mu = optim.Adam([ns.raw_mu], lr=1e-3)  # Using Adam for `mu` optimization

    for epoch in range(epochs):
        initial_data = volume_gen.generate_surface_points(surface='Tmin', num_points=5000)
        top_left = initial_data[(initial_data[:, 0] < 0.5) & (initial_data[:, 1] > 0.5)]
        top_right = initial_data[(initial_data[:, 0] > 0.5) & (initial_data[:, 1] > 0.5)]
        bottom_left = initial_data[(initial_data[:, 0] <= 0.5) & (initial_data[:, 1] <= 0.5)]
        bottom_right = initial_data[(initial_data[:, 0] >= 0.5) & (initial_data[:, 1] <= 0.5)]

        # Generate initial conditions for each quadrant
        tl_ic = torch.cat([
            ic[1, 0] * torch.ones((top_left.shape[0], 1)),
            ic[1, 3] * torch.ones((top_left.shape[0], 1)),
            ic[1, 1] * torch.ones((top_left.shape[0], 1)),
            ic[1, 2] * torch.ones((top_left.shape[0], 1))
        ], dim=1)
        
        tr_ic = torch.cat([
            ic[3, 0] * torch.ones((top_right.shape[0], 1)),
            ic[3, 3] * torch.ones((top_right.shape[0], 1)),
            ic[3, 1] * torch.ones((top_right.shape[0], 1)),
            ic[3, 2] * torch.ones((top_right.shape[0], 1))
        ], dim=1)

        bl_ic = torch.cat([
            ic[0, 0] * torch.ones((bottom_left.shape[0], 1)),
            ic[0, 3] * torch.ones((bottom_left.shape[0], 1)),
            ic[0, 1] * torch.ones((bottom_left.shape[0], 1)),
            ic[0, 2] * torch.ones((bottom_left.shape[0], 1))
        ], dim=1)

        br_ic = torch.cat([
            ic[2, 0] * torch.ones((bottom_right.shape[0], 1)),
            ic[2, 3] * torch.ones((bottom_right.shape[0], 1)),
            ic[2, 1] * torch.ones((bottom_right.shape[0], 1)),
            ic[2, 2] * torch.ones((bottom_right.shape[0], 1))
        ], dim=1)

        collocation_points = volume_gen.generate_volume_points(80000)

        # Define the closure required for L-BFGS
        def closure():
            optimizer.zero_grad()
            # optimizer_mu.zero_grad()

            # Loss calculation
            pde_list, pde_mse = loss.LossPDE(collocation_points, ns)
            mse_tl = loss.LossInitial(top_left, tl_ic)
            mse_tr = loss.LossInitial(top_right, tr_ic)
            mse_bl = loss.LossInitial(bottom_left, bl_ic)
            mse_br = loss.LossInitial(bottom_right, br_ic)

            total_loss = ((pde_list[0] + pde_list[1] + pde_list[2] + pde_list[3]) +
                          40 * (mse_bl + mse_br + mse_tl + mse_tr))
            # print(mse_bl.item())
            
            total_loss.backward()  # Compute gradients
            return total_loss

        # Perform optimizer step
        optimizer.step(closure)
        # optimizer_mu.step()  # Update `mu` using Adam

        # Compute current loss for logging
        current_loss = closure().item()

        # Log and save results every 100 epochs
        if epoch % 10 == 0:
            # data["Epoch"].append(epoch)
            # data["Cont"].append(pde_list[0].item())
            # data["Mom_x"].append(pde_list[1].item())
            # data["Mom_y"].append(pde_list[2].item())
            # data["Energy"].append(pde_list[3].item())
            # data["lefttop"].append(mse_tl.item())
            # data["leftbottom"].append(mse_bl.item())
            # data["righttop"].append(mse_tr.item())
            # data["rightbottom"].append(mse_br.item())
            # data["mu"].append(ns.mu.item())

            # Save results to file
            # with open(filename, 'w') as f:
            #     json.dump(data, f)

            print(f"Epoch: {epoch}, Loss: {current_loss:.4e}, PDE: {current_loss}, mu: {ns.mu.item()}")

            # Save model state
            torch.save(model.state_dict(), f'./results/model_lbfgs.pth')

            # Define step size and ranges for all three dimensions
            step_size = 0.01
            x = torch.arange(0.0, 1.0 + step_size, step_size)
            y = torch.arange(0.0, 1.0 + step_size, step_size)
            t = torch.arange(0.0, 0.4 + step_size, step_size)

            X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
            flat_X = X.flatten()
            flat_Y = Y.flatten()
            flat_T = T.flatten()

            # Stack the flattened coordinates into a single tensor
            grid = torch.stack([flat_X, flat_Y, flat_T], dim=1)
            grid = grid.to(dtype=torch.float32, device=device)

            # Save predictions for the 3D grid and additional coordinates
            model.save_predictions(grid, './predict/predictions.npy')

    print("Training with L-BFGS complete!")


# Measure training time
t0 = time.time()
train_lbfgs(epochs=8001, lr=1e-02)
total_time = (time.time() - t0) / 60
print(f'Total: {total_time} min')
