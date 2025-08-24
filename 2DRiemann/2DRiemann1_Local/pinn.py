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
from torch.utils.data import Dataset, DataLoader, TensorDataset


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
        if self.output2 == 'exp':
            x[:, 0:2] = torch.exp(x[:, 0:2]) # Exponential transformation for the first output
        
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
            

class PINN(nn.Module):
    def __init__(self, model, model2, device='cpu'):
        super().__init__()
        self.model = model
        self.model2 = model2
        self.device = device
        self.gamma = torch.tensor(1.4, device=device, dtype=torch.float32)  # Specific heat ratio
        self.optimizer = optim.Adam(list(model.parameters()) + list(model2.parameters()), lr=0.001)
        self.optimizer2 = torch.optim.LBFGS(list(model.parameters()) + list(model2.parameters()), max_iter=50, tolerance_grad=1e-8, line_search_fn='strong_wolfe')
        # self.optimizer3  = optim.Adam([self.raw_mu], lr=0.001)
        
        self.data ={"Epoch":[],
                    "Cont":[],
                    "MomX":[],
                    "MomY":[],
                    "Energy":[],
                    "ic":[],
                    "mu":[]}
        
    # @property
    # def mu(self):
    #     return torch.nn.functional.softplus(self.raw_mu)   
    
    def compute_gradients(self, outputs, inputs):
        return autograd.grad(outputs=outputs, inputs=inputs,
                             grad_outputs=torch.ones_like(outputs),
                             create_graph=True, retain_graph=True)[0]
        
    def _physics_loss(self, x_train):
        # self.x_train = torch.tensor(x_train, device=self.device, dtype=torch.float32)

        x_train.requires_grad = True

        output = self.model(x_train)
        rho, p, u, v = output[:, 0], output[:, 1], output[:, 2], output[:,3]
        
        self.mu = 0.01 * self.model2(x_train)**2

        E = p/(self.gamma - 1) + 0.5*rho*(u**2 + v**2)

        U1 = rho
        U2 = rho*u
        U3 = rho*v
        U4 = E
        
        u_x = self.compute_gradients(u, x_train)[:,0]
        v_y = self.compute_gradients(v, x_train)[:,1]

        U1_x = self.compute_gradients(U1, x_train)[:,0]
        U2_x = self.compute_gradients(U2, x_train)[:,0]
        U3_x = self.compute_gradients(U3, x_train)[:,0]
        U4_x = self.compute_gradients(U4, x_train)[:,0]
        
        U1_y = self.compute_gradients(U1, x_train)[:,1]
        U2_y = self.compute_gradients(U2, x_train)[:,1]
        U3_y = self.compute_gradients(U3, x_train)[:,1]
        U4_y = self.compute_gradients(U4, x_train)[:,1]
        
        U1_xx = self.compute_gradients(U1_x, x_train)[:,0]
        U2_xx = self.compute_gradients(U2_x, x_train)[:,0]
        U3_xx = self.compute_gradients(U3_x, x_train)[:,0]
        U4_xx = self.compute_gradients(U4_x, x_train)[:,0]
        
        U1_yy = self.compute_gradients(U1_y, x_train)[:,1]
        U2_yy = self.compute_gradients(U2_y, x_train)[:,1]
        U3_yy = self.compute_gradients(U3_y, x_train)[:,1]
        U4_yy = self.compute_gradients(U4_y, x_train)[:,1]
        
        U1_t = self.compute_gradients(U1, x_train)[:,2]
        U2_t = self.compute_gradients(U2, x_train)[:,2]
        U3_t = self.compute_gradients(U3, x_train)[:,2]
        U4_t = self.compute_gradients(U4, x_train)[:,2]

        f1 = rho*u
        f2 = rho * u**2 + p
        f3 = rho*u*v
        f4 = u*(E + p)

        f1_x = self.compute_gradients(f1, x_train)[:,0]
        f2_x = self.compute_gradients(f2, x_train)[:,0]
        f3_x = self.compute_gradients(f3, x_train)[:,0]
        f4_x = self.compute_gradients(f4, x_train)[:,0]

        g1 = rho*v
        g2 = rho * u *v
        g3 = rho * v**2 + p
        g4 = v*(E + p)
    
        g1_y = self.compute_gradients(g1, x_train)[:,1]
        g2_y = self.compute_gradients(g2, x_train)[:,1]
        g3_y = self.compute_gradients(g3, x_train)[:,1]
        g4_y = self.compute_gradients(g4, x_train)[:,1]
        
        self.lam = 1/(0.1*(torch.abs(u_x) + torch.abs(v_y) -(u_x + v_y)) + 1)
    
        res1 = U1_t + f1_x + g1_y - self.mu *(U1_xx + U1_yy) 
        r1 = torch.mean(res1**2)
        
        res2 = U2_t + f2_x + g2_y - self.mu *(U2_xx + U2_yy) 
        r2 = torch.mean(res2**2)
        
        res3 = U3_t + f3_x + g3_y - self.mu *(U3_xx + U3_yy) 
        r3 = torch.mean( res3**2)
        
        res4 = U4_t + f4_x + g4_y - self.mu *(U4_xx + U4_yy) 
        r4 = torch.mean(res4**2)

        return r1, r2, r3, r4
    
    def _initial_loss(self, x_initial, ic):
        
        top_left = x_initial[(x_initial[:, 0] < 0.5) & (x_initial[:, 1] > 0.5)]
        top_right = x_initial[(x_initial[:, 0] > 0.5) & (x_initial[:, 1] > 0.5)]
        bottom_left = x_initial[(x_initial[:, 0] < 0.5) & (x_initial[:, 1] < 0.5)]
        bottom_right = x_initial[(x_initial[:, 0] > 0.5) & (x_initial[:, 1] < 0.5)]

        tl_ic = torch.cat([
            ic[1,0]*torch.ones((top_left.shape[0], 1), device=self.device),       
            ic[1,3]*torch.ones((top_left.shape[0], 1), device=self.device),           
            ic[1,1]*  torch.ones((top_left.shape[0], 1), device=self.device),    
            ic[1,2]* torch.ones((top_left.shape[0], 1), device=self.device)      
        ], dim=1).to(self.device)
        
        tr_ic = torch.cat([
            ic[3,0]*torch.ones((top_right.shape[0], 1), device=self.device),          
            ic[3,3]*torch.ones((top_right.shape[0], 1), device=self.device),          
            ic[3,1]*  torch.ones((top_right.shape[0], 1), device=self.device),    
            ic[3,2]* torch.ones((top_right.shape[0], 1), device=self.device)     
        ], dim=1).to(self.device)

        bl_ic = torch.cat([
            ic[0,0]*torch.ones((bottom_left.shape[0], 1), device=self.device),       
            ic[0,3]*torch.ones((bottom_left.shape[0], 1), device=self.device),           
            ic[0,1]*  torch.ones((bottom_left.shape[0], 1), device=self.device),    
            ic[0,2]* torch.ones((bottom_left.shape[0], 1), device=self.device)      
        ], dim=1).to(self.device)
        
        br_ic = torch.cat([
            ic[2,0]*torch.ones((bottom_right.shape[0], 1), device=self.device),          
            ic[2,3]*torch.ones((bottom_right.shape[0], 1), device=self.device),          
            ic[2,1]*  torch.ones((bottom_right.shape[0], 1), device=self.device),    
            ic[2,2]* torch.ones((bottom_right.shape[0], 1), device=self.device)     
        ], dim=1).to(self.device)
        
        tl_pred = self.model(top_left)
        tr_pred = self.model(top_right) 
        bl_pred = self.model(bottom_left)
        br_pred = self.model(bottom_right)
        
        loss_tl = torch.mean((tl_pred[:,0] - tl_ic[:,0])**2) + \
                     torch.mean((tl_pred[:,1] - tl_ic[:,1])**2) + \
                     torch.mean((tl_pred[:,2] - tl_ic[:,2])**2) + \
                     torch.mean((tl_pred[:,3] - tl_ic[:,3])**2)
        loss_tr = torch.mean((tr_pred[:,0] - tr_ic[:,0])**2) + \
                     torch.mean((tr_pred[:,1] - tr_ic[:,1])**2) + \
                     torch.mean((tr_pred[:,2] - tr_ic[:,2])**2) + \
                     torch.mean((tr_pred[:,3] - tr_ic[:,3])**2)
        loss_bl = torch.mean((bl_pred[:,0] - bl_ic[:,0])**2) + \
                     torch.mean((bl_pred[:,1] - bl_ic[:,1])**2) + \
                     torch.mean((bl_pred[:,2] - bl_ic[:,2])**2) + \
                     torch.mean((bl_pred[:,3] - bl_ic[:,3])**2)
        loss_br = torch.mean((br_pred[:,0] - br_ic[:,0])**2) + \
                     torch.mean((br_pred[:,1] - br_ic[:,1])**2) + \
                     torch.mean((br_pred[:,2] - br_ic[:,2])**2) + \
                     torch.mean((br_pred[:,3] - br_ic[:,3])**2)
        
        return loss_tl + loss_tr + loss_bl + loss_br

    def train(self, x_train, x_initial, ic, epochs=1000, batch_size=10000):
        train_dataset = TensorDataset(x_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.adam_epochs = epochs

        for epoch in range(epochs):
            epoch_loss = 0.0
            lambda1 = lambda epoch: 10**(-(2/10000)*epoch)
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
            for batch in train_loader:
                x_batch = batch[0]
                
                self.optimizer.zero_grad()
                # self.optimizer3.zero_grad()
                
                r1, r2, r3, r4 = self._physics_loss(x_batch)
                physics_loss = r1 + r2 + r3 + r4
                
                initial_loss = self._initial_loss(x_initial, ic)
                
                loss = physics_loss + 10*initial_loss + 0.1*torch.mean(self.mu**2)
                
                loss.backward()
                self.optimizer.step()
                # self.optimizer3.step()
                epoch_loss += loss.item()
                
            scheduler.step()

            if epoch % 100 == 0:
                self.data["Epoch"].append(epoch)
                self.data["Cont"].append(r1.item()) 
                self.data["MomX"].append(r2.item())
                self.data["MomY"].append(r3.item())
                self.data["Energy"].append(r4.item())
                self.data["ic"].append(initial_loss.item())
                self.data["mu"].append(self.mu.mean().item())
                with open('loss.json', 'w') as f:
                    json.dump(self.data, f)
                print(f'Epoch {epoch}, Total Loss: {epoch_loss:.4e}, Physics Loss: {physics_loss.item():.4e}, Initial Loss: {initial_loss.item():.4e}, self.mu: {self.mu.mean().item():.4e}')
            if epoch % 1000 == 0:
                torch.save(self.model.state_dict(), f'./results/model_adam.pth')

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
                grid = grid.to(dtype=torch.float32, device=self.device)
                # Save predictions for the 3D grid and additional coordinates
                self.model.save_predictions(grid, './predict/predictions.npy')
                self.model2.save_predictions(grid, './predict/predictions2.npy')
                
    def train_lbfgs(self, x_train, x_initial, ic, epochs=1000):
        self.model.train()

        def closure():
            self.optimizer2.zero_grad()
            r1, r2, r3, r4 = self._physics_loss(x_train)
            physics_loss = r1 + r2 + r3 + r4
            initial_loss = self._initial_loss(x_initial, ic)
            loss = physics_loss + 10*initial_loss + 0.1*torch.mean(self.mu**2)
            loss.backward()
            self.loss_val = loss.detach()  # Save for logging outside
            self.loss_parts = (r1, r2, r3, r4, initial_loss)
            return loss
        
        for epoch in range(epochs):
            self.optimizer2.step(closure)
            
            if epoch % 10 == 0:
                total_loss = self.loss_val.item()
                r1, r2, r3, r4, initial_loss = self.loss_parts
                # r1, r2, r3, r4 = self._physics_loss(x_train)
                # initial_loss = self._initial_loss(x_initial, ic)
                
                self.data["Epoch"].append(epoch + self.adam_epochs)
                self.data["Cont"].append(r1.item())
                self.data["MomX"].append(r2.item()) 
                self.data["MomY"].append(r3.item())
                self.data["Energy"].append(r4.item())
                self.data["ic"].append(initial_loss.item())
                self.data["mu"].append(self.mu.mean().item())
                with open('loss.json', 'w') as f:
                    json.dump(self.data, f)
                
                print(f'Epoch {epoch}, Total Loss: {total_loss:.4e}, Physics Loss: {(r1 + r2 + r3 + r4).item():.4e}, Initial Loss: {initial_loss.item():.4e}, self.mu: {self.mu.mean().item():.4e}')

            
            if epoch % 100 == 0:
                
                torch.save(self.model.state_dict(), f'./results/model_lbfgs.pth')
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
                grid = grid.to(dtype=torch.float32, device=self.device)
                # Save predictions for the 3D grid and additional coordinates
                self.model.save_predictions(grid, './predict/predictions.npy')
                self.model2.save_predictions(grid, './predict/predictions2.npy')
                
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt    
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Example usage
    Xmin, Xmax = 0.0, 1.0
    Ymin, Ymax = 0.0, 1.0
    Tmin, Tmax = 0.0, 0.4

    volume_gen = VolumeGenerator(Xmin, Xmax, Ymin, Ymax, Tmin, Tmax)
    x_train = volume_gen.generate_volume_points(100000)
    x_train_lbfgs = volume_gen.generate_volume_points(40000)
    
    inputs_lbfgs = x_train_lbfgs.to(dtype=torch.float32, device=device)
    inputs = x_train.to(dtype=torch.float32, device=device)
    layers=[3]+5*[96]+[4]

    model = FNN(inputs, layers, init_type='xavier')
    model.to(device)
    
    layers2 = [3] + 5 * [96] + [1]
    model2 = FNN(inputs, layers2, init_type='xavier', output2='not_exp') 
    model2.to(device)
    
    x_initial = volume_gen.generate_surface_points(surface='Tmin', num_points=10000)
    x_initial = x_initial.to(dtype=torch.float32, device=device)
    
    ic = torch.tensor([[1.0, -0.75, 0.5, 1.0],[2,0.75,0.5,1.0],[3,-0.75,-0.5,1],[1,0.75,-0.5,1]])
    #ic = torch.tensor([[1.1, 0.8939, 0.8939, 1.1],[0.5065, 0.8939, 0.0, 0.35],[0.5065, 0.0, 0.8939,0.35],[1.1, 0.0, 0.0, 1.1]])
    ic = ic.to(dtype=torch.float32, device=device)
    
    pinn = PINN(model, model2, device=device)
    # pinn.to(device) 
    pinn.train(inputs, x_initial, ic, epochs=11)
    pinn.train_lbfgs(inputs_lbfgs, x_initial, ic, epochs=1001)
