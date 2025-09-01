import os
import sys
import json
import pickle
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.autograd import grad
from scipy.optimize import minimize
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from torch import autograd
from shapely.geometry import Polygon, Point


class PolygonBoundaryPoints:
    def __init__(self, vertices, num_boundary_points=400):
        self.vertices = np.array(vertices)
        self.num_vertices = len(vertices)
        self.num_boundary_points = num_boundary_points
        self.polygon = Polygon(vertices)  # Use Shapely for inside checks

    def generate_points_on_edges(self):
        def points_on_edge(v1, v2, n):
            t = np.random.uniform(0, 1, n)
            return v1 * (1 - t).reshape(-1, 1) + v2 * t.reshape(-1, 1)

        boundary_points = []
        edge_points_list = []

        for i in range(self.num_vertices):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % self.num_vertices]  # Connect last vertex to the first
            edge_points = points_on_edge(v1, v2, self.num_boundary_points)
            edge_points_list.append(edge_points)
            boundary_points.append(edge_points)

        boundary_points = np.vstack(boundary_points)
        return boundary_points, edge_points_list

    def generate_random_points_inside(self, num_points_inside):
        """
        Generates random points strictly inside the polygon using rejection sampling.
        """
        bounds = self.polygon.bounds  # (minx, miny, maxx, maxy)
        min_x, min_y, max_x, max_y = bounds

        points = []
        while len(points) < num_points_inside:
            random_x = np.random.uniform(min_x, max_x, num_points_inside)
            random_y = np.random.uniform(min_y, max_y, num_points_inside)
            candidate_points = np.column_stack((random_x, random_y))

            # Check if points are inside the polygon
            for point in candidate_points:
                if self.polygon.contains(Point(point)):
                    points.append(point)
                    if len(points) == num_points_inside:
                        break

        return np.array(points)
    
    def plot_points(self, boundary_points=None, interior_points=None):
        """
        Plots the polygon with the boundary and interior points.
        
        Args:
        boundary_points: Points on the boundary of the polygon.
        interior_points: Points inside the polygon.
        """
        plt.figure(figsize=(6, 6))
        
        # Plot boundary points
        if boundary_points is not None:
            plt.scatter(boundary_points[:, 0], boundary_points[:, 1], color='r', label='Boundary Points', s=1)
        
        # Plot interior points
        if interior_points is not None:
            plt.scatter(interior_points[:, 0], interior_points[:, 1], color='b', label='Interior Points', s=1)
        
        # Plot polygon edges
        polygon = np.vstack([self.vertices, self.vertices[0]])  # Close the polygon
        plt.plot(polygon[:, 0], polygon[:, 1], 'k-', label='Polygon')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.savefig('test.png')
        plt.close()

class ResidualDataset(Dataset):
    def __init__(self, res_pts):
        self.res = res_pts
        self.n_residuals = self.res.shape[0]

    def __getitem__(self, index):
        return self.res[index]

    def __len__(self):
        return self.n_residuals



class EulerViscousTime1D:
    def __init__(self, model, device='cpu', value=-5.0):
        self.model = model
        self.device = device
#        self.raw_mu = torch.nn.Parameter(torch.tensor([value], requires_grad=True, device=self.device, dtype=torch.float32))


    def compute_gradients(self, outputs, inputs):
        return autograd.grad(outputs=outputs, inputs=inputs,
                             grad_outputs=torch.ones_like(outputs),
                             create_graph=True, retain_graph=True)[0]
    
    def compute_loss(self, x_train):
        self.x_train = torch.tensor(x_train, device=self.device, dtype=torch.float32)
        self.gamma = torch.tensor(1.4, device=self.device, dtype=torch.float32)  # Specific heat ratio

        self.x_train.requires_grad = True

        output = self.model(self.x_train)
        rho, p, u, self.mu = output[:, 0], output[:, 1], output[:, 2], output[:, 3]**2

        E = p/(self.gamma - 1) + 0.5*rho*(u**2)
        s = torch.log(p/ rho**(self.gamma))

        U1 = rho
        U2 = rho*u
        U3 = E
        u_x = self.compute_gradients(u, self.x_train)[:,0]
        
        U1_x = self.compute_gradients(U1, self.x_train)[:,0]
        U2_x = self.compute_gradients(U2, self.x_train)[:,0]
        U3_x = self.compute_gradients(U3, self.x_train)[:,0]
        
        U1_xx = self.compute_gradients(U1_x, self.x_train)[:,0]
        U2_xx = self.compute_gradients(U2_x, self.x_train)[:,0]
        U3_xx = self.compute_gradients(U3_x, self.x_train)[:,0]
        
        U1_t = self.compute_gradients(U1, self.x_train)[:,1]
        U2_t = self.compute_gradients(U2, self.x_train)[:,1]
        U3_t = self.compute_gradients(U3, self.x_train)[:,1]

        f1 = rho*u
        f2 = rho * u**2 + p
        f3 = u*(E + p)

        f1_x = self.compute_gradients(f1, self.x_train)[:,0]
        f2_x = self.compute_gradients(f2, self.x_train)[:,0]
        f3_x = self.compute_gradients(f3, self.x_train)[:,0]
        
        self.lam = 1/(0.1*(torch.abs(u_x) - u_x) + 1)
        #self.lam = torch.abs(u_x)

        r1 = U1_t + f1_x - self.mu * U1_xx
        r2 = U2_t + f2_x - self.mu * U2_xx
        r3 = U3_t + f3_x - self.mu * U3_xx

        return r1, r2, r3


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

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

Xmin, Xmax = 0.0, 1.0
Tmin, Tmax = 0.0, 0.2



gamma = 1.4
r_left_inf = 1.0
p_left_inf = 1.0
u_left_inf = 0.0

r_right_inf = 0.125
p_right_inf = 0.1
u_right_inf = 0.0

vertices = [(Xmin, Tmin), (Xmax, Tmin), (Xmax, Tmax), (Xmin, Tmax)]  
polygon = PolygonBoundaryPoints(vertices, num_boundary_points=500)

boundary_points, edge_points_list = polygon.generate_points_on_edges()
interior_points = polygon.generate_random_points_inside(20000)

inputs = torch.tensor(interior_points, dtype=torch.float32, device=device)
layers=[2]+5*[192]+[4]

model = FNN(inputs, layers, init_type='xavier')
model.to(device)

losses=[]

data= {"Epoch":[],
        "Cont":[],
        "Mom_x":[],
        "Energy":[],
        "entropy":[],
        "Initialleft":[],
        "Initialright":[],
        "mu":[]
        }

filename  = 'loss.json'

class Loss:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.loss_fn = nn.MSELoss()  # renamed to avoid potential shadowing

    def LossPDE(self, coords, pde):
        self.pde = pde
        self.coords = torch.tensor(coords, dtype=torch.float32, device=self.device).clone().detach().requires_grad_(True)
        
        e1, e2, e3= self.pde.compute_loss(self.coords)

        loss = (self.loss_fn(e1 * self.pde.lam, torch.zeros_like(e1)),
                 self.loss_fn(e2 * self.pde.lam, torch.zeros_like(e2)), 
                 self.loss_fn(e3 * self.pde.lam, torch.zeros_like(e3)))

        mse = (self.loss_fn(e1 * self.pde.lam, torch.zeros_like(e1)) + 
               self.loss_fn(e2 * self.pde.lam, torch.zeros_like(e2)) + 
               self.loss_fn(e3 * self.pde.lam, torch.zeros_like(e3)))

        return loss, mse

    def LossInitial(self, coords, target):
        self.coords = torch.tensor(coords, dtype=torch.float32, device=self.device).clone().detach().requires_grad_(True)
        self.target = torch.tensor(target, dtype=torch.float32, device=self.device).clone().detach()

        output = self.model(self.coords)

        mse = self.loss_fn(self.target[:, 0], output[:,0]) + self.loss_fn(self.target[:, 1], output[:,1]) + self.loss_fn(self.target[:, 2], output[:,2])
        return mse
    
loss = Loss(model, device=device)
ns = EulerViscousTime1D(model, device=device, value = -5.0)


initial = edge_points_list[0]



def train(epochs=10000, lr = 0.001):

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    lambda1 = lambda epoch: 10**(-(2/epochs)*epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    
    for epoch in range(epochs):

        boundary_points, edge_list = polygon.generate_points_on_edges()
        collocation_points = polygon.generate_random_points_inside(20000)

        initial = edge_list[0]
        left = initial[initial[:,0] < 0.5]
        right = initial[initial[:,0] > 0.5]
        
        dataset = ResidualDataset(collocation_points)
        dataloader_ = DataLoader(dataset=dataset, batch_size=10000, shuffle=True, pin_memory=True)
        for residuals in dataloader_:

            left_rho=r_left_inf*np.ones_like(left[:,0]).reshape(-1,1)
            left_p = p_left_inf*np.ones_like(left[:,0]).reshape(-1,1)
            left_u = u_left_inf*np.ones_like(left[:,1]).reshape(-1,1)
           
            right_rho=r_right_inf*np.ones_like(right[:,0]).reshape(-1,1)
            right_p = p_right_inf*np.ones_like(right[:,0]).reshape(-1,1)
            right_u = u_right_inf*np.ones_like(right[:,1]).reshape(-1,1)

            left_condition = np.concatenate((left_rho,left_p, left_u),axis=1)
            right_condition = np.concatenate((right_rho,right_p, right_u),axis=1)

            pde_list, pde_mse = loss.LossPDE(residuals, ns)
            mse_left = loss.LossInitial(left, left_condition)
            mse_right = loss.LossInitial(right, right_condition)

            total = ( (pde_list[0] + pde_list[1] + pde_list[2]) + 10*(mse_left + mse_right)) + 0.1*torch.mean(ns.mu)
            
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            scheduler.step()
            
            loss_ = total.item()
        if epoch%100==0:

            data["Epoch"].append(epoch)
            data["Cont"].append(pde_list[0].item())
            data["Mom_x"].append(pde_list[1].item())
            data["Energy"].append(pde_list[2].item())
            data["Initialleft"].append(mse_left.item())
            data["Initialright"].append(mse_right.item())
  

            with open(filename, 'w') as f:
                json.dump(data, f)

            print(f"Epoch: {epoch}, Loss: {loss_:.4e}, PDE: {pde_mse.item()}")
            
        if epoch % 1000==0:
            torch.save(model.state_dict(), f'./results/model_adam.pth')
            step_size_space=0.01
            step_size_time=0.002
            x = torch.arange(Xmin, Xmax+step_size_space, step_size_space)
            y = torch.arange(Tmin, Tmax+step_size_time, step_size_time)

            X, Y = torch.meshgrid(x, y, indexing='ij')
            flat_X = X.flatten()
            flat_Y = Y.flatten()

            grid = torch.stack([flat_X, flat_Y], dim=1)
            grid = torch.tensor(grid, dtype=torch.float32, device=device)

            #model.save_predictions(grid, './predict/predictions_{}.npy'.format(epoch))
            model.save_predictions(grid, './predict/predictions.npy')
            
t0 = time.time()
train(epochs=50001, lr =1e-03)
total_time = (time.time()-t0)/60
print(f'Total: {total_time} min')
