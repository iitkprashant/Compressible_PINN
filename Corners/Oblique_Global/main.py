import os
import sys
import json
import pickle
import numpy as np
import time
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.nn import init
import torch.nn.functional as F
import pickle
from torch.autograd import grad
from scipy.optimize import minimize
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class ResidualDataset(Dataset):
    def __init__(self, res_pts):
        self.res = res_pts
        self.n_residuals = self.res.shape[0]

    def __getitem__(self, index):
        return self.res[index]

    def __len__(self):
        return self.n_residuals

class PolygonBoundaryPoints:
    def __init__(self, vertices, num_boundary_points=400):
        """
        Initializes the polygon with n vertices.
        
        Args:
        vertices: List of vertices as (x, y) tuples.
        num_boundary_points: Number of points to generate on each boundary edge.
        """
        self.vertices = np.array(vertices)
        self.num_vertices = len(vertices)
        self.num_boundary_points = num_boundary_points
    
    def generate_points_on_edges(self):
        """
        Generates random points on the boundary edges of the polygon.
        """
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
        Generates random points inside the polygon using a random point selection method.
        
        Args:
        num_points_inside: Number of points to generate inside the polygon.
        """
        def random_points_in_polygon(vertices, num_points):
            n = len(vertices)
            points = []
            for _ in range(num_points):
                # Generate random barycentric coordinates for each vertex triplet
                selected_vertices = np.random.choice(n, 3, replace=False)
                v1, v2, v3 = vertices[selected_vertices]
                u = np.random.rand()
                v = np.random.rand()
                if u + v > 1:
                    u, v = 1 - u, 1 - v
                w = 1 - u - v
                point = u * v1 + v * v2 + w * v3
                points.append(point)
            return np.array(points)
        
        interior_points = random_points_in_polygon(self.vertices, num_points_inside)
        return interior_points

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

class FNN(nn.Module):
    def __init__(self, inputs, layers, init_type='xavier'):
        super(FNN, self).__init__()

        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.mean = torch.mean(self.inputs, axis=0)
        self.std = torch.std(self.inputs, axis=0)

        self.layers = layers
        self.init_type = init_type

        self.num_layers = len(layers) - 1
        self.hidden_weights = nn.ParameterList()
        self.hidden_biases = nn.ParameterList()

        for i in range(self.num_layers):
            weight = nn.Parameter(torch.empty(layers[i], layers[i + 1]))
            bias = nn.Parameter(torch.empty(1, layers[i + 1]))
            self.hidden_weights.append(weight)
            self.hidden_biases.append(bias)

        self.initialize_weights()

    def initialize_weights(self):
        for weight in self.hidden_weights:
            if self.init_type == 'xavier':
                init.xavier_normal_(weight)
            elif self.init_type == 'he':
                init.kaiming_normal_(weight, mode='fan_in', nonlinearity='relu')

        for bias in self.hidden_biases:
            init.constant_(bias, 0)
                    
    def forward(self, x):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        x = (x - self.mean) / self.std

        for i, (weight, bias) in enumerate(zip(self.hidden_weights, self.hidden_biases)):
            H = torch.matmul(x, weight) + bias
            if i < self.num_layers - 1:
                x = F.tanh(H)  # Use SiLU (Sigmoid Linear Unit) for hidden layers
            else:
                x = H  # No activation for the output layer
        
        x[:, 0] = torch.exp(x[:, 0])  # Exponential for the first output
        x[:, 1] = torch.exp(x[:, 1])  # Exponential for the second output
        return x

    def print_state_dict(self):
        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            prediction = self.forward(x)
        return prediction

    def save_predictions(self, x_input, filename):
        predictions = self.predict(x_input)
        predictions_np = predictions.detach().cpu().numpy()
        input_np = x_input.detach().cpu().numpy()
        data = np.concatenate((input_np, predictions_np), axis=1)
        np.save(filename, data)

    def save_parameters(self, filepath):
        params = {
            'hidden_weights': [weight.detach().cpu().numpy() for weight in self.hidden_weights],
            'hidden_biases': [bias.detach().cpu().numpy() for bias in self.hidden_biases],
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)


class EulerViscous2D:
    def __init__(self, model, device='cpu', value = -5.0):
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
        rho, p, u, v = output[:, 0], output[:, 1], output[:, 2], output[:, 3]

        E = p/(self.gamma - 1) + 0.5*rho*(u**2 + v**2)

        U1 = rho
        U2 = rho*u
        U3 = rho*v
        U4 = E

        U1_xy = self.compute_gradients(U1, self.x_train)
        U2_xy = self.compute_gradients(U2, self.x_train)
        U3_xy = self.compute_gradients(U3, self.x_train)
        U4_xy = self.compute_gradients(U4, self.x_train)

        U1_x, U1_y = U1_xy[:,0], U1_xy[:,1]
        U2_x, U2_y = U2_xy[:,0], U2_xy[:,1]
        U3_x, U3_y = U3_xy[:,0], U3_xy[:,1]
        U4_x, U4_y = U4_xy[:,0], U4_xy[:,1]

        U1_xx = self.compute_gradients(U1_x, self.x_train)[:,0]
        U2_xx = self.compute_gradients(U2_x, self.x_train)[:,0]
        U3_xx = self.compute_gradients(U3_x, self.x_train)[:,0]
        U4_xx = self.compute_gradients(U4_x, self.x_train)[:,0]

        U1_yy = self.compute_gradients(U1_y, self.x_train)[:,1]
        U2_yy = self.compute_gradients(U2_y, self.x_train)[:,1]
        U3_yy = self.compute_gradients(U3_y, self.x_train)[:,1]
        U4_yy = self.compute_gradients(U4_y, self.x_train)[:,1]

        f1 = rho*u
        f2 = rho * u**2 + p
        f3 = rho * u * v
        f4 = u*(E + p)

        f1_x = self.compute_gradients(f1, self.x_train)[:,0]
        f2_x = self.compute_gradients(f2, self.x_train)[:,0]
        f3_x = self.compute_gradients(f3, self.x_train)[:,0]
        f4_x = self.compute_gradients(f4, self.x_train)[:,0]

        g1 = rho*v
        g2 = rho * u * v
        g3 = rho * v**2 + p   
        g4 = v*(E + p)

        g1_y = self.compute_gradients(g1, self.x_train)[:,1]
        g2_y = self.compute_gradients(g2, self.x_train)[:,1]
        g3_y = self.compute_gradients(g3, self.x_train)[:,1]
        g4_y = self.compute_gradients(g4, self.x_train)[:,1]
    
        r1 = f1_x + g1_y - self.mu*(U1_xx + U1_yy)
        r2 = f2_x + g2_y - self.mu*(U2_xx + U2_yy)
        r3 = f3_x + g3_y - self.mu*(U3_xx + U3_yy)
        r4 = f4_x + g4_y - self.mu*(U4_xx + U4_yy)

        return r1, r2, r3, r4



device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

gamma = 1.4
r_inf = 1.0
p_inf = 1.0
M_inf = 2.0
v_inf = 0.0
u_inf = np.sqrt(gamma*p_inf/r_inf)*M_inf
print(u_inf)
print(v_inf)


vertices = [(0, 0), (0.5, 0.0), (1.5, np.sin(np.pi/18)), (1.5, 1.0),(0.0, 1.0)]  
polygon = PolygonBoundaryPoints(vertices, num_boundary_points=500)

boundary_points, edge_points_list = polygon.generate_points_on_edges()
interior_points = polygon.generate_random_points_inside(20000)

base= edge_points_list[0]
ramp = edge_points_list[1]
outlet = edge_points_list[2]
top = edge_points_list[3]
inlet = edge_points_list[4]

plt.figure()
plt.scatter(base[:,0], base[:,1],s=1, label='base')
plt.scatter(ramp[:,0], ramp[:,1],s=1, label='ramp')
plt.scatter(outlet[:,0], outlet[:,1],s=1, label='outlet')
plt.scatter(top[:,0], top[:,1], s=1, label='top')
plt.scatter(inlet[:,0], inlet[:,1], s=1, label='inlet')
plt.legend()
plt.savefig('test.png')
plt.close()

inputs = torch.tensor(interior_points, dtype=torch.float32, device=device)
layers=[2]+5*[72]+[4]
#layers=[2]+4*[256]+[3]
model = FNN(inputs, layers, init_type='xavier')
model.to(device)

losses=[]

data= {"Epoch":[],
        "Cont":[],
        "Mom_x":[],
        "Mom_y":[],
        "Energy":[],
        "Inlet":[],
        "base":[],
        "Ramp":[],
        "mu":[]
        }

filename  = 'loss.json'

file_path = 'compression.csv'
dat = pd.read_csv(file_path)

coordi = np.array([dat['X'], dat['Y']]).T

class Loss:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.loss_fn = nn.MSELoss()  # renamed to avoid potential shadowing

    def LossPDE(self, coords, pde):
        self.pde = pde
        self.coords = torch.tensor(coords, dtype=torch.float32, device=self.device).clone().detach().requires_grad_(True)
        
        e1, e2, e3, e4 = self.pde.compute_loss(self.coords)

        loss = (self.loss_fn(e1, torch.zeros_like(e1)),
                 self.loss_fn(e2, torch.zeros_like(e2)), 
                 self.loss_fn(e3, torch.zeros_like(e3)), 
                 self.loss_fn(e4, torch.zeros_like(e4)))

        mse = (self.loss_fn(e1, torch.zeros_like(e1)) + 
               self.loss_fn(e2, torch.zeros_like(e2)) + 
               self.loss_fn(e3, torch.zeros_like(e3)) + 
               self.loss_fn(e4, torch.zeros_like(e4)))

        return loss, mse

    def LossInlet(self, coords, target):
        self.coords = torch.tensor(coords, dtype=torch.float32, device=self.device).clone().detach().requires_grad_(True)
        self.target = torch.tensor(target, dtype=torch.float32, device=self.device).clone().detach()

        output = self.model(self.coords)
        mse = self.loss_fn(self.target[:, 0], output[:,0]) + self.loss_fn(self.target[:, 1], output[:,1]) + self.loss_fn(self.target[:, 2], output[:,2])+ self.loss_fn(self.target[:, 3], output[:,3])
        return mse
    
    def LossSymmetry(self, coords, target):
        self.coords = torch.tensor(coords, dtype=torch.float32, device=self.device).clone().detach().requires_grad_(True)
        self.target = torch.tensor(target, dtype=torch.float32, device=self.device).clone().detach()

        output = self.model(self.coords)

        mse = self.loss_fn(self.target, output[:,3])
        return mse
    
    def LossSlipNormal(self, coords, target):
        # Detach from computation graph and make tensors require gradient if necessary
        coords = torch.tensor(coords, dtype=torch.float32, device=self.device).clone().detach().requires_grad_(True)
        target = torch.tensor(target, dtype=torch.float32, device=self.device).clone().detach()

        output = self.model(coords)
        angle = torch.tensor(torch.pi / 18, dtype=torch.float32, device=self.device)
        transformed_output = - output[:, 2] * torch.sin(angle) + output[:, 3] * torch.cos(angle)
        mse = self.loss_fn(target, transformed_output)

        return mse

    
loss = Loss(model, device=device)
ns = EulerViscous2D(model, device=device, value=-6.0)


def train(epochs=10000, lr = 0.001):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer1 = optim.Adam([ns.raw_mu], lr=0.001)

    lambda1 = lambda epoch: 10**(-(2/epochs)*epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    num_points = 10000
    for epoch in range(epochs):

        boundary_points, edge_list = polygon.generate_points_on_edges()
        collocation_points = polygon.generate_random_points_inside(40000)

        base= edge_list[0]
        ramp = edge_list[1]
        top = edge_list[3]
        inlet = edge_list[4]

        dataset = ResidualDataset(collocation_points)
        dataloader_ = DataLoader(dataset=dataset, batch_size=5000, shuffle=True, pin_memory=True)
        
        inlet_rho=np.ones_like(inlet[:,0]).reshape(-1,1)
        inlet_p = np.ones_like(inlet[:,0]).reshape(-1,1)
        inlet_u = u_inf*np.ones_like(inlet[:,0]).reshape(-1,1)
        inlet_v = v_inf*np.ones_like(inlet[:,0]).reshape(-1,1)
        
        slip_ramp = np.zeros_like(ramp[:,0]).reshape(-1,1)

        inlet_condition = np.concatenate((inlet_rho,inlet_p, inlet_u, inlet_v),axis=1)
        slip_base = v_inf*np.ones_like(base[:,1]).reshape(-1,1)
        slip_top = v_inf*np.ones_like(top[:,1]).reshape(-1,1)
        
        
        for residuals in dataloader_:

            pde_list, pde_mse = loss.LossPDE(residuals, ns)
            mse_inlet = loss.LossInlet(inlet, inlet_condition)
            mse_base = loss.LossSymmetry(base, slip_base)
            mse_top= loss.LossSymmetry(top, slip_top)
            mse_ramp = loss.LossSlipNormal(ramp, slip_ramp)

            total = ((pde_list[0] + pde_list[1] + pde_list[2] + pde_list[3]) + 10*(mse_inlet + mse_base + mse_ramp + mse_top))
            
            optimizer.zero_grad()
            optimizer1.zero_grad()
            total.backward()
            optimizer.step()
            optimizer1.step()
            scheduler.step()
            
            loss_ = total.item()
        if epoch%100==0:

            data["Epoch"].append(epoch)
            data["Cont"].append(pde_list[0].item())
            data["Mom_x"].append(pde_list[1].item())
            data["Mom_y"].append(pde_list[2].item())
            data["Energy"].append(pde_list[3].item())
            data["Inlet"].append(mse_inlet.item())
            data["base"].append(mse_base.item())
            data["Ramp"].append(mse_ramp.item())
            data["mu"].append(ns.mu.item())


            with open(filename, 'w') as f:
                json.dump(data, f)

            print(f"Epoch: {epoch}, Loss: {loss_:.4e}, PDE: {pde_mse.item()}, mu : {ns.mu.item()}")
            
        if epoch % 1000==0:
            torch.save(model.state_dict(), f'./results/model_adam.pth')
            step_size=0.01
            x = torch.arange(0.0, 1.5+step_size, step_size)
            y = torch.arange(0.0, 1.0+step_size, step_size)

            X, Y = torch.meshgrid(x, y, indexing='ij')
            flat_X = X.flatten()
            flat_Y = Y.flatten()

            grid = torch.stack([flat_X, flat_Y], dim=1)
            grid = torch.tensor(grid, dtype=torch.float32, device=device)

            coords = torch.tensor(coordi, dtype=torch.float32, device=device)            
            
            #model.save_predictions(grid, './predict/predictions_{}.npy'.format(epoch))
            model.save_predictions(grid, './predict/predictions.npy')
            model.save_predictions(coords, './predict/predictions1.npy')

            
t0 = time.time()
train(epochs=50001, lr =1e-03)
total_time = (time.time()-t0)/60
print(f'Total: {total_time} min')
