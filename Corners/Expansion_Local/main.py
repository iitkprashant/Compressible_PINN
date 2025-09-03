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
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt



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
        self.optimizer = optim.Adam( list(model.parameters()) + list(model2.parameters()), lr=0.001)
        self.optimizer2 = torch.optim.LBFGS (list(model.parameters()) + list(model2.parameters()), max_iter=30, tolerance_grad=1e-8, line_search_fn='strong_wolfe')

        self.data={"Epoch":[],
                "Cont":[],
                "Mom_x":[],
                "Mom_y":[],
                "Energy":[],
                "Inlet":[],
                "base":[],
                "Ramp":[],
                "mu":[]
                }
    
    def compute_gradients(self, outputs, inputs):
        return autograd.grad(outputs=outputs, inputs=inputs,
                             grad_outputs=torch.ones_like(outputs),
                             create_graph=True, retain_graph=True)[0]
        
    def _physics_loss(self, x_train):
        x_train.requires_grad = True

        output = self.model(x_train)
        rho, p, u, v = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        self.mu = 0.01*self.model2(x_train)**2  # Dynamic viscosity

        E = p/(self.gamma - 1) + 0.5*rho*(u**2 + v**2)

        U1 = rho
        U2 = rho*u
        U3 = rho*v
        U4 = E

        u_x = self.compute_gradients(u, x_train)[:,0]
        v_y = self.compute_gradients(v, x_train)[:,1]
        
        U1_xy = self.compute_gradients(U1, x_train)
        U2_xy = self.compute_gradients(U2, x_train)
        U3_xy = self.compute_gradients(U3, x_train)
        U4_xy = self.compute_gradients(U4, x_train)

        U1_x, U1_y = U1_xy[:,0], U1_xy[:,1]
        U2_x, U2_y = U2_xy[:,0], U2_xy[:,1]
        U3_x, U3_y = U3_xy[:,0], U3_xy[:,1]
        U4_x, U4_y = U4_xy[:,0], U4_xy[:,1]

        U1_xx = self.compute_gradients(U1_x, x_train)[:,0]
        U2_xx = self.compute_gradients(U2_x, x_train)[:,0]
        U3_xx = self.compute_gradients(U3_x, x_train)[:,0]
        U4_xx = self.compute_gradients(U4_x, x_train)[:,0]

        U1_yy = self.compute_gradients(U1_y, x_train)[:,1]
        U2_yy = self.compute_gradients(U2_y, x_train)[:,1]
        U3_yy = self.compute_gradients(U3_y, x_train)[:,1]
        U4_yy = self.compute_gradients(U4_y, x_train)[:,1]

        f1 = rho*u
        f2 = rho * u**2 + p
        f3 = rho * u * v
        f4 = u*(E + p)

        f1_x = self.compute_gradients(f1, x_train)[:,0]
        f2_x = self.compute_gradients(f2, x_train)[:,0]
        f3_x = self.compute_gradients(f3, x_train)[:,0]
        f4_x = self.compute_gradients(f4, x_train)[:,0]

        g1 = rho*v
        g2 = rho * u * v
        g3 = rho * v**2 + p   
        g4 = v*(E + p)

        g1_y = self.compute_gradients(g1, x_train)[:,1]
        g2_y = self.compute_gradients(g2, x_train)[:,1]
        g3_y = self.compute_gradients(g3, x_train)[:,1]
        g4_y = self.compute_gradients(g4, x_train)[:,1]
   
        self.lam = 1/0.1*((torch.abs(u_x) + torch.abs(v_y) -(u_x + v_y)) + 1)

        res1 = f1_x + g1_y - self.mu*(U1_xx + U1_yy)
        r1 = torch.mean(res1**2)
        res2 = f2_x + g2_y - self.mu*(U2_xx + U2_yy)
        r2 = torch.mean(res2**2)
        res3 = f3_x + g3_y - self.mu*(U3_xx + U3_yy)
        r3 = torch.mean( res3**2)
        res4 = f4_x + g4_y - self.mu*(U4_xx + U4_yy)
        r4 = torch.mean( res4**2)

        return r1, r2, r3, r4
    
    def _inlet_loss(self, x_inlet, U_inlet):
        output = self.model(x_inlet)
        rho, p, u, v = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        loss = torch.mean((rho - U_inlet[:, 0])**2) +  torch.mean((p - U_inlet[:, 1])**2) +  torch.mean((u - U_inlet[:, 2])**2) + torch.mean( (v - U_inlet[:, 3])**2)
        return loss
    
    def _symmetry_loss(self, x_symmetry):
        output = self.model(x_symmetry)
        rho, p, u, v = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        loss = torch.mean( (v )**2)
        return loss
    
    def _slipNormal_loss(self, x_slipNormal):
        output = self.model(x_slipNormal)
        angle = -torch.tensor(torch.pi / 18, dtype=torch.float32, device=self.device)
        
        rho, p, u, v = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        transformed_output = - output[:, 2] * torch.sin(angle) + output[:, 3] * torch.cos(angle)
        loss = torch.mean((transformed_output )**2) 
        return loss
        
        
    def train(self, x_train, x_inlet, U_inlet, x_base, x_top, x_slipNormal, x_test, epochs=1000, batch_size=8192):
        train_dataset = TensorDataset(x_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.adam_epochs = epochs
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: 10**(-(2/10000)*epoch)
        )

        for epoch in range(epochs):
            epoch_loss = 0.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 10**(-(2/10000)*epoch)
            )

            for batch in train_loader:
                x_batch = batch[0]

                self.optimizer.zero_grad()
                r1, r2, r3, r4 = self._physics_loss(x_batch)
                physics_loss = r1 + r2 + r3 + r4

                inlet_loss = self._inlet_loss(x_inlet, U_inlet)
                base_loss = self._symmetry_loss(x_base)
                top_loss = self._symmetry_loss(x_top)
                slipNormal_loss = self._slipNormal_loss(x_slipNormal)

                total_loss = (
                    physics_loss
                    + 10 * (inlet_loss + base_loss + top_loss + slipNormal_loss)
                    + 0.1 * torch.mean(self.mu**2)  # Regularization term
                )
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()

            scheduler.step()

            if epoch % 100 == 0:
                self.data["Epoch"].append(epoch)
                self.data["Cont"].append(r1.item()) 
                self.data["Mom_x"].append(r2.item())
                self.data["Mom_y"].append(r3.item())
                self.data["Energy"].append(r4.item())
                self.data["Inlet"].append(inlet_loss.item())
                self.data["base"].append(base_loss.item())
                self.data["Ramp"].append(slipNormal_loss.item())
                self.data["mu"].append(torch.mean(self.mu).item())
                
                with open('loss.json', 'w') as f:
                    json.dump(self.data, f)
                    
                print(f"Epoch {epoch}, Total Loss: {epoch_loss:.4e}, Physics Loss: {physics_loss.item():.4e} mu: {torch.mean(self.mu).item():.4e}")
                torch.save(self.model.state_dict(), f'./results/model_adam.pth')

                step_size = 0.01
                x = torch.arange(0.0, 1.5 + step_size, step_size)
                y = torch.arange(0.0, 1.0 + step_size, step_size)
                X, Y = torch.meshgrid(x, y, indexing='ij')

                grid = torch.stack([X.flatten(), Y.flatten()], dim=1).to(self.device)

                self.model.save_predictions(grid, './predict/predictions.npy')
                self.model.save_predictions(x_test, './predict/predictions1.npy')

                
    def train_lbfgs(self, x_train, x_inlet, U_inlet, x_base, x_top, x_slipNormal, x_test, epochs=1000):
        def closure():
            self.optimizer2.zero_grad()
            r1, r2, r3, r4 = self._physics_loss(x_train)
            physics_loss = r1 + r2 + r3 + r4
            inlet_loss = self._inlet_loss(x_inlet, U_inlet)
            base_loss = self._symmetry_loss(x_base)
            top_loss = self._symmetry_loss(x_top)
            slipNormal_loss = self._slipNormal_loss(x_slipNormal)
            total_loss = physics_loss + 10*(inlet_loss + base_loss + top_loss + slipNormal_loss) + 0.1*torch.mean(self.mu**2)
            
            total_loss.backward()
            return total_loss
        
        for epoch in range(epochs):
            self.optimizer2.step(closure)
            if epoch % 10 == 0:
                total_loss = closure().item()
                r1, r2, r3, r4 = self._physics_loss(x_train)
                inlet_loss = self._inlet_loss(x_inlet, U_inlet)
                base_loss = self._symmetry_loss(x_base)
                top_loss = self._symmetry_loss(x_top)
                slipNormal_loss = self._slipNormal_loss(x_slipNormal)
                
                self.data["Epoch"].append(epoch+ self.adam_epochs)
                self.data["Cont"].append(r1.item()) 
                self.data["Mom_x"].append(r2.item())
                self.data["Mom_y"].append(r3.item())    
                self.data["Energy"].append(r4.item())
                self.data["Inlet"].append(inlet_loss.item())
                self.data["base"].append(base_loss.item())
                self.data["Ramp"].append(slipNormal_loss.item())    
                self.data["mu"].append(torch.mean(self.mu).item())
                
                with open('loss.json', 'w') as f:
                    json.dump(self.data, f)
                    
                print(f'Epoch {epoch}, Total Loss: {total_loss:.4e}, Physics Loss: {r1.item()+r2.item()+r3.item()+r4.item():.4e}, Inlet Loss: {inlet_loss.item():.4e}, Base Loss: {base_loss.item():.4e}, Top Loss: {top_loss.item():.4e}, Slip Normal Loss: {slipNormal_loss.item():.4e}, mu: {torch.mean(self.mu).item():.4e}')   
                
            if epoch % 100 == 0:
                torch.save(self.model.state_dict(), f'./results/model_lbfgs.pth')
                step_size=0.01
                x = torch.arange(0.0, 1.5+step_size, step_size)
                y = torch.arange(0.0, 1.0+step_size, step_size)

                X, Y = torch.meshgrid(x, y, indexing='ij')
                flat_X = X.flatten()
                flat_Y = Y.flatten()

                grid = torch.stack([flat_X, flat_Y], dim=1)
                grid = torch.tensor(grid, dtype=torch.float32, device=self.device)

                # coords = torch.tensor(coordi, dtype=torch.float32, device=self.device)            
                
                #model.save_predictions(grid, './predict/predictions_{}.npy'.format(epoch))
                self.model.save_predictions(grid, './predict/predictions.npy')
                self.model.save_predictions(x_test, './predict/predictions1.npy')
                self.model2.save_predictions(x_test, './predict/predictions2.npy')
                

if __name__ == "__main__":
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    vertices = [(0, 0), (0.5, 0.0), (1.5, -np.sin(np.pi/18)), (1.5, 1.0),(0.0, 1.0)]  
    polygon = PolygonBoundaryPoints(vertices, num_boundary_points=1000)

    boundary_points, edge_points_list = polygon.generate_points_on_edges()
    interior_points = polygon.generate_random_points_inside(40960)
    lbfgs_interior_points = polygon.generate_random_points_inside(30000)
    
    inputs = torch.tensor(interior_points, dtype=torch.float32, device=device)
    
    file_path = 'expansion.csv'
    dat = pd.read_csv(file_path)

    coordi = np.array([dat['X'], dat['Y']]).T
    
    layers=[2]+5*[96]+[4]
    #layers=[2]+4*[256]+[3]
    model = FNN(inputs, layers, init_type='xavier')
    model.to(device)
    layers2 = [2]+4*[96]+[1]
    model2 = FNN(inputs, layers2, init_type='xavier', output2='non_exp')
    
    model2.to(device)
    
    x_base  = torch.tensor(edge_points_list[0], dtype=torch.float32, device=device)
    x_top   = torch.tensor(edge_points_list[3], dtype=torch.float32, device=device)
    x_inlet = torch.tensor(edge_points_list[4], dtype=torch.float32, device=device)
    x_slipNormal = torch.tensor(edge_points_list[1], dtype=torch.float32, device=device)
    
    gamma = 1.4
    r_inf = 1.0
    p_inf = 1.0
    M_inf = 2.0
    v_inf = 0.0
    u_inf = np.sqrt(gamma*p_inf/r_inf)*M_inf
    
    inlet_rho=r_inf*torch.ones_like(x_inlet[:,0]).reshape(-1,1)
    inlet_p = p_inf*torch.ones_like(x_inlet[:,0]).reshape(-1,1)
    inlet_u = u_inf*torch.ones_like(x_inlet[:,0]).reshape(-1,1)
    inlet_v = v_inf*torch.ones_like(x_inlet[:,0]).reshape(-1,1)
    U_inlet = torch.cat((inlet_rho, inlet_p, inlet_u, inlet_v), dim=1).to(device)
    
    x_train = torch.tensor(interior_points, dtype=torch.float32, device=device)
    lbfgs_x_train = torch.tensor(lbfgs_interior_points, dtype=torch.float32, device=device)
    x_test = torch.tensor(coordi, dtype=torch.float32, device=device)
    
    pinn = PINN(model, model2, device=device)
    pinn.train(x_train, x_inlet, U_inlet, x_base, x_top, x_slipNormal, x_test, epochs=4001)
    pinn.train_lbfgs(lbfgs_x_train, x_inlet, U_inlet, x_base, x_top, x_slipNormal, x_test, epochs=501)
