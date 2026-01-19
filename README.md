# Physics-Informed Neural Networks for Compressible Flows for Shocks.

This repository implements Physics-Informed Neural Networks (PINNs) for solving the Compressible Euler equations with learnable global and local artificial viscosity models. It includes benchmark problems in 1D and 2D, including Riemann problems and corner flows.

## 📂 Repository Structure

```
.
├── 1DRiemann/
│   ├── LST_Global/          # Lax-Shock Tube with Global Viscosity
│   │   ├── main.py          # Main execution script
│   │   ├── plot.py          # Contour visualization script
│   │   └── loss_results.py  # Loss convergence plotting
│   ├── LST_Local/           # Lax-Shock Tube with Local Viscosity
│   │   ├── main.py
│   │   ├── plot.py
│   │   └── loss_results.py
│   ├── SST_Global/          # Sod-Shock Tube with Global Viscosity
│   │   ├── main.py
│   │   ├── plot.py
│   │   └── loss_results.py
│   └── SST_Local/           # Sod-Shock Tube with Local Viscosity
│       ├── main.py
│       ├── plot.py
│       └── loss_results.py
│  
├── 2DRiemann/
│   ├── 2DRiemann1_Global/   # 2D Riemann Problem 1 - Global
│   │   ├── main.py
│   │   ├── plot.py
│   │   └── loss_results.py
│   ├── 2DRiemann1_Local/    # 2D Riemann Problem 1 - Local
│   │   ├── main.py
│   │   ├── plot.py
│   │   └── loss_results.py
│   ├── 2DRiemann2_Global/   # 2D Riemann Problem 2 - Global
│   │   ├── main.py
│   │   ├── plot.py
│   │   └── loss_results.py
│   ├── 2DRiemann2_Local/    # 2D Riemann Problem 2 - Local
│   │   ├── main.py
│   │   ├── plot.py
│   │   └── loss_results.py
│   ├── 2DRiemann3_Global/   # 2D Riemann Problem 3 - Global
│   │   ├── main.py
│   │   ├── plot.py
│   │   └── loss_results.py
│   └── 2DRiemann3_Local/    # 2D Riemann Problem 3 - Local
│       ├── main.py
│       ├── plot.py
│       └── loss_results.py
│
└── Corners/
    ├── Expansion_Global/    # Corner Expansion - Global Viscosity
    │   ├── main.py
    │   ├── plot.py
    │   └── loss_results.py
    ├── Expansion_Local/     # Corner Expansion - Local Viscosity
    │   ├── main.py
    │   ├── plot.py
    │   └── loss_results.py
    ├── Compression_Global/  # Corner Compression - Global Viscosity
    │   ├── main.py
    │   ├── plot.py
    │   └── loss_results.py
    └── Compression_Local/   # Corner Compression - Local Viscosity
        ├── main.py
        ├── plot.py
        └── loss_results.py
```

## 🚀 Features

- **PINN-based Solvers**: Physics-informed neural networks for solving Euler equations in 1D and 2D
- **Artificial Viscosity Models**:
  - Self-learnable Global Artificial Viscosity
  - Two-Network Local Artificial Viscosity
- **Benchmark Problems**:
  - 1D Riemann problems (Lax-Shock Tube and Sod-Shock Tube)
  - 2D Riemann problems (3 configurations)
  - Corner flows (Expansion and Compression cases)
- **Visualization Tools**:
  - Contour plotting for solution visualization
  - Loss convergence tracking and visualization

## 🏃‍♂️ Running the Code

Each directory contains three main scripts:

1. `main.py` - Runs the PINN simulation
2. `plot.py` - Generates contour plots of the results
3. `loss_results.py` - Plots loss convergence curves

### 1D Riemann Problems
```bash
# Navigate to desired configuration
cd 1DRiemann/LST_Global

# Run the simulation
python main.py

# Generate contour plots
python plot.py

# Plot loss convergence
python loss_results.py
```

Available configurations:
- `LST_Global` - Lax-Shock Tube with Global Viscosity
- `LST_Local` - Lax-Shock Tube with Local Viscosity
- `SST_Global` - Sod-Shock Tube with Global Viscosity
- `SST_Local` - Sod-Shock Tube with Local Viscosity

### 2D Riemann Problems
Navigate to any of the 2D Riemann problem directories and run the same set of commands.

### Corner Flows
Navigate to either the Expansion or Compression directories and run the scripts for global or local viscosity models.

## 📖 Citation

If you use this code in your research, please cite:
```bibtex
@article{kumar2025shocks,
  title = {A Robust Data-Free Physics-Informed Neural Network for Compressible Flows with Shocks}},
  author = {Prashant Kumar and Rajesh Ranjan},
  doi = {https://doi.org/10.1016/j.compfluid.2026.106975},
  url = {https://www.sciencedirect.com/science/article/pii/S0045793026000174},
  year = {2026},
  journal = {Computer & Fluids}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
