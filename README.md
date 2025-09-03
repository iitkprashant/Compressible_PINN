# Physics-Informed Neural Networks for Euler Equations with Artificial Viscosity

This repository implements Physics-Informed Neural Networks (PINNs) for solving the Euler equations with global and local artificial viscosity models. It includes benchmark problems in 1D and 2D, including Riemann problems and corner flows.

## 📂 Repository Structure

```
.
├── 1DRiemann/
│   ├── LST_Global/          # Lax-Shock Tube with Global Viscosity
│   ├── LST_Local/           # Lax-Shock Tube with Local Viscosity
│   ├── SST_Global/          # Sod-Shock Tube with Global Viscosity
│   ├── SST_Local/           # Sod-Shock Tube with Local Viscosity
│   └── main.py              # Main script for 1D problems
│
├── 2DRiemann/
│   ├── 2DRiemann1_Global/   # 2D Riemann Problem 1 - Global
│   ├── 2DRiemann1_Local/    # 2D Riemann Problem 1 - Local
│   ├── 2DRiemann2_Global/   # 2D Riemann Problem 2 - Global
│   ├── 2DRiemann2_Local/    # 2D Riemann Problem 2 - Local
│   ├── 2DRiemann3_Global/   # 2D Riemann Problem 3 - Global
│   └── 2DRiemann3_Local/    # 2D Riemann Problem 3 - Local
│
└── Corners/
    ├── Expansion_Global/    # Corner Expansion - Global Viscosity
    ├── Expansion_Local/     # Corner Expansion - Local Viscosity
    ├── Compression_Global/  # Corner Compression - Global Viscosity
    └── Compression_Local/   # Corner Compression - Local Viscosity
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

## 🏃‍♂️ Running the Code

### 1D Riemann Problems
```bash
cd 1DRiemann/LST_Global
python main.py
```

Replace `LST_Global` with the desired configuration:
- `LST_Local`
- `SST_Global`
- `SST_Local`

### 2D Riemann Problems
Navigate to any of the 2D Riemann problem directories and run the respective main script.

### Corner Flows
Navigate to either the Expansion or Compression directories and run the corresponding scripts for global or local viscosity models.

## 📌 Problem Setup

### 1D Riemann Problems
- **LST (Lax-Shock Tube)**: Classical shock tube problem with specified initial conditions
- **SST (Sod-Shock Tube)**: Another common shock tube configuration

### 2D Riemann Problems
Three different configurations of 2D Riemann problems with varying initial conditions.

### Corner Flows
- **Expansion Flow**: Flow expansion around a corner
- **Compression Flow**: Flow compression around a corner


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.