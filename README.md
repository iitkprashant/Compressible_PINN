Physics-Informed Neural Network (PINN) for Euler Equations with Artificial Viscosity

This repository contains implementations of Physics-Informed Neural Networks (PINNs) for solving the Euler equations with global and local artificial viscosity models.
It provides both 1D Riemann problems and 2D Riemann problems, along with corner flow problems (expansion and compression cases).


📂 Repository Structure
.
├── 1DRiemann/
│   ├── LST_Global/
│   ├── LST_Local/
│   ├── SST_Global/
│   ├── SST_Local/
│   └── main.py
│
├── 2DRiemann/
│   ├── 2DRiemann1_Global/
│   ├── 2DRiemann1_Local/
│   ├── 2DRiemann2_Global/
│   ├── 2DRiemann2_Local/
│   ├── 2DRiemann3_Global/
│   └── 2DRiemann3_Local/
│
└── Corners/
    ├── Expansion_Global/
    ├── Expansion_Local/
    ├── Compression_Global/
    └── Compression_Local/


⸻

🚀 Features
	•	PINN-based solvers for Euler equations in 1D and 2D.
	•	Incorporates artificial viscosity for stability and shock capturing:
	•	Self Learnable Global Artificial Viscosity Model
	•	Two Network Local Artificial Viscosity Model
	•	Benchmark problems:
	•	1D Riemann problems (2 cases)
	•	2D Riemann problems (3 cases)
	•	Corner expansion and compression flows

⸻

📌 Problem Setup

1D Riemann Problems
	•	Implemented in the 1DRiemann/ folder.
	•	Subfolders correspond to:
	•	LST (Lax-Shock Tube)
	•	SST (Sod-Shock Tube)
	•	Each case has Global and Local viscosity models.
	•	Run with:

cd 1DRiemann/LST_Global
python main.py



2D Riemann Problems
	•	Implemented in the 2DRiemann/ folder.
	•	Three test cases:
	•	2DRiemann1, 2DRiemann2, 2DRiemann3
	•	Each test case includes both Global and Local viscosity versions.

Corner Flows
	•	Implemented in the Corners/ folder.
	•	Includes:
	•	Expansion flow (Global / Local)
	•	Compression flow (Global / Local)
