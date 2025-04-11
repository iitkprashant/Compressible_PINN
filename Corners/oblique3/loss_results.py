import json
import matplotlib.pyplot as plt

# Load the JSON file
filename = 'loss.json'
with open(filename, 'r') as f:
    data = json.load(f)

filename1 = '../oblique2/loss.json'
with open(filename1, 'r') as f:
    data1 = json.load(f)

# Plot the data
# plt.figure(figsize=(10, 6))

# plt.plot(data["Epoch"], data["Cont"], label='continuity')
# plt.plot(data["Epoch"], data["Mom_x"], label='MOMX')
# plt.plot(data["Epoch"], data["Mom_y"], label='MOMY')
# plt.plot(data["Epoch"], data["Energy"], label='Energy')
# plt.plot(data["Epoch"], data["Inlet"], label='Inlet')
# plt.plot(data["Epoch"], data["base"], label='Base')
# plt.plot(data["Epoch"], data["Ramp"], label='Ramp')
# plt.yscale('log')
# plt.xlabel('Iteration')
# plt.ylabel('Values')
# plt.title('Variables over Iterations')
# plt.legend()
# plt.grid(True)
# plt.savefig('plot.png')
# plt.show()
plt.rcParams['font.family'] = 'serif'  # Set to desired font, e.g., 'serif', 'Times New Roman'
plt.rcParams['font.size'] = 18         # Set global font size
plt.rcParams['axes.labelsize'] = 18    # Font size for x and y axis labels
plt.rcParams['xtick.labelsize'] = 18   # Font size for x ticks
plt.rcParams['ytick.labelsize'] = 18   # Font size for y ticks
plt.rcParams['legend.fontsize'] = 18   # Font size for legend
plt.rcParams['axes.titleweight'] = 'bold'

variables = ["Cont", "Mom_x", "Mom_y", "Energy", "Inlet", "base", "Ramp"]
titles = ['Continuity', 'Momentum (X)', 'Momentum (Y)', 'Energy', 'Inlet', 'Base', 'Ramp']
colors = ['firebrick', 'firebrick', 'firebrick', 'firebrick', 'firebrick', 'firebrick', 'firebrick']

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), constrained_layout=True)
axes = axes.flatten()

for i, var in enumerate(variables):
    axes[i].plot(data["Epoch"], data[var], label=titles[i], color=colors[i])
    axes[i].set_yscale('log')
    # axes[i].set_title(titles[i], fontsize=14, pad=10)
    axes[i].set_xlabel('Epochs')
    axes[i].set_ylabel('LOss')
    axes[i].grid(True, alpha=0.4)
    axes[i].legend(loc='upper right')

# Hide the last empty subplot (if any)
for j in range(len(variables), len(axes)):
    fig.delaxes(axes[j])

# fig.suptitle('Variables Over Iterations', fontsize=22, weight='bold', y=1.02)
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()


# Group the variables for PDE and BC
pde_variables = ["Cont", "Mom_x", "Mom_y", "Energy"]
pde_titles = ['Continuity', 'Momentum (X)', 'Momentum (Y)', 'Energy']
pde_colors = ['firebrick'] * len(pde_variables)

bc_variables = ["Inlet", "base", "Ramp"]
bc_titles = ['Inlet', 'Base', 'Ramp']
bc_colors = ['firebrick'] * len(bc_variables)

# Plot PDE constraints
fig, axes = plt.subplots(nrows=1, ncols=len(pde_variables), figsize=(16, 4), constrained_layout=True)

for i, var in enumerate(pde_variables):
    axes[i].plot(data["Epoch"][:200], data[var][:200], label=pde_titles[i], color=pde_colors[i])
    axes[i].set_yscale('log')
    # axes[i].set_title(pde_titles[i], fontsize=14, pad=10)
    axes[i].set_xlabel('Epochs')
    axes[i].set_ylabel('Loss')
    axes[i].grid(True, alpha=0.4)
    axes[i].legend(loc='upper right')


plt.savefig('pde_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot BC variables
fig, axes = plt.subplots(nrows=1, ncols=len(bc_variables), figsize=(12, 4), constrained_layout=True)

for i, var in enumerate(bc_variables):
    axes[i].plot(data["Epoch"][:200], data[var][:200], label=bc_titles[i], color=bc_colors[i])
    axes[i].set_yscale('log')
    # axes[i].set_title(bc_titles[i], fontsize=14, pad=10)
    axes[i].set_xlabel('Epochs')
    axes[i].set_ylabel('Loss')
    axes[i].grid(True, alpha=0.4)
    axes[i].legend(loc='upper right')


plt.savefig('bc_plot.png', dpi=300, bbox_inches='tight')
plt.show()



plt.figure(figsize=(10, 6))

# Plot the data with enhancements
plt.plot(data["Epoch"][:200], data["mu"][:200], label=r'$\nu_{1}$ (Exp 1)', 
         linestyle='-', marker='o', markersize=2, linewidth=1.5, color='blue')
plt.plot(data1["Epoch"][:200], data1["mu"][:200], label=r'$\nu_{2}$ (Exp 2)', 
         linestyle='--', marker='s', markersize=2, linewidth=1.5, color='red')

plt.grid(alpha=0.3)
plt.xlabel('Epochs', weight='bold')
plt.ylabel(r'$\nu_{av}$', weight='bold')
plt.legend( loc='best', frameon=True, edgecolor='gray')
# plt.title('Convergence of Artificial Viscosity', fontsize=10, weight='bold')
plt.savefig('mu.png', dpi=300, bbox_inches='tight')
plt.show()


# print(data["mu"][-1])
# print(data1["mu"][-1])
print(data["Cont"][200])
print(data["Mom_x"][200])
print(data["Mom_y"][200])
print(data["Energy"][200])
print(data["Inlet"][200])
print(data["base"][200])
print(data["Ramp"][200])
