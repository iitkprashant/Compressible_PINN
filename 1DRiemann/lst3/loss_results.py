import json
import matplotlib.pyplot as plt

# Load the JSON file
filename = 'loss.json'
with open(filename, 'r') as f:
    data = json.load(f)

filename1 = '../lst2/loss.json'
with open(filename1, 'r') as f:
    data1 = json.load(f)
    
filename2 = '../lst/loss.json'
with open(filename2, 'r') as f:
    data2 = json.load(f)
    
    
# Plot the data
# plt.figure(figsize=(10, 6))

# plt.plot(data["Epoch"], data["Cont"], label='continuity')
# plt.plot(data["Epoch"], data["Mom_x"], label='MOMX')
# plt.plot(data["Epoch"], data["Energy"], label='Energy')
# plt.plot(data["Epoch"], data["Initialleft"], label='left')
# plt.plot(data["Epoch"], data["Initialright"], label='right')

# plt.yscale('log')
# plt.xlabel('Iteration')
# plt.ylabel('Values')
# plt.title('Variables over Iterations')
# plt.legend()
# plt.grid(True)
# plt.savefig('plot.png')
# plt.close()

plt.rcParams['font.family'] = 'serif'  # Set to desired font, e.g., 'serif', 'Times New Roman'
plt.rcParams['font.size'] = 18         # Set global font size
plt.rcParams['axes.labelsize'] = 18    # Font size for x and y axis labels
plt.rcParams['xtick.labelsize'] = 18   # Font size for x ticks
plt.rcParams['ytick.labelsize'] = 18   # Font size for y ticks
plt.rcParams['legend.fontsize'] = 18   # Font size for legend
plt.rcParams['axes.titleweight'] = 'bold'

fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

# Flatten axes for easier indexing
axs = axs.flatten()

# Plot each variable in separate subplots
variables = [
    ("Continuity", data["Cont"], 'firebrick', '-', 'o'),
    (r"Momentum ($x$)", data["Mom_x"], 'firebrick', '--', 's'),
    ("Energy", data["Energy"], 'firebrick', '-.', '^'),
    ("Initial Condition (Left)", data["Initialleft"], 'firebrick', '-', 'D'),
    ("Initial Condition (Right)", data["Initialright"], 'firebrick', '--', 'x')
]

for i, (label, values, color, linestyle, marker) in enumerate(variables):
    axs[i].plot(data["Epoch"], values, label=label, 
                linestyle=linestyle, marker=marker, markersize=2, linewidth=1.5, color=color)
    axs[i].set_yscale('log')
    axs[i].set_xlabel('Epochs', fontsize=18, weight='bold')
    axs[i].set_ylabel('Loss', fontsize=18, weight='bold')
    # axs[i].set_title(label, fontsize=18, weight='bold')
    axs[i].grid(alpha=0.3)
    axs[i].legend(fontsize=15, loc='best')

# Remove the unused subplot (last one in the 2x3 layout)
fig.delaxes(axs[-1])  # Removes the 6th subplot

# Save the figure
plt.savefig('plot.png', dpi=300, bbox_inches='tight')

# Display the plots
plt.show()

# PDE-related loss plots
fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

# Variables for PDE-related losses
pde_variables = [
    ("Continuity", data["Cont"], 'firebrick', '-', 'o'),
    (r"Momentum ($x$)", data["Mom_x"], 'firebrick', '--', 's'),
    ("Energy", data["Energy"], 'firebrick', '-.', '^'),
]

# Plot each PDE loss variable
for i, (label, values, color, linestyle, marker) in enumerate(pde_variables):
    axs[i].plot(data["Epoch"], values, label=label, 
                linestyle=linestyle, marker=marker, markersize=2, linewidth=1.5, color=color)
    axs[i].set_yscale('log')
    axs[i].set_xlabel('Epochs', fontsize=18)
    axs[i].set_ylabel('Loss', fontsize=18)
    # axs[i].set_title(label, fontsize=12, weight='bold')
    axs[i].grid(alpha=0.3)
    axs[i].legend(fontsize=15, loc='best')
    if i>0:
        axs[i].set_ylabel('')

# Save the figure for PDE losses
plt.savefig('pde_losses.png', dpi=300, bbox_inches='tight')
plt.show()

# Initial condition loss plots
fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# Variables for initial condition losses
ic_variables = [
    ("Initial Condition (Left)", data["Initialleft"], 'firebrick', '-', 'D'),
    ("Initial Condition (Right)", data["Initialright"], 'firebrick', '--', 'x'),
]

# Plot each IC loss variable
for i, (label, values, color, linestyle, marker) in enumerate(ic_variables):
    axs[i].plot(data["Epoch"], values, label=label, 
                linestyle=linestyle, marker=marker, markersize=2, linewidth=1.5, color=color)
    axs[i].set_yscale('log')
    axs[i].set_xlabel('Epochs', fontsize=18)
    axs[i].set_ylabel('Loss', fontsize=18)
    # axs[i].set_title(label, fontsize=12, weight='bold')
    axs[i].grid(alpha=0.3)
    axs[i].legend(fontsize=15, loc='best')
    if i>0:
        axs[i].set_ylabel('')

# Save the figure for IC losses
plt.savefig('ic_losses.png', dpi=300, bbox_inches='tight')
plt.show()
# plt.figure(figsize=(10, 6))
# plt.plot(data["Epoch"], data["mu"], label='mu')

# plt.yscale('log')
# plt.xlabel('Iteration')
# plt.ylabel('mu')
# plt.legend()
# plt.savefig('mu.png')

# Plot the data with enhancements
plt.figure(figsize=(10, 6))

plt.plot(data["Epoch"], data["mu"], label=r'$\nu_{1}$ (Exp 1)', 
         linestyle='-', marker='o', markersize=2, linewidth=1.5, color='blue')
plt.plot(data1["Epoch"][:158], data1["mu"][:158], label=r'$\nu_{2}$ (Exp 2)', 
         linestyle='--', marker='s', markersize=2, linewidth=1.5, color='red')
# plt.plot(data2["Epoch"], data2["mu"], label=r'$\mu_{3}$ (Exp 3)', 
#          linestyle='-.', marker='s', markersize=2, linewidth=1.5, color='green')

plt.grid(alpha=0.3)
plt.xlabel('Epochs', fontsize=18, weight='bold')
plt.ylabel(r'$\nu_{av}$', fontsize=18, weight='bold')
plt.legend(fontsize=10, loc='best', frameon=True, edgecolor='gray')
# plt.title('Convergence of Artificial Viscosity', fontsize=10, weight='bold')
plt.savefig('mu.png', dpi=300, bbox_inches='tight')
plt.show()


# print(data["mu"][-1])
# print(data1["mu"][-1])
print(data["Cont"][-1])
print(data["Mom_x"][-1])
print(data["Energy"][-1])
print(data["Initialleft"][-1])
print(data["Initialright"][-1])

