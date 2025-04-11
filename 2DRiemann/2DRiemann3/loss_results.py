import json
import matplotlib.pyplot as plt


# Load the JSON file
filename = 'loss.json'
with open(filename, 'r') as f:
    data = json.load(f)


plt.rcParams['font.family'] = 'serif'  # Set to desired font, e.g., 'serif', 'Times New Roman'
plt.rcParams['font.size'] = 18         # Set global font size
plt.rcParams['axes.labelsize'] = 18    # Font size for x and y axis labels
plt.rcParams['xtick.labelsize'] = 18   # Font size for x ticks
plt.rcParams['ytick.labelsize'] = 18   # Font size for y ticks
plt.rcParams['legend.fontsize'] = 18   # Font size for legend
plt.rcParams['axes.titleweight'] = 'bold'


plt.figure(figsize=(10, 6))

# Plot the data with enhancements
plt.plot(data["Epoch"], data["mu"], label=r'$\nu_{1}$ (Exp 1)', 
         linestyle='-', marker='o', markersize=2, linewidth=1.5, color='blue')


plt.grid(alpha=0.3)
plt.xlabel('Epochs', weight='bold')
plt.ylabel(r'$\nu_{av}$', weight='bold')
plt.legend(fontsize=10, loc='best', frameon=True, edgecolor='gray')
# plt.title('Convergence of Artificial Viscosity', fontsize=10, weight='bold')
plt.savefig('mu.png', dpi=300, bbox_inches='tight')
plt.show()

print(data["mu"][-1])



fig, axs = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)

# Flatten axes for easier indexing
axs = axs.flatten()

# Plot each variable in separate subplots
variables = [
    ("Continuity", data["Cont"], 'firebrick', '-', 'o'),
    (r"Momentum ($x$)", data["Mom_x"], 'firebrick', '--', 's'),
    (r"Momentum ($y$)", data["Mom_y"], 'firebrick', '--', 's'),
    ("Energy", data["Energy"], 'firebrick', '-.', '^'),
    ("IC (LeftTop)", data["lefttop"], 'firebrick', '-', 'D'),
    ("IC (LeftBottom)", data["leftbottom"], 'firebrick', '--', 'x'),
    ("IC (RightTop)", data["righttop"], 'firebrick', '-', 'D'),
    ("IC (RightBottom)", data["rightbottom"], 'firebrick', '--', 'x')
]

for i, (label, values, color, linestyle, marker) in enumerate(variables):
    axs[i].plot(data["Epoch"], values, label=label, 
                linestyle=linestyle, marker=marker, markersize=2, linewidth=1.5, color=color)
    axs[i].set_yscale('log')
    axs[i].set_xlabel('Epochs', fontsize=18)
    axs[i].set_ylabel('Loss', fontsize=18)
    # axs[i].set_title(label, fontsize=12, weight='bold')
    axs[i].grid(alpha=0.3)
    axs[i].legend(fontsize=15, loc='best')

# Save the figure
plt.savefig('plot.png', dpi=300, bbox_inches='tight')

# Display the plots
plt.show()


# PDE-related loss plots
fig, axs = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)

# Variables for PDE-related losses
pde_variables = [
    ("Continuity", data["Cont"], 'firebrick', '-', 'o'),
    (r"Momentum ($x$)", data["Mom_x"], 'firebrick', '--', 's'),
    (r"Momentum ($y$)", data["Mom_y"], 'firebrick', '--', 's'),
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
fig, axs = plt.subplots(1, 4, figsize=(14, 5), constrained_layout=True)

# Variables for initial condition losses
ic_variables = [
    ("IC (LeftTop)", data["lefttop"], 'firebrick', '-', 'D'),
    ("IC (LeftBottom)", data["leftbottom"], 'firebrick', '--', 'x'),
    ("IC (RightTop)", data["righttop"], 'firebrick', '-', 'D'),
    ("IC (RightBottom)", data["rightbottom"], 'firebrick', '--', 'x')
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

# print(data["Cont"][300])
# print(data["Mom_x"][300])
# print(data["Energy"][300])
# print(data["Initialleft"][300])
# print(data["Initialright"][300])