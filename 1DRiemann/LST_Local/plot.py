import numpy as np
import csv
import matplotlib.pyplot as plt

def read_csv_data(filename):
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # Skip the header row
        data = np.array([list(map(float, row)) for row in csv_reader])
    return data, header

filepath = './predict/predictions1.npy'
data = np.load(filepath)

filename = "time_step_162.csv" 

data1, header = read_csv_data(filename)

# Extract data columns
X = data1[:, 0]       # x-coordinates
RHO = data1[:, 1]     # Density
U = data1[:, 2]       # Velocity
P = data1[:, 3]       # Pressure
e = data1[:, 4]       # Energy
rho_ref = data1[:, 6] # Reference Density
u_ref = data1[:, 7]   # Reference Velocity
P_ref = data1[:, 8]   # Reference Pressure
e_ref = data1[:, 9]   # Reference Energy

print(data.shape)
#print(data)
Xmin = np.min(data[:,0])
Xmax = np.max(data[:,0])


Ymin = np.min(data[:,1])
Ymax = np.max(data[:,1])

x = data[:,0].reshape(101,71) 
y = data[:,1].reshape(101,71)
rho = data[:,2].reshape(101,71)
u = data[:,4].reshape(101,71) 
p = data[:,3].reshape(101,71) 

plt.rcParams['font.family'] = 'serif'  # Set to desired font, e.g., 'serif', 'Times New Roman'
plt.rcParams['font.size'] = 18         # Set global font size
plt.rcParams['axes.labelsize'] = 18    # Font size for x and y axis labels
plt.rcParams['xtick.labelsize'] = 18   # Font size for x ticks
plt.rcParams['ytick.labelsize'] = 18   # Font size for y ticks
plt.rcParams['legend.fontsize'] = 18   # Font size for legend
# plt.rcParams['axes.titleweight'] = 'bold'

plt.figure(figsize=(8, 6))
contour = plt.contourf(x, y, u, cmap='rainbow', levels=100, extend='both')
cbar = plt.colorbar(contour, label='U-velocity')
cbar.ax.tick_params(labelsize=18)  # Colorbar tick size
plt.xlim([Xmin, Xmax])
plt.ylim([Ymin, Ymax])
plt.xlabel('X (Position)')
plt.ylabel('T (Time)')
# plt.title('U-Velocity Contour', fontsize=14, weight='bold')
plt.savefig('contour1.png', dpi=200, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
contour = plt.contourf(x, y, p, cmap='rainbow', levels=100, extend='both')
cbar = plt.colorbar(contour, label='Pressure')
cbar.ax.tick_params(labelsize=18)
plt.xlim([Xmin, Xmax])
plt.ylim([Ymin, Ymax])
plt.xlabel('X (Position)')
plt.ylabel('T (Time)')
# plt.title('Pressure Contour', fontsize=14, weight='bold')
plt.savefig('contour2.png', dpi=200, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
contour = plt.contourf(x, y, rho, cmap='rainbow', levels=100, extend='both')
cbar = plt.colorbar(contour, label='Density')
cbar.ax.tick_params(labelsize=18)
plt.xlim([Xmin, Xmax])
plt.ylim([Ymin, Ymax])
plt.xlabel('X (Position)')
plt.ylabel('T (Time)')
# plt.title('Density Contour', fontsize=14, weight='bold')
plt.savefig('contour3.png', dpi=200, bbox_inches='tight')
plt.show()

plt.rcParams['font.family'] = 'serif'  # Set to desired font, e.g., 'serif', 'Times New Roman'
plt.rcParams['font.size'] = 18         # Set global font size
plt.rcParams['axes.labelsize'] = 18    # Font size for x and y axis labels
plt.rcParams['xtick.labelsize'] = 18   # Font size for x ticks
plt.rcParams['ytick.labelsize'] = 18   # Font size for y ticks
plt.rcParams['legend.fontsize'] = 18   # Font size for legend
# plt.rcParams['axes.titleweight'] = 'bold'

fig, axs = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

# Density
axs[0].set_title("Density", fontsize=18)
axs[0].plot(X, RHO, label='Roe', linestyle='--', color='blue', linewidth=1.5)
axs[0].plot(X, rho_ref, label='Analytic', linestyle='--', color='green', linewidth=1.5)
axs[0].plot(x[:, 0], rho[:, 70], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
axs[0].set_xlim([0., 1.])
axs[0].grid(alpha=0.3)
axs[0].legend(fontsize=10)

# Velocity
axs[1].set_title("U-velocity", fontsize=18)
axs[1].plot(X, U, label='Roe', linestyle='--', color='blue', linewidth=1.5)
axs[1].plot(X, u_ref, label='Analytic', linestyle='--', color='green', linewidth=1.5)
axs[1].plot(x[:, 0], u[:, 70], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
axs[1].set_xlim([0., 1.])
axs[1].grid(alpha=0.3)

# Pressure
axs[2].set_title("Pressure", fontsize=18)
axs[2].plot(X, P, label='Roe', linestyle='--', color='blue', linewidth=1.5)
axs[2].plot(X, P_ref, label='Analytic', linestyle='--', color='green', linewidth=1.5)
axs[2].plot(x[:, 0], p[:, 70], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
axs[2].set_xlim([0., 1.])
axs[2].grid(alpha=0.3)

# Apply global legend and save
fig.supxlabel('X (Position)', fontsize=15)
plt.savefig('results.png', dpi=300)


#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################



fig,ax = plt.subplots(figsize=(11, 8))
ax.plot(X, RHO, label='Roe', marker='o', markevery=2, markersize=4, color='blue', linewidth=1.5)
ax.plot(X, rho_ref, label='Analytic', linestyle='--', color='green', linewidth=1.5)
ax.plot(x[:, 0], rho[:, 70], label='PINN', marker='o',markevery=2, markersize=4, color='red', linewidth=1.5)


X11 = X[20:36] #roe
Y11 = RHO[20:36] #roe

X21 = x[10:18,70]
Y21 = rho[10:18,70]

X31 = X[20:36]
Y31 = rho_ref[20:36]

zm1 = ax.inset_axes([0.1,0.3, 0.25,0.25])
zm1.plot(X11,Y11,marker='o', markersize=4,markevery=3, color='blue', linewidth=1.5)
zm1.plot(X21, Y21, marker='o', markersize=4, markevery=1,color='red', linewidth=1.5)
zm1.plot(X31, Y31,linestyle='--', color='green', linewidth=1.5)
zm1.set_ylim([0.41,0.45])
zm1.set_xlim([0.1,0.17])
zm1.tick_params(axis='x', labelsize=12)
zm1.tick_params(axis='y', labelsize=12)
zm1.set_title('(i)', fontsize=10)
ax.indicate_inset_zoom(zm1,edgecolor='magenta', lw=2)


# plt.xlim([0., 1.])
# plt.ylim([-.05, 1.05])
ax.grid(alpha=0.3)
ax.legend(fontsize=16)
ax.set_xlabel('X')
ax.set_ylabel('Density')
plt.savefig('density01.png', dpi =200)
plt.close()

#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################



fig,ax = plt.subplots(figsize=(11, 8))
ax.plot(X, P, label='Roe', marker='o', markevery=2, markersize=4, color='blue', linewidth=1.5)
ax.plot(X, P_ref, label='Analytic', linestyle='--', color='green', linewidth=1.5)
ax.plot(x[:, 0], p[:, 70], label='PINN', marker='o',markevery=2, markersize=4, color='red', linewidth=1.5)


# X11 = X[20:36] #roe
# Y11 = RHO[20:36] #roe

# X21 = x[10:18,70]
# Y21 = rho[10:18,70]

# X31 = X[20:36]
# Y31 = rho_ref[20:36]

# zm1 = ax.inset_axes([0.1,0.3, 0.25,0.25])
# zm1.plot(X11,Y11,marker='o', markersize=4,markevery=3, color='blue', linewidth=1.5)
# zm1.plot(X21, Y21, marker='o', markersize=4, markevery=1,color='red', linewidth=1.5)
# zm1.plot(X31, Y31,linestyle='--', color='green', linewidth=1.5)
# zm1.set_ylim([0.41,0.45])
# zm1.set_xlim([0.1,0.17])
# zm1.tick_params(axis='x', labelsize=12)
# zm1.tick_params(axis='y', labelsize=12)
# zm1.set_title('(i)', fontsize=10)
# ax.indicate_inset_zoom(zm1,edgecolor='magenta', lw=2)


# plt.xlim([0., 1.])
# plt.ylim([-.05, 1.05])
ax.grid(alpha=0.3)
ax.legend(fontsize=16)
ax.set_xlabel('X')
ax.set_ylabel('Pressure')
plt.savefig('pressure01.png', dpi =200)
plt.close()

#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################



fig,ax = plt.subplots(figsize=(11, 8))
ax.plot(X, U, label='Roe', marker='o', markevery=2, markersize=4, color='blue', linewidth=1.5)
ax.plot(X, u_ref, label='Analytic', linestyle='--', color='green', linewidth=1.5)
ax.plot(x[:, 0], u[:, 70], label='PINN', marker='o',markevery=2, markersize=4, color='red', linewidth=1.5)


# X11 = X[20:36] #roe
# Y11 = RHO[20:36] #roe

# X21 = x[10:18,70]
# Y21 = rho[10:18,70]

# X31 = X[20:36]
# Y31 = rho_ref[20:36]

# zm1 = ax.inset_axes([0.1,0.3, 0.25,0.25])
# zm1.plot(X11,Y11,marker='o', markersize=4,markevery=3, color='blue', linewidth=1.5)
# zm1.plot(X21, Y21, marker='o', markersize=4, markevery=1,color='red', linewidth=1.5)
# zm1.plot(X31, Y31,linestyle='--', color='green', linewidth=1.5)
# zm1.set_ylim([0.41,0.45])
# zm1.set_xlim([0.1,0.17])
# zm1.tick_params(axis='x', labelsize=12)
# zm1.tick_params(axis='y', labelsize=12)
# zm1.set_title('(i)', fontsize=10)
# ax.indicate_inset_zoom(zm1,edgecolor='magenta', lw=2)


# plt.xlim([0., 1.])
# plt.ylim([-.05, 1.05])
ax.grid(alpha=0.3)
ax.legend(fontsize=16)
ax.set_xlabel('X')
ax.set_ylabel('Velocity')
plt.savefig('velocity01.png', dpi =200)
plt.close()

print(X.shape)
