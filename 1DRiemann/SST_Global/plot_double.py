import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import newton

# 0.2 sec Roe
data2 = np.loadtxt('sol.dat')

# Extract columns
x2 = data2[:, 0]        # x-coordinates
density2 = data2[:, 1]  # Density (rho)
velocity2 = data2[:, 2] # Velocity (u)
pressure2 = data2[:, 3] # Pressure (p)

#0.1 sec Roe
data3 = np.loadtxt('sol1.dat')

# Extract columns
x3 = data3[:, 0]        # x-coordinates
density3 = data3[:, 1]  # Density (rho)
velocity3 = data3[:, 2] # Velocity (u)
pressure3 = data3[:, 3] # Pressure (p)


# Function to find the roots of!
def f(P, pL, pR, cL, cR, gg):
    a = (gg-1)*(cR/cL)*(P-1) 
    b = np.sqrt( 2*gg*(2*gg + (gg+1)*(P-1) ) )
    return P - pL/pR*( 1 - a/b )**(2.*gg/(gg-1.))

# Analtyic Sol to Sod Shock
def SodShockAnalytic(rL, uL, pL, rR, uR, pR, xs, x0, T, gg):
    # rL, uL, pL, rR, uR, pR : Initial conditions of the Reimann problem 
    # xs: position array (e.g. xs = [0,dx,2*dx,...,(Nx-1)*dx])
    # x0: THIS IS AN INDEX! the array index where the interface sits.
    # T: the desired solution time
    # gg: adiabatic constant 1.4=7/5 for a 3D diatomic gas
    dx = xs[1];
    Nx = len(xs)
    v_analytic = np.zeros((3,Nx),dtype='float64')

    # compute speed of sound
    cL = np.sqrt(gg*pL/rL); 
    cR = np.sqrt(gg*pR/rR);
    # compute P
    P = newton(f, 0.5, args=(pL, pR, cL, cR, gg), tol=1e-12);

    # compute region positions right to left
    # region R
    c_shock = uR + cR*np.sqrt( (gg-1+P*(gg+1)) / (2*gg) )
    x_shock = x0 + int(np.floor(c_shock*T/dx))
    v_analytic[0,x_shock-1:] = rR
    v_analytic[1,x_shock-1:] = uR
    v_analytic[2,x_shock-1:] = pR
    
    # region 2
    alpha = (gg+1)/(gg-1)
    c_contact = uL + 2*cL/(gg-1)*( 1-(P*pR/pL)**((gg-1.)/2/gg) )
    x_contact = x0 + int(np.floor(c_contact*T/dx))
    v_analytic[0,x_contact:x_shock-1] = (1 + alpha*P)/(alpha+P)*rR
    v_analytic[1,x_contact:x_shock-1] = c_contact
    v_analytic[2,x_contact:x_shock-1] = P*pR
    
    # region 3
    r3 = rL*(P*pR/pL)**(1/gg);
    p3 = P*pR;
    c_fanright = c_contact - np.sqrt(gg*p3/r3)
    x_fanright = x0 + int(np.ceil(c_fanright*T/dx))
    v_analytic[0,x_fanright:x_contact] = r3;
    v_analytic[1,x_fanright:x_contact] = c_contact;
    v_analytic[2,x_fanright:x_contact] = P*pR;
    
    # region 4
    c_fanleft = -cL
    x_fanleft = x0 + int(np.ceil(c_fanleft*T/dx))
    u4 = 2 / (gg+1) * (cL + (xs[x_fanleft:x_fanright]-xs[x0])/T )
    v_analytic[0,x_fanleft:x_fanright] = rL*(1 - (gg-1)/2.*u4/cL)**(2/(gg-1));
    v_analytic[1,x_fanleft:x_fanright] = u4;
    v_analytic[2,x_fanleft:x_fanright] = pL*(1 - (gg-1)/2.*u4/cL)**(2*gg/(gg-1));

    # region L
    v_analytic[0,:x_fanleft] = rL
    v_analytic[1,:x_fanleft] = uL
    v_analytic[2,:x_fanleft] = pL

    return v_analytic

# Physics
gg=1.4  # gamma = C_v / C_p = 7/5 for ideal gas
rL, uL, pL =  1.0,  0.0, 1; 
rR, uR, pR = 0.125, 0.0, .1;

# Set Disretization
Nx = 100
X = 1.
dx = X/(Nx-1)
xs = np.linspace(0,X,Nx)
x0 = Nx//2
T = 0.1

analytic = SodShockAnalytic(rL, uL, pL, rR, uR, pR, xs, x0, T, gg)
analytic2 = SodShockAnalytic(rL, uL, pL, rR, uR, pR, xs, x0, 0.2, gg)

def read_csv_data(filename):
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # Skip the header row
        data = np.array([list(map(float, row)) for row in csv_reader])
    return data, header


filepath = './predict/predictions_lbfgs21.npy'
data = np.load(filepath)

AVfilepath = './predict/predictions2_lbfgs21.npy'
AVdata = np.load(AVfilepath)

filename = "time_step_100.csv" 

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

x = data[:,0].reshape(101,101) 
y = data[:,1].reshape(101,101)
rho = data[:,2].reshape(101,101)
u = data[:,4].reshape(101,101) 
p = data[:,3].reshape(101,101) 

mu = AVdata[:,2].reshape(101,101)**2



filename_analytic = 'sst_analytic.csv'
data4, header2 = read_csv_data(filename_analytic)

x_analytic = data4[:,0] + 0.5
rho_analytic = data4[:,1]
p_analytic = data4[:,2]
u_analytic = data4[:,3]
# print(data4.shape)

filename_analytic2 = 'sst_analytic2.csv'
data5, header3 = read_csv_data(filename_analytic2)

x_analytic2 = data5[:,0] + 0.5
rho_analytic2 = data5[:,1]
p_analytic2 = data5[:,2]
u_analytic2 = data5[:,3]
# print(data4.shape)



plt.rcParams['font.family'] = 'serif'  # Set to desired font, e.g., 'serif', 'Times New Roman'
plt.rcParams['font.size'] = 18         # Set global font size
plt.rcParams['axes.labelsize'] = 18    # Font size for x and y axis labels
plt.rcParams['xtick.labelsize'] = 18   # Font size for x ticks
plt.rcParams['ytick.labelsize'] = 18   # Font size for y ticks
plt.rcParams['legend.fontsize'] = 18   # Font size for legend
plt.rcParams['axes.titleweight'] = 'bold'


# Set a global style for consistency
# plt.style.use('seaborn')

# Contour Plots
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

plt.figure(figsize=(8, 6))
contour = plt.contourf(x, y, mu, cmap='rainbow', levels=100, extend='both') 
cbar = plt.colorbar(contour, label='Viscosity')
cbar.ax.tick_params(labelsize=18)
plt.xlim([Xmin, Xmax])
plt.ylim([Ymin, Ymax])
plt.xlabel('X (Position)')
plt.ylabel('T (Time)')
# plt.title('Viscosity Contour', fontsize=14, weight='bold')
plt.savefig('contour4.png', dpi=200, bbox_inches='tight')
plt.show()

plt.rcParams['font.family'] = 'serif'  # Set to desired font, e.g., 'serif', 'Times New Roman'
plt.rcParams['font.size'] = 20         # Set global font size
plt.rcParams['axes.labelsize'] = 20    # Font size for x and y axis labels
plt.rcParams['xtick.labelsize'] = 20   # Font size for x ticks
plt.rcParams['ytick.labelsize'] = 20   # Font size for y ticks
plt.rcParams['legend.fontsize'] = 20   # Font size for legend
plt.rcParams['axes.titleweight'] = 'bold'



# Line Plots
fig, axs = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

# Density
axs[0].set_title("(a)", fontsize=18)
axs[0].plot(x3, density3, label='Roe', marker='o', markersize=2, color='blue', linewidth=1.5)
axs[0].plot(x_analytic, rho_analytic, label='Analytic', linestyle='--', color='green', linewidth=1.5)
axs[0].plot(x[:, 0], rho[:, 51], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
axs[0].set_xlim([0., 1.])
axs[0].set_ylim([-.05, 1.05])
axs[0].grid(alpha=0.3)
axs[0].legend(fontsize=10)

# Velocity
axs[1].set_title("(b)", fontsize=18)
axs[1].plot(x3, velocity3, label='Roe', marker='o', markersize=2, color='blue', linewidth=1.5)
axs[1].plot(x_analytic, u_analytic, label='Analytic', linestyle='--', color='green', linewidth=1.5)
axs[1].plot(x[:, 0], u[:, 51], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
axs[1].set_xlim([0., 1.])
axs[1].set_ylim([-.05, 1.05])
axs[1].grid(alpha=0.3)

# Pressure
axs[2].set_title("(c)", fontsize=18)
axs[2].plot(x3, pressure3, label='Roe', marker='o', markersize=2, color='blue', linewidth=1.5)
axs[2].plot(x_analytic, p_analytic, label='Analytic', linestyle='--', color='green', linewidth=1.5)
axs[2].plot(x[:, 0], p[:, 51], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
axs[2].set_xlim([0., 1.])
axs[2].set_ylim([-.05, 1.05])
axs[2].grid(alpha=0.3)

# Apply global legend and save
fig.supxlabel('X (Position)', fontsize=15)
plt.savefig('results.png', dpi=300)

############################

# Density Plot
plt.figure(figsize=(8, 6))
# plt.title("Density", fontsize=18)
plt.plot(x3, density3, label='Roe', marker='o', markersize=2, color='blue', linewidth=1.5)
plt.plot(x_analytic, rho_analytic, label='Analytic', linestyle='--', color='green', linewidth=1.5)
plt.plot(x[:, 0], rho[:, 51], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
plt.xlim([0., 1.])
plt.ylim([-.05, 1.05])
plt.xlabel("X (Position)")
plt.ylabel("Density")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("density01p.png", dpi=300)
plt.show()

# Velocity Plot
plt.figure(figsize=(8, 6))
# plt.title("Velocity", fontsize=18)
plt.plot(x3, velocity3, label='Roe', marker='o', markersize=2, color='blue', linewidth=1.5)
plt.plot(x_analytic, u_analytic, label='Analytic', linestyle='--', color='green', linewidth=1.5)
plt.plot(x[:, 0], u[:, 51], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
plt.xlim([0., 1.])
plt.ylim([-.05, 1.05])
plt.xlabel("X (Position)")
plt.ylabel("Velocity")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("velocity01p.png", dpi=300)
plt.show()

# Pressure Plot
plt.figure(figsize=(8, 6))
# plt.title("Pressure", fontsize=18)
plt.plot(x3, pressure3, label='Roe', marker='o', markersize=2, color='blue', linewidth=1.5)
plt.plot(x_analytic, p_analytic, label='Analytic', linestyle='--', color='green', linewidth=1.5)
plt.plot(x[:, 0], p[:, 51], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
plt.xlim([0., 1.])
plt.ylim([-.05, 1.05])
plt.xlabel("X (Position)")
plt.ylabel("Pressure")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("pressure01p.png", dpi=300)
plt.show()
#############






# Second Line Plot
fig, axs = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

# Density
axs[0].set_title("(a)", fontsize=18)
axs[0].plot(x2, density2, label='Roe', marker='o', markersize=2, color='blue', linewidth=1.5)
axs[0].plot(x_analytic2, rho_analytic2, label='Analytic', linestyle='--', color='green', linewidth=1.5)
axs[0].plot(x[:, 0], rho[:, -1], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
axs[0].set_xlim([0., 1.])
axs[0].set_ylim([-.05, 1.05])
axs[0].grid(alpha=0.3)
axs[0].legend(fontsize=10)

# Velocity
axs[1].set_title("(b)", fontsize=18)
axs[1].plot(x2, velocity2, label='Roe', marker='o', markersize=2, color='blue', linewidth=1.5)
axs[1].plot(x_analytic2, u_analytic2, label='Analytic', linestyle='--', color='green', linewidth=1.5)
axs[1].plot(x[:, 0], u[:, -1], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
axs[1].set_xlim([0., 1.])
axs[1].set_ylim([-.05, 1.05])
axs[1].grid(alpha=0.3)

# Pressure
axs[2].set_title("(c)", fontsize=18)
axs[2].plot(x2, pressure2, label='Roe', marker='o', markersize=2, color='blue', linewidth=1.5)
axs[2].plot(x_analytic2,p_analytic2, label='Analytic', linestyle='--', color='green', linewidth=1.5)
axs[2].plot(x[:, 0], p[:, -1], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
axs[2].set_xlim([0., 1.])
axs[2].set_ylim([-.05, 1.05])
axs[2].grid(alpha=0.3)

# Apply global legend and save
fig.supxlabel('X (Position)', fontsize=15)
plt.savefig('results2.png', dpi=300)


# Density Plot
plt.figure(figsize=(8,6))
# plt.title("Density", fontsize=18)
plt.plot(x2, density2, label='Roe', marker='o', markersize=2, color='blue', linewidth=1.5)
plt.plot(x_analytic2, rho_analytic2, label='Analytic', linestyle='--', color='green', linewidth=1.5)
plt.plot(x[:, 0], rho[:, -1], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
plt.xlim([0., 1.])
plt.ylim([-.05, 1.05])
plt.xlabel('X (Position)')
plt.ylabel('Density')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('density02p.png', dpi=300)

# Velocity Plot
plt.figure(figsize=(8, 6))
# plt.title("Velocity", fontsize=18)
plt.plot(x2, velocity2, label='Roe', marker='o', markersize=2, color='blue', linewidth=1.5)
plt.plot(x_analytic2, u_analytic2, label='Analytic', linestyle='--', color='green', linewidth=1.5)
plt.plot(x[:, 0], u[:, -1], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
plt.xlim([0., 1.])
plt.ylim([-.05, 1.05])
plt.xlabel('X (Position)')
plt.ylabel('Velocity')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('velocity02p.png', dpi=300)

# Pressure Plot
plt.figure(figsize=(8, 6))
# plt.title("Pressure", fontsize=18)
plt.plot(x2, pressure2, label='Roe', marker='o', markersize=2, color='blue', linewidth=1.5)
plt.plot(x_analytic2, p_analytic2, label='Analytic', linestyle='--', color='green', linewidth=1.5)
plt.plot(x[:, 0], p[:, -1], label='PINN', marker='o', markersize=2, color='red', linewidth=1.5)
plt.xlim([0., 1.])
plt.ylim([-.05, 1.05])
plt.xlabel('X (Position)')
plt.ylabel('Pressure')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('pressure02p.png', dpi=300)


#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################



fig,ax = plt.subplots(figsize=(11, 8))
ax.plot(x3, density3, label='Roe', marker='o', markevery=6, markersize=4, color='blue', linewidth=1.5)
ax.plot(x_analytic, rho_analytic, label='Analytic', linestyle='--', color='green', linewidth=1.5)
ax.plot(x[:, 0], rho[:, 51], label='PINN', marker='o',markevery=2, markersize=4, color='red', linewidth=1.5)


X11 = x3[165:192] #roe
Y11 = density3[165:192] #roe

X21 = x[55:64,51]
Y21 = rho[55:64,51]

X31 = x_analytic[550:640]
Y31 = rho_analytic[550:640]

zm1 = ax.inset_axes([0.1,0.075, 0.25,0.25])
zm1.plot(X11,Y11,marker='o', markersize=4,markevery=3, color='blue', linewidth=1.5)
zm1.plot(X21, Y21, marker='o', markersize=4, markevery=1,color='red', linewidth=1.5)
zm1.plot(X31, Y31,linestyle='--', color='green', linewidth=1.5)
zm1.set_ylim([0.23,0.45])
zm1.set_xlim([0.55,0.63])
zm1.tick_params(axis='x', labelsize=12)
zm1.tick_params(axis='y', labelsize=12)
zm1.set_title('(ii)', fontsize=10)
ax.indicate_inset_zoom(zm1,edgecolor='magenta', lw=2)


X12 = x3[192:210] #roe
Y12 = density3[192:210] #roe

X22 = x[64:70,51]
Y22 = rho[64:70,51]

X32 = x_analytic[640:700]
Y32 = rho_analytic[640:700]

zm2 = ax.inset_axes([0.68,0.43, 0.25,0.25])
zm2.plot(X12,Y12,marker='o', markersize=4,markevery=3, color='blue', linewidth=1.5)
zm2.plot(X22, Y22, marker='o', markersize=4, markevery=1,color='red', linewidth=1.5)
zm2.plot(X32, Y32,linestyle='--', color='green', linewidth=1.5)
zm2.set_ylim([0.11,0.28])
zm2.set_xlim([0.65,0.70])
zm2.tick_params(axis='x', labelsize=12, labeltop=True, labelbottom=False)
zm2.tick_params(axis='y', labelsize=12, labelright=True, labelleft=False)
zm2.set_title('(iii)', fontsize=10)
ax.indicate_inset_zoom(zm2,edgecolor='magenta', lw=2)


X13 = x3[99:126] #roe
Y13 = density3[99:126] #roe

X23 = x[33:42,51]
Y23 = rho[33:42,51]

X33 = x_analytic[330:420]
Y33 = rho_analytic[330:420]

zm3 = ax.inset_axes([0.07,0.5, 0.25,0.25])
zm3.plot(X13,Y13,marker='o', markersize=4,markevery=3, color='blue', linewidth=1.5)
zm3.plot(X23, Y23, marker='o', markersize=4, markevery=1,color='red', linewidth=1.5)
zm3.plot(X33, Y33,linestyle='--', color='green', linewidth=1.5)
zm3.set_ylim([0.75,1.02])
zm3.set_xlim([0.33,0.42])
zm3.tick_params(axis='x', labelsize=12)
zm3.tick_params(axis='y', labelsize=12)
zm3.set_title('(i)', fontsize=10)
ax.indicate_inset_zoom(zm3,edgecolor='magenta', lw=2)


plt.xlim([0., 1.])
plt.ylim([-.05, 1.05])
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
ax.plot(x3, pressure3, label='Roe', marker='o', markevery=6, markersize=4, color='blue', linewidth=1.5)
ax.plot(xs, analytic[2].T, label='Analytic', linestyle='--', color='green', linewidth=1.5)
ax.plot(x[:, 0], p[:, 51], label='PINN', marker='o',markevery=2, markersize=4, color='red', linewidth=1.5)


# X11 = x3[165:192] #roe
# Y11 = density3[165:192] #roe

# X21 = x[55:64,51]
# Y21 = rho[55:64,51]

# X31 = xs[55:64]
# Y31 = analytic[0][55:64].T

# zm1 = ax.inset_axes([0.1,0.075, 0.25,0.25])
# zm1.plot(X11,Y11,marker='o', markersize=4,markevery=3, color='blue', linewidth=1.5)
# zm1.plot(X21, Y21, marker='o', markersize=4, markevery=1,color='red', linewidth=1.5)
# zm1.plot(X31, Y31,linestyle='--', color='green', linewidth=1.5)
# zm1.set_ylim([0.23,0.45])
# zm1.set_xlim([0.55,0.63])
# zm1.tick_params(axis='x', labelsize=12)
# zm1.tick_params(axis='y', labelsize=12)
# ax.indicate_inset_zoom(zm1,edgecolor='magenta', lw=2)


X12 = x3[192:210] #roe
Y12 = pressure3[192:210] #roe

X22 = x[64:70,51]
Y22 = p[64:70,51]

X32 = xs[64:70]
Y32 = analytic[2][64:70].T

zm2 = ax.inset_axes([0.68,0.43, 0.25,0.25])
zm2.plot(X12,Y12,marker='o', markersize=4,markevery=3, color='blue', linewidth=1.5)
zm2.plot(X22, Y22, marker='o', markersize=4, markevery=1,color='red', linewidth=1.5)
zm2.plot(X32, Y32,linestyle='--', color='green', linewidth=1.5)
zm2.set_ylim([0.08,0.33])
zm2.set_xlim([0.65,0.70])
zm2.tick_params(axis='x', labelsize=12, labeltop=True, labelbottom=False)
zm2.tick_params(axis='y', labelsize=12, labelright=True, labelleft=False)
zm2.set_title('(ii)', fontsize=10)
ax.indicate_inset_zoom(zm2,edgecolor='magenta', lw=2)


X13 = x3[99:126] #roe
Y13 = pressure3[99:126] #roe

X23 = x[33:42,51]
Y23 = p[33:42,51]

X33 = xs[33:42]
Y33 = analytic[2][33:42].T

zm3 = ax.inset_axes([0.07,0.5, 0.25,0.25])
zm3.plot(X13,Y13,marker='o', markersize=4,markevery=3, color='blue', linewidth=1.5)
zm3.plot(X23, Y23, marker='o', markersize=4, markevery=1,color='red', linewidth=1.5)
zm3.plot(X33, Y33,linestyle='--', color='green', linewidth=1.5)
zm3.set_ylim([0.75,1.02])
zm3.set_xlim([0.33,0.42])
zm3.tick_params(axis='x', labelsize=12)
zm3.tick_params(axis='y', labelsize=12)
zm3.set_title('(i)', fontsize=10)
ax.indicate_inset_zoom(zm3,edgecolor='magenta', lw=2)


plt.xlim([0., 1.])
plt.ylim([-.05, 1.05])
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
ax.plot(x3, velocity3, label='Roe', marker='o', markevery=6, markersize=4, color='blue', linewidth=1.5)
ax.plot(xs, analytic[1].T, label='Analytic', linestyle='--', color='green', linewidth=1.5)
ax.plot(x[:, 0], u[:, 51], label='PINN', marker='o',markevery=2, markersize=4, color='red', linewidth=1.5)


# X11 = x3[165:192] #roe
# Y11 = density3[165:192] #roe

# X21 = x[55:64,51]
# Y21 = rho[55:64,51]

# X31 = xs[55:64]
# Y31 = analytic[0][55:64].T

# zm1 = ax.inset_axes([0.1,0.075, 0.25,0.25])
# zm1.plot(X11,Y11,marker='o', markersize=4,markevery=3, color='blue', linewidth=1.5)
# zm1.plot(X21, Y21, marker='o', markersize=4, markevery=1,color='red', linewidth=1.5)
# zm1.plot(X31, Y31,linestyle='--', color='green', linewidth=1.5)
# zm1.set_ylim([0.23,0.45])
# zm1.set_xlim([0.55,0.63])
# zm1.tick_params(axis='x', labelsize=12)
# zm1.tick_params(axis='y', labelsize=12)
# ax.indicate_inset_zoom(zm1,edgecolor='magenta', lw=2)


# X12 = x3[192:210] #roe
# Y12 = density3[192:210] #roe

# X22 = x[64:70,51]
# Y22 = rho[64:70,51]

# X32 = xs[64:70]
# Y32 = analytic[0][64:70].T

# zm2 = ax.inset_axes([0.68,0.43, 0.25,0.25])
# zm2.plot(X12,Y12,marker='o', markersize=4,markevery=3, color='blue', linewidth=1.5)
# zm2.plot(X22, Y22, marker='o', markersize=4, markevery=1,color='red', linewidth=1.5)
# zm2.plot(X32, Y32,linestyle='--', color='green', linewidth=1.5)
# zm2.set_ylim([0.11,0.28])
# zm2.set_xlim([0.65,0.70])
# zm2.tick_params(axis='x', labelsize=12, labeltop=True, labelbottom=False)
# zm2.tick_params(axis='y', labelsize=12, labelright=True, labelleft=False)
# ax.indicate_inset_zoom(zm2,edgecolor='magenta', lw=2)


# X13 = x3[99:126] #roe
# Y13 = density3[99:126] #roe

# X23 = x[33:42,51]
# Y23 = rho[33:42,51]

# X33 = xs[33:42]
# Y33 = analytic[0][33:42].T

# zm3 = ax.inset_axes([0.07,0.5, 0.25,0.25])
# zm3.plot(X13,Y13,marker='o', markersize=4,markevery=3, color='blue', linewidth=1.5)
# zm3.plot(X23, Y23, marker='o', markersize=4, markevery=1,color='red', linewidth=1.5)
# zm3.plot(X33, Y33,linestyle='--', color='green', linewidth=1.5)
# zm3.set_ylim([0.75,1.02])
# zm3.set_xlim([0.33,0.42])
# zm3.tick_params(axis='x', labelsize=12)
# zm3.tick_params(axis='y', labelsize=12)
# ax.indicate_inset_zoom(zm3,edgecolor='magenta', lw=2)


plt.xlim([0., 1.])
plt.ylim([-.05, 1.05])
ax.grid(alpha=0.3)
ax.legend(fontsize=16)
ax.set_xlabel('X')
ax.set_ylabel('Velocity')
plt.savefig('velocity01.png', dpi =200)
plt.close()




# plt.figure()
# plt.plot(x_analytic+0.5,rho_analytic)
# plt.plot(xs, analytic2[0].T)
# plt.savefig('temp.png')
