import numpy as np
import matplotlib.pyplot as plt

filepath = './predict/predictions1.npy'
data = np.load(filepath)

array = np.array([[0,0],[0.5,0],[1.5, -np.sin(np.pi/18)], [0, -np.sin(np.pi/18)]])

print(data.shape)
#print(data)
Xmin = np.min(data[:,0])
Xmax = np.max(data[:,0])


Ymin = np.min(data[:,1])
Ymax = np.max(data[:,1])

x = data[:,0].reshape(100,200) 
y = data[:,1].reshape(100,200)
rho = data[:,2].reshape(100,200)
u = data[:,4].reshape(100,200) 
v = data[:,5].reshape(100,200) 
p = data[:,3].reshape(100,200) 

speed = np.sqrt(u**2 + v**2)

mach = speed/np.sqrt(1.4*p/rho)
p_t_inlet = (1 + 0.2*mach[0,10]**2)**3.5 * p[0,10]
Pressure_loss = (1 + 0.2*mach**2)**3.5 * p - p_t_inlet

print(p.min())
print(rho.min())
print(mach.max())

# plt.figure()
# contour = plt.contourf(x, y, u, cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='U-velocity')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# plt.title('U-velocity contour')
# plt.savefig('contour1.png', dpi =100)
# plt.show()

# plt.figure()
# contour = plt.contourf(x, y, v, cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='V-velocity')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# plt.title('V-velocity contour')
# plt.savefig('contour2.png', dpi =100)
# plt.show()

# plt.figure()
# contour = plt.contourf(x, y, p, cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='Pressure')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# plt.title('Pressure contour')
# plt.savefig('contour3.png', dpi =100)
# plt.show()

# plt.figure()
# contour = plt.contourf(x, y, speed, cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='speed')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# plt.title('Speed contour')
# plt.savefig('contour4.png', dpi =100)
# plt.show()


# plt.figure()
# contour = plt.contourf(x, y, rho, cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='Density')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# plt.title('Density contour')
# plt.savefig('contour5.png', dpi =100)
# plt.show()

# plt.figure()
# contour = plt.contourf(x, y, mach, cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='Mach')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# plt.title('Mach contour')
# plt.savefig('contour6.png', dpi =100)
# plt.show()


# General plot styling
plt.rcParams['font.family'] = 'serif'  # Set to desired font, e.g., 'serif', 'Times New Roman'
plt.rcParams['font.size'] = 22         # Set global font size
plt.rcParams['axes.labelsize'] = 22    # Font size for x and y axis labels
plt.rcParams['xtick.labelsize'] = 22   # Font size for x ticks
plt.rcParams['ytick.labelsize'] = 22   # Font size for y ticks
plt.rcParams['legend.fontsize'] = 22   # Font size for legend
plt.rcParams['axes.titleweight'] = 'bold'

# List of fields to plot
fields = {
    'U-velocity': u,
    'V-velocity': v,
    'Pressure': p,
    'Speed': speed,
    'Density': rho,
    'Mach': mach,
    'Pressure_loss': Pressure_loss
}
filenames = ['contour1.png', 'contour2.png', 'contour3.png', 'contour4.png', 'contour5.png', 'contour6.png', 'contour7.png']
labels = ['U-velocity', 'V-velocity', 'Pressure', 'Speed', 'Density', 'Mach', 'Pressure loss']

for field_name, data, filename, label in zip(fields.keys(), fields.values(), filenames, labels):
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x, y, data, cmap='rainbow', levels=100, extend='both')
    cbar = plt.colorbar(contour, pad=0.01)
    cbar.set_label(label)
    # plt.fill(cover[:, 0], cover[:, 1], 'white')
    # plt.grid(alpha=0.3)
    plt.xlim([Xmin, Xmax])
    plt.ylim([Ymin, Ymax])
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.title(f'{field_name} Contour', pad=15)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
