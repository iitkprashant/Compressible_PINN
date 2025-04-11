import numpy as np
import matplotlib.pyplot as plt

filepath = './predict/predictions.npy'
data = np.load(filepath)



print(data.shape)
#print(data)
Xmin = np.min(data[:,0])
Xmax = np.max(data[:,0])


Ymin = np.min(data[:,1])
Ymax = np.max(data[:,1])

x = data[:,0].reshape(101,101, 41) 
y = data[:,1].reshape(101,101, 41)
t = data[:,2].reshape(101,101, 41)
rho = data[:,3].reshape(101,101, 41)
u = data[:,5].reshape(101,101, 41) 
v = data[:,6].reshape(101,101, 41) 
p = data[:,4].reshape(101,101, 41) 

speed = np.sqrt(u**2 + v**2)


# plt.figure()
# contour = plt.contourf(x[:,:,-1], y[:,:,-1], u[:,:,-1], cmap='rainbow', levels=50, extend='both')
# plt.colorbar(contour, label='U-velocity')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# # plt.title('U-velocity at -10.0 Degree at M=2')
# plt.savefig('contour1.png', dpi =100)
# plt.show()

# plt.figure()
# contour = plt.contourf(x[:,:,-1], y[:,:,-1], v[:,:,-1], cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='V-velocity')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# # plt.title('V-velocity at -10.0 Degree at M=2')
# plt.savefig('contour2.png', dpi =100)
# plt.show()

# plt.figure()
# contour = plt.contourf(x[:,:,-1], y[:,:,-1], p[:,:,-1], cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='Pressure')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# # plt.title('Pressure at -10.0 Degree at M=2')
# plt.savefig('contour3.png', dpi =100)
# plt.show()

# plt.figure()
# contour = plt.contourf(x[:,:,-1], y[:,:,-1], speed[:,:,-1], cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='speed')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# # plt.title('Speed at -10.0 Degree at M=2')
# plt.savefig('contour4.png', dpi =100)
# plt.show()


# plt.figure()
# contour = plt.contourf(x[:,:,-1], y[:,:,-1], rho[:,:,-1], cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='Density')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# # plt.title('Density at -10.0 Degree at M=2')
# plt.savefig('contour5.png', dpi =100)
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
    'U-velocity': u[:,:,-1],
    'V-velocity': v[:,:,-1],
    'Pressure': p[:,:,-1],
    'Speed': speed[:,:,-1],
    'Density': rho[:,:,-1],
}
filenames = ['contour1.png', 'contour2.png', 'contour3.png', 'contour4.png', 'contour5.png']
labels = ['U-velocity', 'V-velocity', 'Pressure', 'Speed', 'Density', 'Mach']

for field_name, data, filename, label in zip(fields.keys(), fields.values(), filenames, labels):
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x[:,:,-1], y[:,:,-1], data, cmap='rainbow', levels=100, extend='both')
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