import numpy as np
import matplotlib.pyplot as plt

filepath = './predict/predictions1.npy'
data = np.load(filepath)
cover = np.array([[0.5,0.0],[1.5,0.0],[1.5, np.sin(np.pi/18)]])


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




print(p.max())
print(rho.max())
print(mach.min())
# plt.figure()
# contour = plt.contourf(x, y, u, cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='U-velocity')
# plt.fill(cover[:,0], cover[:,1],'w')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# plt.title('U-velocity contour')
# plt.savefig('contour1.png', dpi =100)
# plt.show()

# plt.figure()
# contour = plt.contourf(x, y, v, cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='V-velocity')
# plt.fill(cover[:,0], cover[:,1],'w')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# plt.title('V-velocity contour')
# plt.savefig('contour2.png', dpi =100)
# plt.show()

# plt.figure()
# contour = plt.contourf(x, y, p, cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='Pressure')
# plt.fill(cover[:,0], cover[:,1],'w')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# plt.title('Pressure contour')
# plt.savefig('contour3.png', dpi =100)
# plt.show()

# plt.figure()
# contour = plt.contourf(x, y, speed, cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='speed')
# plt.fill(cover[:,0], cover[:,1],'w')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# plt.title('Speed contour')
# plt.savefig('contour4.png', dpi =100)
# plt.show()


# plt.figure()
# contour = plt.contourf(x, y, rho, cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='density')
# plt.fill(cover[:,0], cover[:,1],'w')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# plt.title('Density contour')
# plt.savefig('contour5.png', dpi =100)
# plt.show()

# plt.figure()
# contour = plt.contourf(x, y, mach, cmap='rainbow', levels=100, extend='both')
# plt.colorbar(contour, label='Mach')
# plt.fill(cover[:,0], cover[:,1],'w')
# plt.xlim([Xmin, Xmax])
# plt.ylim([Ymin, Ymax])
# plt.title('Mach contour')
# plt.savefig('contour6.png', dpi =100)
# plt.show()


# General plot styling
plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# List of fields to plot
fields = {
    'U-velocity': u,
    'V-velocity': v,
    'Pressure': p,
    'Speed': speed,
    'Density': rho,
    'Mach': mach
}
filenames = ['contour1.png', 'contour2.png', 'contour3.png', 'contour4.png', 'contour5.png', 'contour6.png']
labels = ['U-velocity', 'V-velocity', 'Pressure', 'Speed', 'Density', 'Mach']

for field_name, data, filename, label in zip(fields.keys(), fields.values(), filenames, labels):
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x, y, data, cmap='rainbow', levels=100, extend='both')
    cbar = plt.colorbar(contour, pad=0.01)
    cbar.set_label(label)
    plt.fill(cover[:, 0], cover[:, 1], 'white')
    # plt.grid(alpha=0.3)
    plt.xlim([Xmin, Xmax])
    plt.ylim([Ymin, Ymax])
    plt.xlabel('$x$-coordinate')
    plt.ylabel('$y$-coordinate')
    plt.title(f'{field_name} Contour', pad=15)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()