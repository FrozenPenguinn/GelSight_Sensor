import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import atan2, sqrt

# import gradients
grad = np.load( "grad.npy" )
gx = grad[:,:,0]
gy = grad[:,:,1]
ydim, xdim, dim = grad.shape

print(xdim)
print(ydim)
'''
# get magnitude
magnitude = np.zeros([ydim, xdim], dtype = float)
orientation = np.zeros([ydim, xdim], dtype = float)
for x in range(0, xdim):
    for y in range(0, ydim):
        magnitude[y, x] = sqrt(pow(gx[y, x],2) + pow(gy[y, x],2))
        orientation[y, x] = atan2(gy[y, x], gx[y, x])
'''
x_upper = 145
x_lower = 45
y_upper = 145
y_lower = 45

x,y = np.meshgrid(np.linspace(x_lower,x_upper,x_upper-x_lower),np.linspace(y_lower,y_upper,y_upper-y_lower))

x_dir = gx[x_lower:x_upper,y_lower:y_upper]
y_dir = gy[x_lower:x_upper,y_lower:y_upper]

u = x_dir/np.sqrt(x_dir**2 + y_dir**2)
v = y_dir/np.sqrt(x_dir**2 + y_dir**2)

matplotlib.rc('figure', figsize=(8, 8))
plt.quiver(x,y,u,v)
plt.show()

'''
# visualize orientation and magnitude
x, y = np.meshgrid(x, y)
# plot vector field
u = gx/np.sqrt(gx**2 + gy**2)
v = gy/np.sqrt(gx**2 + gy**2)
plt.quiver(x,y,u,v)
# graph
#matplotlib.rc('figure', figsize=(10, 5))
#fig, axes = plt.subplots(1, 2)
# difference img
#im1 = axes[0].imshow(magnitude, origin='lower', extent=[0, xdim, 0, ydim])
# extracted gradients
#im2 = axes[1].imshow(orientation, origin='lower', extent=[0, xdim, 0, ydim])
# set labels
#plt.setp(axes[:], xlabel='x')
#plt.setp(axes[:], ylabel='y')
#axes[0].set_title("magnitude")
#axes[1].set_title("orientation")
#fig.colorbar(im1, ax=axes[0])
#fig.colorbar(im2, ax=axes[1])
plt.show()
'''
