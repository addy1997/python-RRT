#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

# if using a Jupyter notebook, include:
get_ipython().run_line_magic('matplotlib', 'inline')


fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121, projection='3d')
ax = fig.add_subplot(122, projection='3d')

x = np.arange(-9,10,0.01) 
y = np.arange(-9,10,0.01) 

y.size
print(y.size)

y_new = y.reshape(1,y.size)

X,Y = np.meshgrid(x,y)

Z = -X**4*np.sqrt(x**2+y**2) + Y**2*np.sqrt(x**2+y**2)

#projection on right
x1 = np.arange(-9,10,0.01) 
y1 = np.arange(-9,10,0.01) 

y1_axis = y1[100]*y1.size

y1_axis

y1.size
print(y1.size)

y1_new = y1.reshape(y1.size,1)
x1_new = x1.reshape(1,y1.size)

X1,Y1 = np.meshgrid(x1,y1_axis)
Z1 = -X1**4*np.sqrt(x1**2+y1_axis**2) + Y1**2*np.sqrt(x1**2+y1_axis**2)

# Plot a basic wireframe
ax.plot_wireframe(X1, Y1, Z1, rstride=1, cstride=1)
ax.set_title('row step size 10, column step size 10')

ax.plot_wireframe(X1, Y1, Z1, rstride=1, cstride=1)
ax.set_title('row step size 20, column step size 20')

#plt.show()

surf = ax1.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.8, cmap=cm.ocean)
cset = ax1.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.ocean)
cset = ax1.contourf(X, Y, Z, zdir='x', offset=5, cmap=cm.ocean)
cset = ax1.contourf(X, Y, Z, zdir='y', offset=-4, cmap=cm.ocean)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax1.set_xlabel('X')
ax1.set_xlim(-15, 35)
ax1.set_ylabel('Y')
ax1.set_ylim(-15, 35)
ax1.set_zlabel('Z')
ax1.set_zlim(np.min(Z), np.max(Z))
ax1.set_title('3D surface with 2D contour plot projections')

ax.set_xlabel('X1')
ax.set_xlim(-15,35)
ax.set_ylabel('Y1')
ax.set_xlim(-15,35)
ax.set_zlabel('Z1')
ax.set_zlim(np.min(Z1), np.max(Z1))

plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

# if using a Jupyter notebook, include:
get_ipython().run_line_magic('matplotlib', 'inline')


fig = plt.figure(figsize=(12,6))

ax1 = fig.add_subplot(121, projection='3d')
ax = fig.add_subplot(122, projection='3d')

x = np.arange(-10,-4,0.01) 
y = np.arange(-10,-4,0.01)

y.size
print(y.size)

y_new = y.reshape(1,y.size)

X,Y = np.meshgrid(x,y)

Z = - X**4*np.sqrt(x**2+y**2) + Y**2*np.sqrt(x**2+y**2)

x1 = np.arange(-10,-4,0.01) 
y1 = np.arange(-10,-4,0.01) 


y1_axis = (y1[0]+y[2])*y1.size

y1_axis

y1.size
print(y1.size)

y1_new = y1.reshape(y1.size,1)
x1_new = x1.reshape(1,y1.size)

X1,Y1 = np.meshgrid(x1,y1_axis)
Z1 = - X1**4*np.sqrt(x1**2+y1_axis**2) + Y1**2*np.sqrt(x1**2+y1_axis**2)

# Plot a basic wireframe
ax.plot_wireframe(X1, Y1, Z1, rstride=1, cstride=1)
ax.set_title('row step size 10, column step size 10')

ax.plot_wireframe(X1, Y1, Z1, rstride=1, cstride=1)
ax.set_title('row step size 20, column step size 20')

#plt.show()

surf = ax1.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.8, cmap=cm.ocean)
cset = ax1.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.ocean)
cset = ax1.contourf(X, Y, Z, zdir='x', offset=-5, cmap=cm.ocean)
cset = ax1.contourf(X, Y, Z, zdir='y', offset=-4, cmap=cm.ocean)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax1.set_xlabel('X')
ax1.set_xlim(-15, 5)
ax1.set_ylabel('Y')
ax1.set_ylim(-15, 5)
ax1.set_zlabel('Z')
ax1.set_zlim(np.min(Z), np.max(Z))
ax1.set_title('3D surface with 2D contour plot projections')

plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

# if using a Jupyter notebook, include:
get_ipython().run_line_magic('matplotlib', 'inline')


fig = plt.figure(figsize=(12,6))

ax1 = fig.add_subplot(121, projection='3d')
ax = fig.add_subplot(122, projection='3d')

x = np.arange(0,4,0.01) + np.arange(4,8,0.01)
y = np.arange(0,4,0.01) +np.arange(4,8,0.01) 

y.size
print(y.size)

y_new = y.reshape(1,y.size)

X,Y = np.meshgrid(x,y)

Z = - X**4*np.sqrt(x**2+y**2) + Y**2*np.sqrt(x**2+y**2)

x1 = np.arange(0,4,0.01) + np.arange(4,8,0.01) 
y1 = np.arange(0,4,0.01) + np.arange(4,8,0.01) 


y1_axis = y1[0]*y1.size

y1_axis

y1.size
print(y1.size)

y1_new = y1.reshape(y1.size,1)
x1_new = x1.reshape(1,y1.size)

X1,Y1 = np.meshgrid(x1,y1_axis)
R = X1*np.cos(np.sqrt(x1**2+y1_axis**2)) + Y1*np.cos(np.sqrt(x1**2+y1_axis**2))

# Plot a basic wireframe
ax.plot_wireframe(X1, Y1, Z1, rstride=1, cstride=1)
ax.set_title('row step size 10, column step size 10')

ax.plot_wireframe(X1, Y1, Z1, rstride=1, cstride=1)
ax.set_title('row step size 20, column step size 20')

#plt.show()

surf = ax1.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.8, cmap=cm.ocean)
cset = ax1.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.ocean)
cset = ax1.contourf(X, Y, Z, zdir='x', offset=-5, cmap=cm.ocean)
cset = ax1.contourf(X, Y, Z, zdir='y', offset=-4, cmap=cm.ocean)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax1.set_xlabel('X')
ax1.set_xlim(-15, 15)
ax1.set_ylabel('Y')
ax1.set_ylim(-15, 15)
ax1.set_zlabel('Z')
ax1.set_zlim(np.min(Z), np.max(Z))
ax1.set_title('3D surface with 2D contour plot projections')

plt.show()


# In[11]:





# In[ ]:




