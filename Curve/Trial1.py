#!/usr/bin/env python
# coding: utf-8

# In[4]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)

X,Y = np.meshgrid(x,y)
Z = - X**4*np.sqrt(x**2+y**2) + Y**2*np.sqrt(x**2+y**2)


fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.8, cmap=cm.ocean)
cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.ocean)
cset = ax.contourf(X, Y, Z, zdir='x', offset=-4, cmap=cm.ocean)
cset = ax.contourf(X, Y, Z, zdir='y', offset=5, cmap=cm.ocean)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax.set_xlabel('X')
ax.set_xlim(-5, 5)
ax.set_ylabel('Y')
ax.set_ylim(-5, 5)
ax.set_zlabel('Z')
ax.set_zlim(np.min(Z), np.max(Z))
ax.set_title('3D surface with 2D contour plot projections')

plt.show()


# In[48]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


x = np.arange(4,8,0.01)
y = np.arange(4,8,0.01)
X,Y = np.meshgrid(x,y)
Z = - X**4*np.sqrt(x**2+y**2) + Y**2*np.sqrt(x**2+y**2)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.8, cmap=cm.ocean)
cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.ocean)
cset = ax.contourf(X, Y, Z, zdir='x', offset=-5, cmap=cm.ocean)
cset = ax.contourf(X, Y, Z, zdir='y', offset=5, cmap=cm.ocean)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax.set_xlabel('X')
ax.set_xlim(-5, 5)
ax.set_ylabel('Y')
ax.set_ylim(-5, 5)
ax.set_zlabel('Z')
ax.set_zlim(np.min(Z), np.max(Z))
ax.set_title('3D surface with 2D contour plot projections')

plt.show()

x = np.arange(4,8,0.01)
y = np.arange(4,8,0.01)
X,Y = np.meshgrid(x,y)
Z = - X**4*np.sqrt(x**2+y**2) - Y**2*np.sqrt(x**2+y**2)

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(122, projection='3d')

surf = ax1.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.8, cmap=cm.ocean)
cset = ax1.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.ocean)
cset = ax1.contourf(X, Y, Z, zdir='x', offset=-5, cmap=cm.ocean)
cset = ax1.contourf(X, Y, Z, zdir='y', offset=5, cmap=cm.ocean)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax1.set_xlabel('X')
ax1.set_xlim(-5, 5)
ax1.set_ylabel('Y')
ax1.set_ylim(-5, 5)
ax1.set_zlabel('Z')
ax1.set_zlim(np.min(Z)*np.sin(np.pi), np.max(Z)*np.sin(np.pi))
ax1.set_title('3D surface with 2D contour plot projections')

plt.show()


# In[6]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)

X,Y = np.meshgrid(x,y)

Z = - X**4*np.sqrt(x**2+y**2) + Y**2*np.sqrt(x**2+y**2)

fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.8, cmap=cm.ocean)

cset = ax.contourf(X, Y, Z, zdir='x', offset=np.min(Z), cmap=cm.ocean)

cset = ax.contourf(X, Y, Z, zdir='x', offset=0, cmap=cm.ocean)

cset = ax.contourf(X, Y, Z, zdir='x', offset=5, cmap=cm.ocean)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax.set_xlabel('X')
ax.set_xlim(-5, 5)
ax.set_ylabel('Y')
ax.set_ylim(-5, 5)
ax.set_zlabel('Z')
ax.set_zlim(np.min(Z), np.max(Z))
ax.set_title('3D surface with 2D contour plot projections')

plt.show()


# In[79]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


x = np.arange(-2,2,0.1)
y = np.arange(-2,2,0.1)

X,Y = np.meshgrid(x,y)

Z = - X**4*np.sqrt(x**2+y**2) + Y**2*np.sqrt(x**2+y**2)

fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.8, cmap=cm.ocean)

cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.ocean)

cset = ax.contourf(X, Y, Z, zdir='x', offset=5, cmap=cm.ocean)

cset = ax.contourf(X, Y, Z, zdir='y', offset=-5, cmap=cm.ocean)

fig.colorbar(surf, ax=ax, shrink=10, aspect=5)

ax.set_xlabel('X')
ax.set_xlim(-5, 5)
ax.set_ylabel('Y')
ax.set_ylim(-5, 5)
ax.set_zlabel('Z')
ax.set_zlim(np.min(Z), np.max(Z))
ax.set_title('3D surface with 2D contour plot projections')

plt.show()


# In[24]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)

X,Y = np.meshgrid(x,y)

Z = - X**4*np.sqrt(x**2+y**2) + Y**2*np.sqrt(x**2+y**2)

fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.8, cmap=cm.ocean)

cset = ax.contourf(X, Y, Z, zdir='x', offset=np.min(Z), cmap=cm.ocean)

cset = ax.contourf(X, Y, Z, zdir='y', offset=-5, cmap=cm.ocean)

cset = ax.contourf(X, Y, Z, zdir='z', offset=5, cmap=cm.ocean)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax.set_xlabel('X')
ax.set_xlim(-5, 5)
ax.set_ylabel('Y')
ax.set_ylim(-5, 5)
ax.set_zlabel('Z')
ax.set_zlim(np.min(Z), np.max(Z))
ax.set_title('3D surface with 2D contour plot projections')

plt.show()


# In[ ]:




