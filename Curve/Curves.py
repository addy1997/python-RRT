#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def paraboloid_function(x, y):
    return (-x**2) + (-y**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-4.0, 4.0, 0.25)
X, Y = np.meshgrid(x, y)
zs = np.array([paraboloid_function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

scale_factor = 0.856

Vertex = (np.cos(x)+np.sin(y))+(np.cos(x)+np.sin(y))/(scale_factor)


ax.plot_surface(X, Y, Z, Vertex)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


fig = plt.figure()

ax1 = fig.add_subplot(111, projection='3d')
x1 = y1 = np.arange(-4.0, 4.0, 0.25)
X1, Y1 = np.meshgrid(x1, y1)
zs1 = np.array([paraboloid_function(x1,y1) for x1,y1 in zip(np.ravel(X1), np.ravel(Y1))])
Z1 = zs1.reshape(X1.shape)

ax1.plot_surface(X1, Y1, Z1)

ax1.set_xlabel('X1 Label')
ax1.set_ylabel('Y1 Label')
ax1.set_zlabel('Z1 Label')

plt.show()


# In[17]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def paraboloid_function(x, y):
    return x**2 + y


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([paraboloid_function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

scale_factor = 4

Vertex = (x+y)/scale_factor

ax.plot_surface(X, Y, Z, Vertex)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
x1 = y1 = np.arange(-3.0, 3.0, 0.05)
X1, Y1 = np.meshgrid(x1, y1)
zs1 = np.array([paraboloid_function(x1,y1) for x1,y1 in zip(np.ravel(X1), np.ravel(Y1))])
Z1 = zs1.reshape(X1.shape)



ax1.plot_surface(X1, Y1, Z1)

ax1.set_xlabel('X1 Label')
ax1.set_ylabel('Y1 Label')
ax1.set_zlabel('Z1 Label')

plt.show()


# In[16]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def paraboloid_function(x, y):
    return x**2 - y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([paraboloid_function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

scale_factor = 4

Vertex = (x+y)/scale_factor

ax.plot_surface(X, Y, Z, Vertex)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# In[17]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


def paraboloid_function(x, y):
    return - x + y**2

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = y = np.arange(-3.0, 3.0, 0.05)

X, Y = np.meshgrid(x, y)

scale_factor = 4

Vertex = (x+y)/scale_factor

zs = np.array([paraboloid_function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])

Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z, Vertex)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# In[ ]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def paraboloid_function(x, y):
    return np.sqrt((np.cos(x**2)+ np.cos(y**2)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([paraboloid_function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# In[18]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def paraboloid_function(x, y):
    return x**2 - y**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([paraboloid_function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

scale_factor = 4

Vertex = (x+y)/scale_factor

ax.plot_surface(X, Y, Z, Vertex)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# In[65]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def paraboloid_function(x, y):
    return np.sqrt(-x**2 + y**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([paraboloid_function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

#scale_factor = 4

#Vertex = (x+y)/scale_factor

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# In[35]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def paraboloid_function(x, y):
    return np.tan(2*x**2 + 2*y**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([paraboloid_function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

scale_factor = 4

Vertex = np.sqrt(x+y)/scale_factor

ax.plot_surface(X, Y, Z, Vertex)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# In[28]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def paraboloid_function(x, y):
    return x**2 + y*2 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([paraboloid_function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

scale_factor = 4

Vertex = (x+y*0.25)/scale_factor

ax.plot_surface(X, Y, Z, Vertex)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# In[38]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def paraboloid_function(x, y):
    return np.cos(np.sqrt(x**2 - y**2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([paraboloid_function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

scale_factor = 4

Vertex = (np.cos(x)+np.sin(y))/scale_factor

ax.plot_surface(X, Y, Z, Vertex)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
 
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):

    #Given a quadratic equation say z = A(x,y)s^2 + B(x,y)s+ C
    #make 3 equations and 3 unknowns and solve simultaneously for A, B, C. 
    
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    
    A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    
    B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    
    C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom
    
    return A,B,C
    
    
x1,y1 = [60,0]

x2,y2 = [-x1,y1]

x3,y3 = [0,5]

x4,y4 = [(x1+x2)/2,(y1+y2)/2]

    #Calculate the unknowns of the equation y=ax^2+bx+c
a,b,c=calc_parabola_vertex(x1, y1, x2, y2, x3, y3)

#a = z[0]
#b = z[0][0]
#c = z[0][1]

print ("the three values",a,b,c)
    
    
x_pos=np.arange(-60,60,0.5)
y_pos=[]

    #Calculate y values 
for x in range(len(x_pos)):
    x_val=x_pos[x]
    y=(a*(x_val**2))+(b*x_val) + c
    y_pos.append(y)

plt.plot(x_pos, y_pos, linestyle='-.', color='black') # parabola line
plt.scatter(x_pos, y_pos, color='gray') # parabola points
plt.scatter(x1,y1,color='red',marker="D",s=50) # 1st known xy
plt.scatter(x2,y2,color='green',marker="D",s=50) # 2nd known xy
plt.scatter(x3,y3,color='black',marker="D",s=50) # 3rd known xy
plt.scatter(x4,y4,color='gold', marker="D", s=50) # 4th mid point
plt.show()    


# In[ ]:




