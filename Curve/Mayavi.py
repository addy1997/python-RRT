#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import pi, cos, sin
from mayavi import mlab 
print("For reference - https://docs.enthought.com/mayavi/mayavi/")



alpha, beta = np.mgrid[0:pi:180j,10:2*pi:360j]

m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 1; m5 = 2; m6 = 2; m7 = 4;

s = sin(m0*alpha)**m1 + cos(m2*alpha)**m3 + sin(m4*beta)**m5 + cos(m6*beta)**m7

x = sin(alpha)*cos(beta) #points for X-axis

y = x[10]*cos(alpha) #points for Y-axis

z = sin(alpha)*sin(beta) #shaping the z-axis 


# Plot the data

# first plot in 3D

fig = mlab.figure(1)

mlab.clf()

mesh = mlab.mesh(x, y, z, scalars=s)

cursor3d = mlab.points3d(0., 0., 0., mode='axes',
                                color=(0, 0, 0),
                                scale_factor=0.5)

mlab.title('Click on the shape')

# A second plot, flat
fig2d = mlab.figure(2)

mlab.clf()

im = mlab.imshow(s)

cursor = mlab.points3d(0, 0, 0, mode='2dthick_cross',
                                color=(0, 0, 0),
                                scale_factor=10)

mlab.view(90, 0)



# Some logic to select 'mesh' and the data index when picking.
#This function gives the coordinates of the point on the surface of the mesh  

def pick_the_coords(picker_obj):
    picked = picker_obj.actors
    if mesh.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
       
        x_, y_ = np.lib.index_tricks.unravel_index(picker_obj.point_id,
                                                                s.shape)
        print("coordinates: %i, %i" % (x_, y_))
        
        n_x, n_y = s.shape
        
        cursor.mlab_source.reset(x=x_ - n_x/4.,
                               y=y_ - n_y/4.)
        cursor3d.mlab_source.reset(x=x[x_, y_],
                                 y=y[x_, y_],
                                 z=z[x_, y_])

fig.on_mouse_pick(pick_the_coords)#to get the coordinates

mlab.show()


# In[1]:





# In[ ]:




