#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Copyright 2019, Adwait P Naik
# All rights reserved.

# created on 21st October at 13:42:50
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIB

#importing the packages
import sys
import copy
import rospy
import moveit_msgs.msg
from geometry_msgs.msg import PoseStamped, Pose
import moveit_commander
from shape_msgs.msg import SolidPrimitive
import geometry_msgs.msg

try:
    from pyassimp import pyassimp
    use_pyassimp = True
except:
    use_pyassimp = False

In [37]:

#class for creating shape
class create_objects(object):
    
    robot = 0
    scene = 0
    group = 0
    planning_frame = 0
    eef_link = 0
    group_names = 0
    pose_goal = 0
    
    def __init__(self):
        super(create_objects, self).__init__()
        
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('create_objects',anonymous=True)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = "manipulator"
        group = moveit_commander.MoveGroupCommander(group_name)
        
        #robot details
        planning_frame = group.get_planning_frame()
        eef_link = group.get_end_effector_link()
        group_names = robot.get_group_names()
        
        #declaring global variables
        self.robot = robot
        self.scene = scene
        self.group = group
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        self.pose_goal = geometry_msgs.msg.Pose()
        
        #Track the objects in the planning scene
        
        

        
        #function to create cylinder
    def addCylinder(self, name, height, radius, pos_x, pos_y, pos_z):
        group = self.group
        scene = self.scene
        eef_link = self.eef_link
        robot = self.robot
        
        s = SolidPrimitive()
        s.dimensions = [height, radius]
        s.type = s.CYLINDER

        ps = PoseStamped()
        ps.header.frame_id = self.planning_frame
        ps.pose.position.x = pos_x
        ps.pose.position.y = pos_y
        ps.pose.position.z = pos_z
        ps.pose.orientation.w = 1.0
        ps.pose.orientation.x = 2.0
        ps.pose.orientation.y = 3.0
        ps.pose.orientation.z = 2.0
            
        scene.add_cylinder(name, ps, height, radius)


        
        
        #Function to add a Box
    def addBox(self, name, size_x, size_y, size_z, pos_x, pos_y, pos_z):
        group = self.group
        scene = self.scene
        eef_link = self.eef_link
        robot = self.robot
        
        s = SolidPrimitive()
        s.dimensions = [size_x, size_y, size_z]
        s.type = s.BOX
             
        ps = PoseStamped()
        ps.header.frame_id = self.planning_frame
        ps.pose.position.x = pos_x
        ps.pose.position.y = pos_y
        ps.pose.position.z = pos_z
        ps.pose.orientation.w = 1.0
        ps.pose.orientation.x = 2.0
        ps.pose.orientation.y = 3.0
        ps.pose.orientation.z = 2.0
                
        scene.add_box(name, ps, size=(size_x, size_y, size_z))     
        

        #Function to add Sphere
    def addSphere(self, name, radius, pos_x, pos_y, pos_z):
        
        group = self.group
        scene = self.scene
        eef_link = self.eef_link
        robot = self.robot
        
        s = SolidPrimitive()
        s.dimensions = [radius]
        s.type = s.SPHERE

        ps = PoseStamped()
        ps.header.frame_id = self.planning_frame
        ps.pose.position.x = pos_x
        ps.pose.position.y = pos_y
        ps.pose.position.z = pos_z
        ps.pose.orientation.w = 1.0
        ps.pose.orientation.x = 2.0
        ps.pose.orientation.y = 3.0
        ps.pose.orientation.z = 2.0
        scene.add_sphere(name, ps, radius) 
        
        
           #Function to add Cube
    def addCube(self, name, size, pos_x,pos_y,pos_z):
        group = self.group
        scene = self.scene
        eef_link = self.eef_link
        robot = self.robot
        
        s = SolidPrimitive()
        s.dimensions = [size]
        s.type = s.CUBE

        ps = PoseStamped()
        ps.header.frame_id = self.planning_frame
        ps.pose.position.x = pos_x
        ps.pose.position.y = pos_y
        ps.pose.position.z = pos_z
        ps.pose.orientation.w = 1.0
        ps.pose.orientation.x = 2.0
        ps.pose.orientation.y = 3.0
        ps.pose.orientation.z = 2.0
        scene.add_cube(name, ps, size) 
        
        #Function to add Cone
    def addCone(self, name, height, radius, pos_x, pos_y, pos_z):
        
        group = self.group
        scene = self.scene
        eef_link = self.eef_link
        robot = self.robot
        
        s = SolidPrimitive()
        s.dimensions = [height, radius]
        s.type = s.CONE

        ps = PoseStamped()
        ps.header.frame_id = self.planning_frame
        ps.pose.position.x = pos_x
        ps.pose.position.y = pos_y
        ps.pose.position.z = pos_z
        ps.pose.orientation.w = 1.0
        ps.pose.orientation.x = 2.0
        ps.pose.orientation.y = 3.0
        ps.pose.orientation.z = 2.0
            
        scene.add_cone(name, ps, height, radius) 
        

In [38]:

p = create_objects()

In [39]:

Scene = p.scene
Group = p.group

In [40]:

#p.addCylinder("name1", 0.35, 0.35, 0.5, 0.5, 0.5)

In [41]:

#p.addCone("name2", 0.25, 0.5, 1.5, 2.5, 3.5)

In [42]:

p.addSphere("name3", 0.35, 1.5, 0.5, 4.5)

In [43]:

#p.addCube( "name4", 3.5, 4.5, 4.5, 4.5)

In [44]:

p.addBox( "name5", 0.2, 0.2, 0.4, 2.5, 2.5, 2.5)

