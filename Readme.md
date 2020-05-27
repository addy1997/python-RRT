## python-RRT

### This repository contains the sampling based motion planning algorithms like RRT(Rapidly exploring Random Tree) and PRM(Probabilistic Roadmap) in 2D configuration space with pygame.

These algorithms are part of my project at HTIC, IIT-Madras during my research internship.

We tried a variety of algorithms like RRT, A-star, Dijkstra's, PRM etc in 2D space. RRT among them proves to be the most promising candidate for efficient motion planning in 2D as well as 3D space.<br>

These algorithms have a variety of applications ranging from protein folding to collision avoidance of mars rover. <br>  

<br>
<b> Developers: </b> Adwait P Naik <br>
<b> Guide: </b> Mr. Manojkumar L @ HTIC, IIT-Madras<br>
<b> Project: </b> Motion Planning for UR5 robotic Arm<br>
<b> Duration:</b> November 2019 - February 2020 <br>

## Problem Statement

To determine a collision free path in Kinodynamic environment provided the obstacles are static.

## Goal

During accuracy testing & TCP calibration we observed that the our UR5 robot after collision with the table or the mount stopped with a backslash. The trajectories were not smooth at all which affected the accuracy. So we tested a few algorithms to

1) have a smooth trajectory
2) accurate obstacle avoidance and
3) be memory efficient and computationally cheap.

# Libraries used

<ul>
<li> <b> for animation</b> - Pygame</li>
<li> <b> for calculations</b> - numpy, itertools etc</li>
</ul>

# Running requirements

<ul>
<li> Python 3.7 </li>
<li> Pygame and Tkinter </li>
</ul>

## Results
### RRT
<ul>
<b> Rapidly-exploring Random Tree </b>
<li>Implemented a basic RRT algorithm using the pygame library in python</li>
<li> <b>Observations -</b> RRT is an asymptotically optimal algorithm. In other words, it obtains a solution path when the number of iterations tend to infinity. It's probabilistically complete which means that if the path exists it will return it.</li>

<li> <b>disadvantages - </b> In RRT, each time the graph is constructed from scratch which makes it inefficient for finding paths in larger space.
<li> Space complexity of RRT is O(n). In other words it has a linear space complexity which means that if the search space increases the iterations to find an optimal path will also increase subsequently. </li>
</ul>

![RRT](https://github.com/addy1997/Internship-HTIC/blob/master/Motion%20planning/RRT%20variants/Screenshots/RRT%20with%20obstacles.png)

### RRT-star
<ul>
<b>Rapidly-exploring Random Tree star </b>
<li>Implemented RRT-star algorithm using pygame.</li>
<li>RRT* unlike RRT has a slightly different strategy for finding the nearest node which makes it computationally more efficient than RRT.</li>
<li>RRT* takes less iterations to find the optimal path as compares to RRT.</li>RRT* is also asymptotically optimal like RRT.</li>

<li> <b>disadvantages - </b> In RRT-star, each time the graph is constructed from scratch which makes it inefficient for finding paths in larger space like RRT.
<li> Space and Time complexity of RRT is O(logn) and O(nlogn). In other words it has a linear space complexity which means that if the search space increases the iterations to find an optimal path will also increase subsequently. </li>
<li> Link to the research paper <a href = "https://link.springer.com/chapter/10.1007/978-3-319-16841-8_7" > Research paper</a></li>
  

</ul>

![RRT*](https://github.com/addy1997/Internship-HTIC/blob/master/Motion%20planning/RRT%20variants/Screenshots/RRT%20star%20with%20obstacles2.png)

### RRT-star FN
<ul>
<b>Rapidly-exploring Random Tree star Fixed Nodes </b>
<li>Implemented RRT-star algorithm using pygame.</li>
<li>Here we have created circular obstacles to test for the collision free path.</li>
<li>This is the modified version of RRT star probabilistically complete in nature. </li>
<li>It grows the tree till the count reaches fixed amount of nodes and then optimizes the tree further by removing the weaker node by adding a node with high probability to converge an efficient path.</li>
  
<li> Link to the research paper <a href = "https://www.researchgate.net/publication/261271325_Rapidly-exploring_random_tree_based_memory_efficient_motion_planning/download" > Research paper</a></li>
  
</ul>

![RRT star FN](https://github.com/addy1997/Internship-HTIC/blob/master/Motion%20planning/RRT%20variants/Screenshots/RRT%20Finite%20Node.png)

### RRT with B-spline and node pruning
<ul>
<b>Rapidly-exploring Random Tree with B-spline and Node pruning</b>

![RRT-Bspline and Pruning](https://github.com/addy1997/Internship-HTIC/blob/master/Motion%20planning/RRT%20variants/Screenshots/RRT-pruning1.png)

![RRT-Bspline and Pruning](https://github.com/addy1997/Internship-HTIC/blob/master/Motion%20planning/RRT%20variants/Screenshots/RRT-pruning2.png)

![RRT-Bspline and Pruning](https://github.com/addy1997/Internship-HTIC/blob/master/Motion%20planning/RRT%20variants/Screenshots/RRT-pruning3.png)

### RRT-bidirectional
<ul>
<b> Rapidly-exploring Random Tree bidirectional</b>
<li> In this variant the tree is made to grow from the start node and the goal node as well until it meets at a particular point</li>

![RRT-bidirectional](https://github.com/addy1997/Internship-HTIC/blob/master/Motion%20planning/RRT%20variants/Screenshots/RRT%20connect%20manhattan.png)

### RRT-bidirectional with Manhattan distance
<ul>
<li>The manhattan distance concept is based on the <a href = "https://en.wikipedia.org/wiki/Taxicab_geometry">Taxicab geometry</a></li>
  
![RRT-bidirectional](https://github.com/addy1997/Internship-HTIC/blob/master/Motion%20planning/RRT%20variants/Screenshots/Screenshot%202020-03-02%20at%2010.09.56%20PM.png)

![RRT-bidirectional](https://github.com/addy1997/Internship-HTIC/blob/master/Motion%20planning/RRT%20variants/Screenshots/Screenshot%202020-03-02%20at%2010.10.09%20PM.png)


### Astar Algorithm
<ul>
<b>Astar Algorithm</b>
<li> Astar is an algorithm based on <b> graph traversal and path search</b> used in travel-routing systems and path finding</li>
<li> It is based o heuristic approach governed by equation f(n) = g(n) + h(n). The estimation is done on the basis of cost of the path and cost to expand the nodes in particular direction.</li>
</ul>

![Astar](https://github.com/addy1997/Internship-HTIC/blob/master/Motion%20planning/A%20star/Screenshot%202020-02-21%20at%2010.58.12%20AM.png)

  
### ROS exercises
<ul>
<b> Please note that these are random exercises to which the source code can be found at <a href = "https://github.com/addy1997/Internship-HTIC/tree/master/ROS%20codes"> Ros-codes </a> </b>
  
![creating objects in rviz](https://github.com/addy1997/Internship-HTIC/blob/master/screenshots/Screenshot%202019-09-25%20at%202.45.28%20PM.png)

![creating objects in rviz](https://github.com/addy1997/Internship-HTIC/blob/master/screenshots/Screenshot%202019-09-25%20at%202.46.30%20PM.png)

![creating objects in rviz](https://github.com/addy1997/Internship-HTIC/blob/master/screenshots/Screenshot%202019-09-29%20at%201.03.28%20PM.png)

![creating objects in rviz](https://github.com/addy1997/Internship-HTIC/blob/master/screenshots/Screenshot%202019-09-29%20at%201.03.41%20PM.png)

![creating objects in rviz](https://github.com/addy1997/Internship-HTIC/blob/master/screenshots/Screenshot%202019-09-25%20at%202.51.00%20PM.png)

</ul>

<b> Robotic arm trajectory and inverse kinematics </b>

![UR5](https://github.com/addy1997/Internship-HTIC/blob/master/screenshots/Simulation-of-UR-10-robot-in-Gazebo-simulator.png)


![UR5](https://github.com/addy1997/Internship-HTIC/blob/master/screenshots/43643941-31346fb8-972d-11e8-91d8-b48f7.png)


![UR5](https://github.com/addy1997/Internship-HTIC/blob/master/screenshots/56207743-097e7780-603f-11e9-8461-b4d353e21496.png)

</ul>

### Quadtree 

<ul>
<b> Quadtree using matplotlib </b>
<li> Quadtree is a data structure used to divide the configuration space recursively into smaller segments. </li>
  
![Quadtree](https://github.com/addy1997/Internship-HTIC-IIT-Madras/blob/master/screenshots/Screenshot%202019-12-28%20at%203.34.58%20PM.png)
  
![Quadtree](https://github.com/addy1997/Internship-HTIC-IIT-Madras/blob/master/screenshots/Screenshot%202019-12-28%20at%203.35.10%20PM.png)

![Quadtree](https://github.com/addy1997/Internship-HTIC-IIT-Madras/blob/master/screenshots/Screenshot%202019-12-28%20at%203.35.23%20PM.png)
  
![Quadtree](https://github.com/addy1997/Internship-HTIC-IIT-Madras/blob/master/screenshots/Screenshot%202019-12-28%20at%203.38.01%20PM.png)

![Quadtree](https://github.com/addy1997/Internship-HTIC-IIT-Madras/blob/master/screenshots/Screenshot%202019-12-28%20at%203.38.27%20PM.png)

![Quadtree](https://github.com/addy1997/Internship-HTIC-IIT-Madras/blob/master/screenshots/Screenshot%202019-12-28%20at%203.39.13%20PM.png)

![Quadtree](https://github.com/addy1997/Internship-HTIC-IIT-Madras/blob/master/screenshots/Screenshot%202019-12-28%20at%203.39.34%20PM.png)
















