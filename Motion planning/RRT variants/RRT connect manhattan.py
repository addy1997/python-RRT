#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math, sys, pygame, random
from math import *
from pygame import *
from scipy.spatial import distance


class Node(object):
    def __init__(self, point, parent):
        super(Node, self).__init__()
        self.point = point
        self.parent = parent
        self.cost=0

#20*15
level1 = ["xxx.xxxxxxxxxxxxxxxx",
          "xs.....x............",
          "xxxx.........xx....x",
          "x......x....x.x....x",
          "x......x......x....x",
          "x......x......x....x",
          "x...xxxxxxxx..x....x",
          "x......x...........x",
          "x.............xxxxxx",
          "xxxxxx.x...........x",
          "x......x.xxx.......x",
          "x......x...........x",
          "x..........xxx..xxxx",
          "x..........x......nx",
          "xxxxxxxxxxxxxxxxxxxx"]

level2 = ["xxx.xxxxxxxxxxxxxxxx",
          "x......xxxxxx.......",
          "x......xxxxxx......x",
          "x......xxxxxx.x....x",
          "x......xxxxxx.x....x",
          "x......xxxxxx.x....x",
          "x..................x",
          "x......xx.xxx......x",
          "x......xx.xxx.xxxxxx",
          "xxxxxx.xx.xxx......x",
          "x......xx..........x",
          "x......xxxxxx......x",
          "x......xxxxxx...xxxx",
          "x......xxxxxx.....xx",
          "xxxxxxxxxxxxxxxxxxxx"]

level3 = ["x...xxxxxxxxxxxxxxxx",
          "x..................x",
          "xxxxxxxxxxxxxxxxx..x",
          "x..................x",
          "x..xxxxxxxxxxxxxxxxx",
          "x..................x",
          "xxxxxxxxxxxxxxxxx..x",
          "x..................x",
          "x..xxxxxxxxxxxxxxxxx",
          "x..................x",
          "xxxxxxxxxxxxxxxxx..x",
          "x..................x",
          "x..xxxxxxxxxxxxxxxxx",
          "x..........x.......x",
          "xxxxxxxxxxxxxxxxxxxx"]

level4 = ["x...xxxxxxxxxxxxxxxx",
          "x..................x",
          "x..................x",
          "x..................x",
          "x..xxxxxxxxxxxxxxxxx",
          "x..................x",
          "x..................x",
          "xxxxxxxxxxxxxxxxx..x",
          "x..................x",
          "x..................x",
          "x..xxxxxxxxxxxxxxxxx",
          "x..................x",
          "x..................x",
          "x..................x",
          "xxxxxxxxxxxxxxxxxxxx"]


XDIM = 800
YDIM = 600
windowSize = [XDIM, YDIM]
delta = 10.0
GAME_LEVEL = 1
GOAL_RADIUS = 10
MIN_DISTANCE_TO_ADD = 1.0
NUMNODES = 5000
pygame.init()
fpsClock = pygame.time.Clock()
screen = pygame.display.set_mode(windowSize)
screenrect = screen.get_rect()
white = 255, 255, 255
black = 0, 0, 0
red = 255, 0, 0
blue = 0, 255, 0
green = 0, 0, 255
cyan = 0,180,105
teal = 193,205,193
olive = 85,107,47
navy = 0,0,128
inred = 205,92,92
dgreen = 0,100,0
dblue = 30,144,255

count = 0
rectObs = []


def dist(p1,p2):    #distance between two points
    return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

def chebyshev_dist(p1,p2):
    return distance.chebyshev(p1[0],p2[0])

def point_circle_collision(p1, p2, radius):
    distance = dist(p1,p2)
    if (distance <= radius):
        return True
    return False

def step_from_to(p1,p2):
    if dist(p1,p2) < delta:
        return p2
    else:
        theta = atan2(p2[1]-p1[1],p2[0]-p1[0])
        return p1[0] + delta*cos(theta), p1[1] + delta*sin(theta)

def collides(p):    #check if point collides with the obstacle
    for rect in rectObs:
        if rect.collidepoint(p) == True:
            return True
    return False



def get_random_clear():
    while True:
        p = random.random()*XDIM, random.random()*YDIM
        noCollision = collides(p)
        if noCollision == False:
            return p


def init_obstacles(level):  
    global rectObs
    rectObs = []

    lines = len(level)
    columns = len(level[0])

    length = screenrect.width / columns
    height = screenrect.height / lines

    for yi in range(lines):
        for xi in range(columns):
            if level[yi][xi] == 'x': # wall
                rectObs.append(pygame.Rect((length * xi, height * yi),(length,height)))

    for rect in rectObs:
        pygame.draw.rect(screen, black, rect)


def reset():
    global count
    screen.fill(white)
    init_obstacles(level3)
    count = 0

def main():
    global count
    
    initPoseSet = False
    initialPoint = Node(None, None)
    goalPoseSet = False
    goalPoint = Node(None, None)
    currentState = 'init'

    nodes = []
    nodes1 = []
    reset()

    while True:
        if currentState == 'init':
            #print('goal point not yet set')
            pygame.display.set_caption('Select Starting Point and then Goal Point')
            fpsClock.tick(10)
        elif currentState == 'goalFound':
            currNode = goalNode
            currNode1 = goalNode1
            pygame.display.set_caption('Goal Reached')
            #print ("Goal Reached")

            
            while currNode.parent != None:
                pygame.draw.line(screen,red,currNode.point,currNode.parent.point,2)
                currNode = currNode.parent
            while currNode1.parent != None:
                pygame.draw.line(screen,navy,currNode1.point,currNode1.parent.point,2)
                currNode1 = currNode1.parent
            optimizePhase = True
        elif currentState == 'optimize':
            fpsClock.tick(0.5)
            pass
        elif currentState == 'buildTree':
            count = count+1
            pygame.display.set_caption('Performing RRT_CONNECT')
            if count < NUMNODES:
                foundNext = False
                while foundNext == False:
                    rand = get_random_clear()
                    parentNode = nodes[0]
                    for p in nodes:
                        if chebyshev_dist(p.point,rand) <= chebyshev_dist(parentNode.point,rand):
                            newPoint = step_from_to(p.point,rand)
                            if collides(newPoint) == False:
                                parentNode = p
                                #print('hello')
                                foundNext = True
                newnode = step_from_to(parentNode.point,rand)
                newnode1 = Node(newnode, parentNode)
                newnode1.cost=parentNode.cost+delta
                nodes.append(newnode1)
                pygame.draw.line(screen,olive,parentNode.point,newnode)
                foundNext1 = False
                while foundNext1 == False:
                    rand1 = newnode1.point
                    parentNode1 = nodes1[0]
                    for p in nodes1:
                        if chebyshev_dist(p.point,rand1) <= chebyshev_dist(parentNode1.point,rand1):
                            newPoint = step_from_to(p.point,rand1)
                            if collides(newPoint) == False:
                                parentNode1 = p
                                foundNext1 = True
                #print('outside loop')
                newnode2 = step_from_to(parentNode1.point,rand1)
                newnode3 = Node(newnode2,parentNode1)
                nodes1.append(newnode3)
                pygame.draw.line(screen,dblue,parentNode1.point,newnode2)
                reached = False
                collide = False
                while (reached == False and collide == False):
                    newnode4 = step_from_to(newnode3.point,rand1)
                    if collides(newnode4) == False:
                        newnode5 = Node(newnode4,nodes1[len(nodes1)-1])
                        nodes1.append(newnode5)
                        pygame.draw.line(screen,dblue,newnode3.point,newnode4)
                        newnode3=newnode5
                        if newnode4 == rand1:
                            reached = True
                        else:
                        	reached = False
                    else:
                        collide = True
                if reached == True:
                    currentState = 'goalFound'
                    goalNode1 = nodes1[len(nodes1)-1]
                    goalNode = nodes[len(nodes)-1]
                    print('goalFound')
        #handle events
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                sys.exit("Exiting")
            if e.type == MOUSEBUTTONDOWN:
                #print('mouse down')
                if currentState == 'init':
                    if initPoseSet == False:
                        nodes = []
                        nodes1 = []
                        if collides(e.pos) == False:
                            print('initiale point set: '+str(e.pos))

                            initialPoint = Node(e.pos, None)
                            nodes.append(initialPoint) # Start in the center
                            initPoseSet = True
                            pygame.draw.circle(screen, olive, initialPoint.point, GOAL_RADIUS)
                    elif goalPoseSet == False:
                        print('goal point set: '+str(e.pos))
                        if collides(e.pos) == False:
                            goalPoint = Node(e.pos,None)
                            nodes1.append(goalPoint)
                            goalPoseSet = True
                            pygame.draw.circle(screen, navy, goalPoint.point, GOAL_RADIUS)
                            currentState = 'buildTree'
                else:
                    currentState = 'init'
                    initPoseSet = False
                    goalPoseSet = False
                    reset()

        pygame.display.update()
        fpsClock.tick(10000)



if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




