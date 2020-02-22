#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math, sys, pygame, random
from math import *
from pygame import *

class Node(object):
    def __init__(self, point, parent):
        super(Node, self).__init__()
        self.point = point
        self.parent = parent

XDIM = 720
YDIM = 500
windowSize = [XDIM, YDIM]
delta = 40.0
GAME_LEVEL = 1
GOAL_RADIUS = 10
MIN_DISTANCE_TO_ADD = 1.0
NUMNODES = 5000
pygame.init()
fpsClock = pygame.time.Clock()
screen = pygame.display.set_mode(windowSize)
white = 255, 255, 255
black = 0, 0, 0
red = 255, 0, 0
green = 0, 255, 0
blue = 0, 0, 255
cyan = 0,180,105
dark_green = 0, 102, 0

count = 3
rectObs = []

def dist(p1,p2):     #distance between two points
    return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

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


def init_obstacles(configNum):  #initialized the obstacle
    global rectObs
    rectObs = []
    rectObs.append(pygame.Rect((XDIM / 2.0 + 200, YDIM / 2.0 - 180),(100,370)))
    rectObs.append(pygame.Rect((370,100),(150,160)))
    rectObs.append(pygame.Rect((400,300),(100,80)))
    rectObs.append(pygame.Rect((250,180),(80,120)))
    rectObs.append(pygame.Rect((100,100),(80,80)))
    for rect in rectObs:
        pygame.draw.rect(screen, cyan, rect)


def reset():
    global count
    screen.fill(white)
    init_obstacles(GAME_LEVEL)
    count = 0

def main():
    global count
    
    initPoseSet = False
    initialPoint = Node(None, None)
    goalPoseSet = False
    goalPoint = Node(None, None)
    currentState = 'init'

    nodes = []
    reset()

    while True:
        if currentState == 'init':
            print('goal point not yet set')
            pygame.display.set_caption('Select Starting Point and then Goal Point')
            fpsClock.tick(10)
        elif currentState == 'goalFound':
            currNode = goalNode.parent
            pygame.display.set_caption('Goal Reached')
            print ("Goal Reached")
            
            
            while currNode.parent != None:
                pygame.draw.line(screen,red,currNode.point,currNode.parent.point)
                currNode = currNode.parent
            optimizePhase = True
        elif currentState == 'optimize':
            fpsClock.tick(0.5)
            pass
        elif currentState == 'buildTree':
            count = count+1
            pygame.display.set_caption('Performing RRT')
            if count < NUMNODES:
                foundNext = False
                while foundNext == False:
                    rand = get_random_clear()
                    rand2 = get_random_clear()
                    rand3 = get_random_clear()
                    rand4 = get_random_clear()
                    rand5 = get_random_clear()
                    rand6 = get_random_clear()
                    rand7 = get_random_clear()
                    rand8 = get_random_clear()
                    rand9 = get_random_clear()
                    rand10 = get_random_clear()
                    parentNode = nodes[0]
                    for p in nodes:
                        if dist(p.point,rand) <= dist(parentNode.point,rand) and dist(p.point,rand2) <= dist(parentNode.point,rand2) and dist(p.point,rand3) <= dist(parentNode.point,rand3) and dist(p.point,rand4) <= dist(parentNode.point,rand4) and dist(p.point,rand5) <= dist(parentNode.point,rand5) and dist(p.point,rand6) <= dist(parentNode.point,rand6) and dist(p.point,rand7) <= dist(parentNode.point,rand7) and dist(p.point,rand8) <= dist(parentNode.point,rand8) and dist(p.point,rand9) <= dist(parentNode.point,rand9) and dist(p.point,rand10) <= dist(parentNode.point,rand10):
                            newPoint = step_from_to(p.point,rand)
                            newPoint2 = step_from_to(p.point,rand2)
                            newPoint3 = step_from_to(p.point,rand3)
                            newPoint4 = step_from_to(p.point,rand4)
                            newPoint5 = step_from_to(p.point,rand5)
                            newPoint6 = step_from_to(p.point,rand6)
                            newPoint7 = step_from_to(p.point,rand7)
                            newPoint8 = step_from_to(p.point,rand8)
                            newPoint9 = step_from_to(p.point,rand9)
                            newPoint10 = step_from_to(p.point,rand10)
                            if collides(newPoint) == False and collides(newPoint2) == False and collides(newPoint3) == False and collides(newPoint4) == False and collides(newPoint5) == False and collides(newPoint6) == False and collides(newPoint7) == False and collides(newPoint8) == False and collides(newPoint9) == False and collides(newPoint10) == False:
                                parentNode = p
                                foundNext = True

                newnode = step_from_to(parentNode.point,rand)
                newnode2 = step_from_to(parentNode.point,rand2)
                newnode3 = step_from_to(parentNode.point,rand3)
                newnode4 = step_from_to(parentNode.point,rand4)
                newnode5 = step_from_to(parentNode.point,rand5)
                newnode6 = step_from_to(parentNode.point,rand6)
                newnode7 = step_from_to(parentNode.point,rand7)
                newnode8 = step_from_to(parentNode.point,rand8)
                newnode9 = step_from_to(parentNode.point,rand9)
                newnode10 = step_from_to(parentNode.point,rand10)
                nodes.append(Node(newnode, parentNode))
                nodes.append(Node(newnode2, parentNode))
                nodes.append(Node(newnode3, parentNode))
                nodes.append(Node(newnode4, parentNode))
                nodes.append(Node(newnode5, parentNode))
                nodes.append(Node(newnode6, parentNode))
                nodes.append(Node(newnode7, parentNode))
                nodes.append(Node(newnode8, parentNode))
                nodes.append(Node(newnode9, parentNode))
                nodes.append(Node(newnode10, parentNode))
                pygame.draw.line(screen,blue,parentNode.point,newnode)
                pygame.draw.line(screen,blue,parentNode.point,newnode2)
                pygame.draw.line(screen,blue,parentNode.point,newnode3)
                pygame.draw.line(screen,blue,parentNode.point,newnode4)
                pygame.draw.line(screen,blue,parentNode.point,newnode5)
                pygame.draw.line(screen,blue,parentNode.point,newnode6)
                pygame.draw.line(screen,blue,parentNode.point,newnode7)
                pygame.draw.line(screen,blue,parentNode.point,newnode8)
                pygame.draw.line(screen,blue,parentNode.point,newnode9)
                pygame.draw.line(screen,blue,parentNode.point,newnode10)

                if point_circle_collision(newnode, goalPoint.point, GOAL_RADIUS) or point_circle_collision(newnode2, goalPoint.point, GOAL_RADIUS) or point_circle_collision(newnode3, goalPoint.point, GOAL_RADIUS) or point_circle_collision(newnode4, goalPoint.point, GOAL_RADIUS) or point_circle_collision(newnode5, goalPoint.point, GOAL_RADIUS) or point_circle_collision(newnode6, goalPoint.point, GOAL_RADIUS) or point_circle_collision(newnode7, goalPoint.point, GOAL_RADIUS) or point_circle_collision(newnode8, goalPoint.point, GOAL_RADIUS) or point_circle_collision(newnode9, goalPoint.point, GOAL_RADIUS) or point_circle_collision(newnode10, goalPoint.point, GOAL_RADIUS):
                    currentState = 'goalFound'

                    goalNode = nodes[len(nodes)-1]

                
            else:
                print("Ran out of nodes... :(")
                return;

        #handle events
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                sys.exit("Exiting")
            if e.type == MOUSEBUTTONDOWN:
                print('mouse down')
                if currentState == 'init':
                    if initPoseSet == False:
                        nodes = []
                        if collides(e.pos) == False:
                            print('initiale point set: '+str(e.pos))

                            initialPoint = Node(e.pos, None)
                            nodes.append(initialPoint)
                            initPoseSet = True
                            pygame.draw.circle(screen, red, initialPoint.point, GOAL_RADIUS)
                    elif goalPoseSet == False:
                        print('goal point set: '+str(e.pos))
                        if collides(e.pos) == False:
                            goalPoint = Node(e.pos,None)
                            goalPoseSet = True
                            pygame.draw.circle(screen, blue, goalPoint.point, GOAL_RADIUS)
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




