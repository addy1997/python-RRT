#code for grasping object
#created on 16th October, 13:04:52


# import libs

###################################
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
###################################



# get verbose boolean

verbose = sys.argv[1]

#with verbose
# python grasp_demo.py 1

#without verbose
# python grasp_demo.py 0


# if verbose=True then show info after each step
# e.g "step 1: successful" or "step 1: failed"

######################################################3




# moveit commander is a class that allow us to communicate with a move group
moveit_commander.roscpp_initialize(sys.argv)

# create ros node
rospy.init_node("move_group_python_interface_tutorial",anonymous=True)

# we are getting the robot that this moveit commander is going to command to send to receive commands
robot = moveit_commander.RobotCommander()


try:
    ################ START STEP 1 ################
    ##############################################


    # "Send the arm to an inital known position" #
    #---------------------------------------------#

    # Desc: In order to move the arm, we are going to use the groups
    # that we created: robot(or arm) and gripper

    # This code, creates an instance of the arm group, and then indicates that,
    # for that group, Moveit has to set the group into the start position


    arm_group = moveit_commander.MoveGroupCommander("arm") #give name of the robot as parameter (not the gripper)

    # put the robot(not the gripper) in the start position
    arm_group.set_named_target("start") # pose may be differ

    plan1 = arm_group.go()


    ################ END OF STEP 1 ################
    ###############################################
    if verbose:
        print "step 1: passed"

except:
    if verbose:
        print "step 1: failed"


try:

    ################ START STEP 2 ################
    ##############################################


    # "Open the gripper" #
    #---------------------------------------------#

    # Desc: Add a new group to control the gripper using one of the defined positions for it (open)

    hand_group = moveit_commander.MoveGroupCommander("gripper") #name may differ

    # Use second type of command for move_group (send a complete pose we defined during configuration)

    #O PEN the gripper
    hand_group.set_named_target("open") #check exact command name
    plan2 = hand_group.go()

    ################ END OF STEP 2 ################
    ###############################################
    if verbose:
        print "step 2: passed"

except:
    if verbose:
        print "step 2: failed"



try:
    ################ START STEP 3 ################
    ##############################################

    # "Bring the arm close to object" #
    #---------------------------------------------#

    # Desc: Provide to the moveit_commander the exact coordinates we want it
    # to put the end effector, based on the /world frame

    # Use second type of command for move_group (send position for the end effector)


    # put the arm at the 1st grasping position
    pose_target = geometry_msgs.msg.Pose()
    pose_target.orientation.w = 0.5
    pose_target.orientation.x = 0.5
    pose_target.orientation.y = 0.5
    pose_target.orientation.z = 0.5

    pose_target.position.x = 0.15
    pose_target.position.y = 0.0
    pose_target.position.z = 1.25

    arm_group.set_pose.target(pose_target)
    plan1 = arm_group.go()

    ################ END OF STEP 3 ################
    ###############################################
    if verbose:
        print "step 3: passed"


except:
    if verbose:
        print "step 3: failed"

try:
    ################ START STEP 4 ################
    ##############################################

    # "Get the gripper closer" #
    #---------------------------------------------#

    # Desc: put the robot at the 2nd grasping position

    pose_target.position.z = 1.125 #assume gripper head looking down, move downward to grasp

    arm_group.set_pose_target(pose_target)
    plan1 = arm_group.go()

    ################ END OF STEP 4 ################
    ###############################################

    if verbose:
        print "step 4: passed"

except:
    if verbose:
        print "step 4: failed"


try:
    ################ START STEP 5 ################
    ##############################################

    # "Close the gripper" #
    #---------------------------------------------#

    # Desc: close the gripper

    hand_group.set_named_target("close") #target name may differ
    plan2 = hand_group.go()

    ################ END OF STEP 5 ################
    ###############################################

    if verbose:
        print "step 5: passed"

except:
    if verbose:
        print "step 5: failed"

try:
    ################ START STEP 6 ################
    ##############################################

    # "Lift the object" #
    #---------------------------------------------#

    # Desc: put the arm at the 3rd grasping position

    pose_target.position.z = 1.5
    arm_group.set_pose_Target(pose_target)
    plan1 = arm_group.go()


    ################ END OF STEP 6 ################
    ###############################################

    if verbose:
        print "step 6: passed"

except:
    if verbose:
        print "step 6: failed"



# Need to finish the code with a shutdown of the commander,
# in order to close properly the code
rospy.sleep(5)
moveit_commander.roscpp_shutdown()
