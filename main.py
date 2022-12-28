# import libraries
# python
import cv2
import numpy as np
import os
import pickle
import pandas as pd
# custom
from getTrainerPose import getTrainerPose
from preProcessing import preProcessing
from poseEstimation import poseEstimation
from calculateLimbPairAngles import calculateLimbPairAngles
from matchFrames import matchFrames
from affineTranformation import affineTransformation

# change to previous directory
os.chdir("./..")

# main function handles the following tasks:
# 1. Gets exercise name from the user.
# 2. Creates trainer key points and trainer limb pair angles, if doesn't exists
# 3. For every current captured frame of the user (using camera), it calculates the user's 
# key points and limb pair angles. This is followed by matching the user frame with the
# appropriate trainer video and frame using a custom metric. Finally, the selected trainer
# frame is overlayed upon the current user frame to provide a visual pose correction.
def main():

    # define parameters
    exerciseList = np.array(["biceps curl","deadlift", "tricep pushdown"]) # ---- taken from metadata.csv
    trainerKeyPointsPath = './output/trainerKeyPoints.txt'
    trainerLimbPairAnglesPath = './output/trainerLimbPairAngles.txt'
    metadataPath = './data/metadata.csv'
    kFrame = 5
    matchFramesMethod = 'default'

    ##########################################################################################

    # Step 1: Prompt the user to enter the exercise.
    print("Welcome! Let's help you get started with your workout! Exercise list: ")
    print('\n'.join(exerciseList))
    exerciseName = input("Please type your exercise using the list above.")

    # check if exercise is in the list
    while exerciseName.lower() not in exerciseList:
        print('Exercise not in the list! Try again!')
        exerciseName = input("Please type your exercise using the list above.")

    print("You have chosen: " + exerciseName)
     
    ##########################################################################################

    # Step 2: Check if key points for trainer videos exists! 

    # if not, then find the key points for the trainer and store them, else move ahead with user live feed.
    if not os.path.exists(trainerKeyPointsPath):
        trainerKeyPoints, trainerLimbPairAngles = getTrainerPose(kFrame)

        # save trainer key points
        file = open(trainerKeyPointsPath, 'wb')
        pickle.dump(trainerKeyPoints, file)
        file.close()

        # save limb pair angles
        file = open(trainerLimbPairAnglesPath, 'wb')
        pickle.dump(trainerLimbPairAngles, file)
        file.close()

    # load trainer key points and limb pair angles
    with open(trainerKeyPointsPath, 'rb') as f:
        trainerKeyPoints = pickle.load(f)
    with open(trainerLimbPairAnglesPath, 'rb') as f:
        trainerLimbPairAngles = pickle.load(f)

  
    ##########################################################################################
    
    # Step 3: Start the live feed.

    print('Capturing live feed! Press \'q\' to exit.')

    # select training video based on user input.
    metadata = pd.read_csv(metadataPath)
    keyVideo = metadata[metadata.exerciseName == exerciseName]['location']

    # create video capture object
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam!")

    # Specify the paths for caffe files used to build the model. 
    protoFile = "./model/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "./model/pose/mpi/pose_iter_160000.caffemodel"

    # Read the network into Memory 
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # Capture frame-by-frame
    while (cap.isOpened()):

        ret, frame = cap.read() # ret is a Boolean value that tells whether the frame was successfully captured. If it is, then it is stored in frame

        # if frame is read correctly ret is True
        if ret==False:
            break
        
        # preprocess the frame
        processedFrame = preProcessing(frame) # processed frame is a blob (4D) that is taken as input by the caffe architecture in pose estimation.
        
        # call poseEstimation to find key points
        keyPoints, poseImg = poseEstimation(frame, processedFrame, net)

        # find the matching training frame for the current user frame.
        # find limb pair angles for current frame of the user.
        userLimbPairAngles = np.expand_dims(calculateLimbPairAngles(keyPoints), axis = 0)
        # call matchFrames to find matching frame
        [chosenFrame, cost] = matchFrames(trainerLimbPairAngles[keyVideo[keyVideo.index[0]]],userLimbPairAngles, matchFramesMethod)

        # perform affine transformation to overlay the trainer skeleton from the chosen from onto the
        # user skeleton.
        # select key points from trainer video corresponding to the chosen frame

        source = trainerKeyPoints[keyVideo[keyVideo.index[0]]][chosenFrame,:].astype(np.float32)
        destination = keyPoints.astype(np.float32)
        # apply affine transformation
        resultFrame = affineTransformation(source, destination, frame)

        # uncomment the below code to display frame.
        cv2.imshow("Affine transformation", resultFrame)

        # don't comment the below line!
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

    return 0

# call main 
if __name__== '__main__':
    main()