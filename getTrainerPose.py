# import libraries
# python
import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
# custom
from preProcessing import preProcessing
from poseEstimation import poseEstimation
from sparseOpticalFlow import sparseOpticalFlow
from calculateLimbPairAngles import calculateLimbPairAngles

def getTrainerPose(kFrame):

    # read metadata file to read video locations
    metadataPath = './data/metadata.csv'
    metadata = pd.read_csv(metadataPath)

    # Specify the paths for caffe files used to build the model. 
    protoFile = "./model/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "./model/pose/mpi/pose_iter_160000.caffemodel"

    # Read the network into Memory 
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # define dictionary to store key points for all trainer videos and the corresponding limb pair angles
    trainerKeyPoints = dict()
    trainerLimbPairAngles = dict()

    # defining parameters for Lukas kanade optical flow
    lk_params = dict(
        winSize=(175,175),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # for every video, find key locations using pose estimation and optical flow
    for i in tqdm(range(metadata.shape[0])):

        filePath = metadata['location'][i]
        videoPath = os.path.join('./data',filePath)

        # create video capture object
        cap = cv2.VideoCapture(videoPath)
        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if video object is opened
        if not cap.isOpened():
            raise IOError("Cannot open webcam!")

        # create a list element to store key points for every video
        trainerKeyPoints[filePath] = list()
        trainerLimbPairAngles[filePath] = list()

        # create variables to keep track of points that needs to be tracked using sparse optical flow.
        pointsToTrack = np.array([])
        oldFrame = np.array([])

        # Create a mask image for drawing purposes
        mask = np.array([])

        # Capture frame-by-frame
        while(cap.isOpened()):

            ret, frame = cap.read() # ret is a Boolean value that tells whether the frame was successfully captured. If it is, then it is stored in frame
        
             #if frame is read correctly ret is True
            if ret==False:
                break

            # get frame number (starts from 1)
            frameNum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Uncomment the below to print frame numbers for each video.
            """print('Frame {}, Total {}'.format(frameNum, totalFrames))"""
            
            if (frameNum%kFrame==0) | (frameNum==1):

                if frameNum ==1:
                    mask = np.zeros_like(frame)

                # preprocess the frame
                processedFrame = preProcessing(frame) # processed frame is a blob (4D) that is taken as input by the caffe architecture in pose estimation.

                # call poseEstimation to find key points
                keyPoints, poseImg = poseEstimation(frame, processedFrame, net)
                
            # apply optical flow to get key points for non-kth frame
            else:
                keyPoints, img, mask = sparseOpticalFlow(oldFrame, frame, pointsToTrack, lk_params, mask)
                
                # uncomment the below to display frame. In order to display the entire video, uncomment the line for wait key and escape as well.
                """cv2.imshow("sparse optical flow", img)"""

            # find limb pair angle for the frame based on the key points. For a (15,2) key points, we get a 1d array of 10 limb pair angles.
            limbPairAngles= calculateLimbPairAngles(keyPoints)
            trainerLimbPairAngles[filePath].append(limbPairAngles)

            # keep track of current frame and points to use for next frame(if not kth frame)
            pointsToTrack = keyPoints.copy()
            oldFrame = frame.copy()

            # append key points for the specific frame and video
            trainerKeyPoints[filePath].append(keyPoints)

            # Uncomment to display optical flow on the video
            """if cv2.waitKey(1) & 0xFF == ord('q'):
                break"""
            
        # convert list into 3d numpy array (2D key points for each frame in the video)
        trainerKeyPoints[filePath] = np.asarray(trainerKeyPoints[filePath])

        # convert list into 2d numpy array (1D limb pair angles for each frame in the video)
        trainerLimbPairAngles[filePath] = np.asarray(trainerLimbPairAngles[filePath])

        cv2.destroyAllWindows()
        cap.release()

    return trainerKeyPoints, trainerLimbPairAngles