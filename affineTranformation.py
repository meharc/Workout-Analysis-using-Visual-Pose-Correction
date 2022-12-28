# import libraries
# python
import cv2
import numpy as np

# function to perform affine transformation on source points to the destination.
# Furthermore, it displays the transformed trainer skelton ove user skeleton
def affineTransformation(source, destination, frame):

    # select 3 points from source and the corresponding points in the destination
    ptsSource = source[:3,:]
    ptsDestination = destination[:3,:]

    # find the warping matrix
    warp_mat = cv2.getAffineTransform(ptsSource, ptsDestination)

    # convert source points from 15x2 to 15x3 by appending 1's.
    onesArray =  np.expand_dims(np.ones(source.shape[0]), axis =1 )
    X = np.concatenate((source, onesArray), axis = 1)

    # transform by X to Y using the following equation : AX = Y, where Y is transformed points (15,2)
    Y = np.matmul(warp_mat, X.T).T

    #################################################################

    BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                    "Background": 15 }

    POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                    ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                    ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if Y[idFrom].all() and Y[idTo].all() and destination[idFrom].all() and destination[idTo].all():
            cv2.line(frame, tuple(destination[idFrom].astype(np.int32)), tuple(destination[idTo].astype(np.int32)), (0, 255, 0), 3)
            cv2.line(frame, tuple(Y[idFrom].astype(np.int32)), tuple(Y[idTo].astype(np.int32)), (255, 0, 0), 3)

    return frame



