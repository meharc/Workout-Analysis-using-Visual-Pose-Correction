# import libraries
# python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# estimate key points for a frame based on multi-person pose estimation model.
def poseEstimation(frame, blob, net):

    BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                   "Background": 15 }

    POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                   ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                   ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Prepare Input to the Network, set frame as input
    net.setInput(blob)

    # Make Predictions 
    output = net.forward() # output is 4D, where mpi give for every image, 44 results.
    #The output is a 4D matrix :
    # The first dimension being the image ID ( in case you pass more than one image to the network ).
    #The second dimension indicates the index of a keypoint. The model produces Confidence Maps and Part Affinity maps which are all concatenated.
    #For MPI, it produces 44 points. We will be using only the first few points which correspond to Keypoints.
    #The third dimension is the height of the output map.
    #The fourth dimension is the width of the output map.

    # check if each keypoint is present in the frame.  We get the location of the keypoint by finding 
    # the maxima of the confidence map of that keypoint. We also use a threshold to reduce false detections.
    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []
    
    for i in range(len(BODY_PARTS)-1):  # removing background 
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :] # confidence map

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > 0 : # threshold
            # draw them on top of image
            """cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), thickness = -1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 2, lineType=cv2.LINE_AA)"""

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # uncomment the below code to draw pose on frame using key points
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
    
    return np.asarray(points), frame
