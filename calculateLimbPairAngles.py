# import libraries
# python
import numpy as np

# calculate limb pair angles (1D numpy array with 10 elements) for every frame
def calculateLimbPairAngles(keyPoints):

    BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                   "Background": 15 }

    LIMB_PAIRS = [["Head", "Neck", "RShoulder"], ["Head", "Neck", "LShoulder"], 
                    ["Neck", "RShoulder", "RElbow"], ["Neck", "LShoulder", "LElbow"],
                    ["RShoulder", "RElbow", "RWrist"],["LShoulder", "LElbow", "LWrist"],
                    ["Neck", "RHip", "RKnee"],  ["Neck", "LHip", "LKnee"], 
                    ["RHip", "RKnee", "RAnkle"], ["LHip", "LKnee", "LAnkle"]]

    # define empty array to store angles for every limb pair mentioned above.
    limbPairAngles = np.array([])

    # for every limb pair, check if the key points of the limb pair exists and then calculate the angle between them.
    for pair in LIMB_PAIRS:
     
        partStart = pair[0]
        partCenter = pair[1]
        partEnd = pair[2]
        assert(partStart in BODY_PARTS)
        assert(partCenter in BODY_PARTS)
        assert(partEnd in BODY_PARTS)

        idStart = BODY_PARTS[partStart]
        idCenter = BODY_PARTS[partCenter]
        idEnd = BODY_PARTS[partEnd]

        line1 = keyPoints[idStart] - keyPoints[idCenter]
        line2 = keyPoints[idEnd] - keyPoints[idCenter]

        cosine_angle = np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2)) #-----?
        angle = np.degrees(np.arccos(cosine_angle)) #-----? should it be degrees or radian
        limbPairAngles = np.append(limbPairAngles,cosine_angle)

    return limbPairAngles