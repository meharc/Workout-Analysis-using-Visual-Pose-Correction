# import libraries
# python
import cv2

# preprocess the data— scale, subtract mean, swap channels
def preProcessing(data):
    
    processedData = cv2.dnn.blobFromImage(data, scalefactor= 1.0/255, size = (368,368), mean = (0,0,0), swapRB=False, crop = False)

    return processedData