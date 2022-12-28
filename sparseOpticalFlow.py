# import libraries
# python
import cv2

# implement sparse optical flow using lucas kanade with pyramids
def sparseOpticalFlow(oldFrame, currentFrame, points, parameters, mask):

    # convert frames to gray scale
    oldGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
    currentGray = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)

    # converting points to (n,1,2) and float 32
    points = points.reshape(-1,1, 2).astype(dtype = 'float32')

    # Calculate Optical Flow
    nextPoints, status, error = cv2.calcOpticalFlowPyrLK(oldGray, currentGray, points, None, **parameters)

    # Select good points
    good_new = nextPoints[status == 1]
    good_old = points[status == 1]

    # check if tracked points are of size (15,2)
    assert (good_new.shape == (15,2)), "Optical flow is not able to track all points!"

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(dtype = 'int')
        c, d = old.ravel().astype(dtype = 'int')
        mask = cv2.line(mask, (a, b), (c, d), (0,0,255), 2)
        currentFrame = cv2.circle(currentFrame, (a, b), 10, (0,0, 255), -1)

    # Display the demo
    img = cv2.add(currentFrame, mask)

    return good_new, img, mask