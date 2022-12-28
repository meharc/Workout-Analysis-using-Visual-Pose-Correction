# Workout-Analysis-using-Visual-Pose-Correction

There is often a higher risk of injuries anticipated when working out at home with the help of a plethora of unverified online videos compared to real-time professional supervision. A 48% increase in emergency room visits was recorded in the United States due to home exercise injuries from 2019 to 2020 [16]. It has become imperative to create resources that can assist people during their home workout sessions, coming close to providing real-time professional guidance. The project aims to create a human pose estimation and workout analyzer that overlays the skeleton of a professional over the user performing a specific exercise in real-time, thus, reducing long-term injuries.

The project pipeline is inspired from Realtime Indoor Workout Analysis Using Machine Learning & Computer Vision and uses an amalgamation of techniques — 2D pose estimation, optical flow, DTW (Dynamic Time Warping), and affine transformation. It aims to build upon these techniques to provide a more computationally efficient workout analysis, as illustrated in figure below. The project will utilize the workout video of a trainer performing an exercise repetition(s) and the user — whose exercise form might need correction.

The project is circumscribed by the following limits:

- Single user per frame.
- Exercises performed in vertical orientation.
- User Position — stand at a minimum distance from the computer to capture the entire user body and the corresponding limbs.

Prior to running the script make sure you have the following packages installed on the machine — pickle, tqdm, cv2. To run the script, follow the steps below:

- Run main.py
- Given the prompt with a list of exercise names, enter your choice of exercise.
- Automatic video feed will begin.
– Ensure your whole body is in front of the camera to obtain the correct pose estimation. – Ensure you are performing the exercise in a vertical standing position.
– Ensure there is only one user in the frame.
- To quit the project, press ‘q’

Note: To extend the project for new training exercises, ensure the constraints mentioned above are followed and the metadata.csv file contains the training video information.

The image below exemplify the results for real time workout analysis.
![](https://github.com/meharc/Workout-Analysis-using-Visual-Pose-Correction/blob/main/result.png =100x100)

