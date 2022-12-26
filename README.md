# Workout-Analysis-using-Visual-Pose-Correction

There is often a higher risk of injuries anticipated when working out at home with the help of a plethora of unverified online videos compared to real-time professional supervision. A 48% increase in emergency room visits was recorded in the United States due to home exercise injuries from 2019 to 2020 [16]. It has become imperative to create resources that can assist people during their home workout sessions, coming close to providing real-time professional guidance. The project aims to create a human pose estimation and workout analyzer that overlays the skeleton of a professional over the user performing a specific exercise in real-time, thus, reducing long-term injuries.

The project pipeline is inspired from Realtime Indoor Workout Analysis Using Machine Learning & Computer Vision [1] and uses an amalgamation of techniques — 2D pose estimation, optical flow, DTW (Dynamic Time Warping), and affine transformation. It aims to build upon these techniques to provide a more computationally efficient workout analysis, as illustrated in figure below. The project will utilize the workout video of a trainer performing an exercise repetition(s) and the user — whose exercise form might need correction.

The project is circumscribed by the following limits:

- Single user per frame.

- Exercises performed in vertical orientation.

- User Position — stand at a minimum distance from the computer to capture the entire user body and the corresponding limbs.

