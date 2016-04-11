Pendulum data:
- noisy_pendulum.csv: A comma separated file where each row correspond to a 2D 
coordinate of the noisy pendulum.

- true_pendulum.csv: A comma separated file where each row correspond to a 2D 
coordinate of the true noise free pendulum (ground truth).



Stand on right foot data:
This data comes from an articulated tracker based on depth camera data. It 
consists of 15 joints in 3D and for each joint there is a joint confidence 
indicating whether the tracker was tracking (confidence 1.0) or not tracking 
(confidence less than 1.0).

- stand_on_right_leg_Joints.csv: A comma separated file where each row correspond 
to the 3D coordinates of the 15 joints.

- stand_on_right_leg_JointConfidence.csv: A comma separated file where each row 
correspond to the confidence values of the 15 joints.

The 15 joints are ordered as columns in the two tables in the following way:

enum SkeletonJoints {
  HEAD_JOINT = 1,
  NECK_JOINT = 2,
  TORSO_JOINT = 3,
  LEFT_SHOULDER_JOINT = 4,
  LEFT_ELBOW_JOINT = 5,
  LEFT_HAND_JOINT = 6,
  LEFT_HIP_JOINT = 7,
  LEFT_KNEE_JOINT = 8,
  LEFT_FOOT_JOINT = 9,
  RIGHT_SHOULDER_JOINT = 10,
  RIGHT_ELBOW_JOINT = 11,
  RIGHT_HAND_JOINT = 12,
  RIGHT_HIP_JOINT = 13,
  RIGHT_KNEE_JOINT = 14,
  RIGHT_FOOT_JOINT = 15,
};

