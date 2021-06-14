# TODOs

## Week 16

- In the simulation experiment, all the actions and states are expressed in global frame (in real panda robot setup, base frame is used)
- The action in simulation including the setpoint position (not delta!!!), and the orientation is delta orientation.

## Week 15

- [x] Have a look at the code of [pddm](https://github.com/google-research/pddm), try to see how to implement a MPC or MPPI (model predictive path integral)
- [x] Fix (not working) [offscreen rendering](https://github.com/ARISE-Initiative/robosuite/issues/114#issuecomment-811295554).
- [x] Implemented random shooting algorithm, by just using the distance in z axis as cost.
- [x] Figure out why the action is not just in z direction
  - Reason: your eef z axis is not well aligned with the world z axis!!! and it would be flunctuate after it's moving!!!
- [x] Should we add constraint of force strength, to make it gentle?
- [x] For the cost, I use exponential average (discounting factor 0.9)
- [x] Train a mlp model match the 20Hz control frequency
- [x] Normalized distance difference, then calculate the cost
- [ ] Plot the force
- [ ] I don't think the MPPI is implemented 100% according to the formular, especially the filtering part!!!

## Week 14

- [x] Collect simulation data
- [x] Train model on the simulation data
- [x] Think about MPC
  - Evaluation metrics, what should be the cost / reward function $J$ / $R$
    - $J = \sum_{k=1}^{H} w_{pos} \cdot ||P^{success}_{pos} - P^{k}_{pos}||_2 + \sum_{k=1}^{H} w_{rot} \cdot distance(P^{success}_{rot} - P^{k}_{rot}) + \sum_{k=1}^{H} w_{force} \cdot ||P^{success}_{force} - P^{k}_{force}||_2$
    - For the insertion task, the reward function should be related to the inserted depth, right?
    - Use the negative loss to the ground truth of the demo?
    - Do I need to add a constraint to reforce the $\Delta A$ should be small (actions won't be large)
  - Randomly generate actions, what distribution should I sample these actions?
    - Uniform?
    - Gaussian and use the last action as the mean?
    - Visen-Halt distribution?
    - Trajectories following using the ground truth trajectories from the demos?
  - Have a look at the `rotation_distance` in the file `amira_ws/src/amira_tools/amira_utils/src/amira_utils/transformation_utils.py`
- [x] Write an simple summary of the method of **MPPI**
- [ ] Make the intial position of the TCP random
- ![MPC prelimilary ideas](img/week14.jpg)

## Week 13

- [x] Make spacemouse work with robosuite environment
- [x] Get the states and actions of robots:
  - state: "tcp_pose_base" (position and rotation), "tcp_wrench_base" (end-effector force)
  - action: "pose", "wrench" in base frame
- [x] Record data to `.h5` files
- [x] Compute the states in base frame, now is in the world coordinate, where the origin is the world coordinates.
  - Just use the `_hand_pos` and `_hand_quat` attributes from the robots
  - What is `eef_rot_offset` at the robot? Quaternion (x,y,z,w) representing rotational offset between the final robot arm link coordinate system and the end effector coordinate system (i.e: the gripper)
- [x] Figure out how to compute the action (set point from space mouse are in tcp frame) in base frame
  - [x] Take a look at `amira_ws/src/amira_hardware/amira_remote_control/src/amira_remote_control`, see how to incrementally add movement
  - This is quite simple, because the transformation from the world coordinate to the base frame is only translation
  - The problem is that the spacemouse output `dpos` and `drotation` (delta), if we accumulate these deltas, we can end up with absolute position in the base frame and the base orientation, but because the sampling rate of the spacemouse, this would not match the real state of the robot, in this case, the accumulated position and rotation are not set point anymore.
  - The other method would be adding the delta on top of the current state, which is really the 'set point'. However, during contact, this action would be end up the same as the state, which is not always the case in real life (imagine you have a barrier in front of you, then the set point is not your state)
  - Homogeneous transformation matrix P of end-effector in base frame is $^{B}_{E}P_{k}$. What I do is, in order to calculate the $$^{B}_{E}P_{k+1} = ^{B}_{E}P_{k}  *  _{\delta M}^{E}P_{k}$$

- [x] Add friction between the gripper and the peg or just weld it.
- ![space mouse and recording](img/week13_1.jpg)
- ![space mouse and recording](img/week13_2.jpg)

- Tricks used for the spacemouse:
  - For OSD, we need to use [this driver](http://spacenav.sourceforge.net/)
  - `lsusb` to find the vendor and product id
  - Start the spacenavd service via this command `sudo systemctl status spacenavd.service`
  - [Give the python-pid enough permission to read spacemouse usb without sudo](https://askubuntu.com/a/1150889/1021137)
  - Since the robosuite only support for the **Spacemouse Wireless**, we need to change the spacemouse readout to [this format](https://github.com/jwick1234/3d-mouse-rpi-python/blob/7ddfc9fb8703c84720eca5815bd993263c935c86/HelloSpaceNavigator.py#L46), in order to adapt to our **Spacemouse Compact**

## Week 12

- [x] Fix the $\Delta s$ of rotation.
  - Not just subtracting to get $\Delta s$ (because the rotation matrix is not at Euclidean space), but use the rotation transformation.
  - Previously: $R_{diff} = R_{t+1} - R_t$
  - Now: $R_{diff} = R_{t+1} * R_{t}^{-1}$
  - Results are much better than before!
- [x] Fix the Homogeneous transformation for data augmentation.
  - The disturbances should be bounded
- [x] Two options to build the simulator
  1. MuJoCo via Ros -> add PegInHole simulation
  2. MuJoCo via mujoco-py and robosuite
     - ROS -> mujoco-py
     - mujoco-py <- ROS (spacemouse)
     - python/mujoco implement impedance controller
  - ![Simulation](img/week12_3.jpg)
- What is the reward function?
  - negative L2 loss?
- Use ground truth dynamics to evaluate the algorithm first?

## Week 11

- [x] Fix the scaling issue at the loss, make sure they scale equally in every feature.
- [x] Try to match the prediction horizon in the paper keypoints to the future.
  - action command in the paper is at 5Hz, 2 second horizon, corresponds to 10 rollout prediction.
- [x] Fix the ugly euler angle scale in axis figure.
  - just clip the values to [-180, 180] 
- [x] Try to augment the data with randomly, not all
  - add noise on the input state and target residuals
- [x] Residual rotation matrix?

## Week 9 & 10

- [x] Learn the delta of the input (state and action)
  - To make the rotation matrice + residuals satisfy the orthonormal and determinant = 1 properties. I just make the output of the forwardpass - the input rotation matrix.
- [x] Use ensemble and MC-Dropout to generate multi-step predictions.
  - sum of the L2 loss and divided by the sample / ensemble number
  1. Use random ensemble / MC sample for each prediction
  2. Use mean of all ensemble / MC samples for each prediction
  3. Evaluate the performance of the rollout using the sum of L2 loss of the predicted state (pos, force and rot)
- [x] Add perterbation to the training data (data augmentation).
  - [x] [via zero mean Gaussian Noise](https://arxiv.org/pdf/1708.02596.pdf).
    - A zero mean Gaussian with 0.001 standard deviation, added before normalization.
    - Std scales with the data mean.
    - Rotation perturb on rotation matrix
  - [x] [via affine transformation](https://arxiv.org/pdf/2009.05085.pdf)
    - give each sample a random transform, the space become too large for the model and the limited data

## Week 8

- [x] Find a better representation of the orientation, here is a [good reference](https://datascience.stackexchange.com/questions/36370/how-to-learn-3d-orientations-reliably?newreg=2954130d00c34b45b3f34538eea02a1e)
  - [x] Learn both sine & cosine of euler angles.
  - [x] Learn the rotation matrix directly (similar to learning sine & cosine, but with 3 more varibles)
  - [x] Learn 6D representation based on this [paper](https://zhouyisjtu.github.io/project_rotation/rotation.html)
- [x] ~~Try to do something about MPC.~~
- [x] Check model (6D vs rotation matrix)
  - [x] Check the if understand and implement the 6D method correctly.
  - [x] Transform the 6D and sine_cosine back to euler angle to check the L2 distance.
- [x] Reproduce uncertainty results.
  - [x] More samples for MCDropout.
  - [ ] Use different subset of training data to train ensemble.
    - The bootstrapping technique is not recommended in the paper [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles by Balaji Lakshminarayanan et al., 2016](https://arxiv.org/abs/1612.01474v2).
    - They argue that using subset of data to train would hurt the performance of the NN, and to produce good result, we need NNs which are both good and diverse.
    - We also don't have enough data yet. Maybe we can do this experiment later.
    - The deep ensemble method I used is does not change the architecture of the NNs, and simply aggregate the results from 5 differently initialized NNs, taking the mean and variance of their predictions. However, in that paper, they suggest to do a Gaussian prediction (a mean and variance output for one value) for the regression tasks? Is it overcomplicated in our case or is it the right thing to try?
- [x] normalize for the cosine and sine output

![Week 8 todos](img/week8.png)


### Rotation experiment results

- Simple euler angles are unstable, when the angle goes from 180 to -180 and also singularity (gimbal lock), it will cause instability.
- Learning both sine and cosine doesn't have such problem, but it will increase the dimensionality.
  - Results are more stable than just learning the Euler vector, but some values exceeded 1 (normalization need to be done).
  - added tahn activation or clamp only for the rotation values.
  - Some problem I have during training with sine and cosine:
    - $s^2 + c^2 = 1$, How to limit the range for sine & cosine while enforcing this constraint?
    - When without these constraints, the model fit the training set very well with sine and cosine (**normalized**), but still have large errors in the test set.
    - I tried tanh and clamp to enforce cosine sine in the range [-1, 1] (**without normalization**), results are pretty bad. Without normalization the distribution of these values are similar to axis angle, some are cluster around only -1 or 1, missing the values inbetween.
    - **Tanh activation** can make the value around -1 and 1 especially hard to learn (vanishing gradients) and causing large errors.
    - **Clamp** would make the gradient beyond range [-1, 1] be zero, leading to not learning at those points.
- The 6D representation yield least loss at the test set.
  - 6D perform descently in both training and test set. Only some fluctuation compared to cosine and sine. But using 6D I don't need to worry about the constraint, because the outputs are already guaranteed to be a valid rotation matrix (orthonormal for each vector)
- 9D or full matrix is similar to 6D, but still no constraints applied (not a valid rotation matrix) and need more parameters(more output neurons).
  - valid rotation matrix should be:
    1. Orthogonal
    2. Determinat = 1

### Questions

- Since the states are recorded in 10Hz, the rotation shouldn't be large? Is it also okay to use euler angle (cosine and sine), because it's unlikily to have a large magnitude of rotation?
- However, the rotation is using the base frame as reference! So it will still be possibilty to encounter instability.

## Week 7

- Normalization is done by concatenating all demos into one and then take the mean and standard deviation over time.
- [x] Try to map the inputs value to the range of $[-1, 1]$.
- [x] See if the normalization introduce some numerical issues, since the predictions for $rot_x$ are bad.
  - I find out that the rotx values in demos cluster around 2 extrem values, one is the super large (around +3), the other one is extremly small (around -3).
  - I tried to exclude the demos with super large one, turns out the regressin results become much more better
- Insertion direction of the data:
  - 03, 16, from right, large value of rotx.
  - 14, 17, from back right, first large postive then sudden change to negative value of rotx.
  - 01, 02, 04, 05, 06, 09, 20, from left, small value (negative) of rotx.
  - 15, 18, from front left, first negative then sudden change to positive value of rotx.
  - 07, from back, first large postive then sudden change to negative value of rotx.
  - 19, from back, first negative then large positive.
