# TODOs

## Week 9

- [ ] Learn the delta of the input (state and action)
  - it would be very tricky for the rotation, because we need to transform the rotation later.
  - Is it really necessary, one trick we can do is to learn the residual except for the rotation, but this also make things complicated.
  - [ ] have a look at the code in this [paper](https://arxiv.org/pdf/1708.02596.pdf), see how they deal with learning the residual while make things remain simple.
- [ ] Use ensemble and MC-Dropout to generate multi-step predictions.
  1. Use random ensemble / MC sample for each prediction
  2. Use mean of all ensemble / MC samples for each prediction
- [ ] Add perterbation to the training data (data augmentation).
  1. [via zero mean Gaussian Noise](https://arxiv.org/pdf/1708.02596.pdf).
  2. [via affine transformation](https://arxiv.org/pdf/2009.05085.pdf)

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
