# TODOs

## Week 8

- [x] find a better representation of the orientation, here is a [good reference](https://datascience.stackexchange.com/questions/36370/how-to-learn-3d-orientations-reliably?newreg=2954130d00c34b45b3f34538eea02a1e)
  - [x] learn both sine & cosine of euler angles.
    - simple euler angles are unstable, when the angle goes from 180 to -180, it will cause instability.
    - learning both sine and cosine doesn't have such problem, but it will increase the dimensionality.
    - results are more stable than just learning the Euler vector, but some values exceeded 1(normalization need to be done).
      - add tahn activation only for the rotation values 
  - [x] learn the rotation matrix directly (similar to learning sine & cosine, but with 3 more varibles)
    - results
  - [x] learn 6D representation
- results, the 6D representation yield least loss at the test set.
  - some problem I have during training with sine and cosine:
    - how to limit the range for sine & cosine while enforce s^2 + c^2 = 1? When without these constraints, the model fit the training set very well with sine and cosine (**normalized**), but still have large errors in the test set.
    - I tried tanh and clamp to enforce cosine sine in the range [-1, 1] (**without normalization**), results are pretty bad. Without normalization the distribution of these values are similar to axis angle, some are cluster around only -1 or 1, missing the values inbetween.
      - tanh makes the value around -1 and 1 especially hard to learn (vanishing gradients) and causing large errors.
  - 6D perform descent in both training and test set. Only some fluctuation compared to cosine and sine. But using 6D I don't need to worry about the constraint, because the outputs are already guaranteed to be a valid rotation matrix (orthonormal for each vector)
  - 9D or full matrix is similar to 6D, but still no constraints applied (not a valid rotation matrix) and need more parameters(more output neurons).
- [ ] try to do something about MPC
- [ ] More samples for MCDropout
- [ ] Use different subset of training data to train ensemble

## Week 7

- normalization is done by concatenating all demos into one and then take the mean and standard deviation over time.
- [x] try to map the inputs value to the range of $[-1, 1]$.
- [x] see if the normalization introduce some numerical issues, since the predictions for $rot_x$ are bad.
  - I find out that the rotx values in demos cluster around 2 extrem values, one is the super large (around +3), the other one is extremly small (around -3).
  - I tried to exclude the demos with super large one, turns out the regressin results become much more better
- insertion direction of the data:
  - 03, 16, from right, large value of rotx.
  - 14, 17, from back right, first large postive then sudden change to negative value of rotx.
  - 01, 02, 04, 05, 06, 09, 20, from left, small value (negative) of rotx.
  - 15, 18, from front left, first negative then sudden change to positive value of rotx.
  - 07, from back, first large postive then sudden change to negative value of rotx.
  - 19, from back, first negative then large positive.
