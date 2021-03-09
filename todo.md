# TODOs

## Week 7

- normalization is done by concatenating all demos into one and then take the mean and standard deviation over time
- [x] try to map the inputs value to the range of $[-1, 1]$
- [ ] see if the normalization introduce some numerical issues, since the predictions for $rot_x$ are bad.
  - I find out that the rotx values in demos cluster around 2 extrem values, one is the super large (around +3), the other one is extremly small (around -3).
  - I tried to exclude the demos with super large one, turns out the regressin results become much more better
- insertion direction of the data:
  - 03, 16, from right, large value of rotx
  - 14, 17, from back right, first large postive then sudden change to negative value of rotx
  - 01, 02, 04, 05, 06, 09, 20, from left, small value (negative) of rotx
  - 15, 18, from front left, first negative then sudden change to positive value of rotx
  - 07, from back, first large postive then sudden change to negative value of rotx
  - 19, from back, first negative then large positive
