# Model Learning

## MLP based

### Training

State action pair ($s_t$, $a_t$) as input, predicts $\hat{s}_{t+1}$, $s_{t+1}$ as target.

- State: pose position of arm
- Action: pose position of controller

### Network

Multi-layer perceptrons, two hidden layers with 500 neurons per layer.

### Loss funtion

$$\mathcal{L}_{dynamics} = \sum_{h=1}^{H} \left\lVert \hat{z}_{t+h} - z_{t+h}\right\rVert ^2 _2, ~\hat{z}_{t+h} = \hat{f}_{\theta_{dyn}}(\hat{z}_{t-l:t, a_{t+h}}), \hat{z}_t = z_{t}$$

- L2 loss, minimize the dynamics prediction error over the horizon H (H=1 for now).



