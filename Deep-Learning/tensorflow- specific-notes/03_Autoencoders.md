# Working of Autoencoders

### Autoencoders

- Type of neural network architecture used for **_unsupervised learning_**.
- Aimed at compressing input data into a smaller latent representation and then reconstructing it back as closely as possible to the original input.
- An autoencoder can learn non-linear transformations with a non-linear activation function and multiple layers.
- It doesn’t have to learn dense layers. It can use convolutional layers to learn which is better for video, image and series data.
- It can make use of pre-trained layers from another model to apply transfer learning to enhance the encoder/decoder.
- Components:
  - **Encoder:** (recognition network) that converts the inputs to an internal representation. Uses **ReLu** or **Sigmoid**.
  - **Decoder:** (generation network) that converts the internal representation to the
    outputs. Uses **Sigmoid** or **TanH**.
  - **Latent Space:**
    - Stores the compact representation (the "summary").
    - Smallest and densest layer (**Bottleneck**).
    - Smaller latent space leads to greater compression but risks losing important details.
- Hyperparameters:
  - Code Size (number of nodes in middle layer)
  - Number of layers
  - Number of nodes per layer
  - Loss function
- Loss function:
  - Mean Squared Error (MSE): For numeric data, it measures the squared difference between the original and reconstructed inputs.
  - Binary Cross-Entropy: For binary or normalized data (e.g., pixel values between 0 and 1).
- Training Process: Forward Propagation -> Calculating Loss -> Backward Propagation -> Optimization.
- Properties of Auto-encoders: Data-specific, Lossy, Unsupervised (or self-supervised).

<hr>

### Types of Autoencoders

- **Undercomplete Autoencoders:**
  - Latent space dimension is smaller than input dimension.
  - Used for dimensionality reduction.
  - Captures most important features.
  - Advantages: No need for regularization, dimensionality reduction, feature learning, reconstruction and completion.
- **Overcomplete Autoencoders:**
  - Latent space dimension is larger than input dimension.
  - Encoder and Decoder have more hidden units than input layer.
  - Has more capacity to learn.
  - Risk of overfitting.
  - Advantages: Feature enrichment, Learn detailed representation, Improved reconstruction, Model complex relationships.

<hr>

### Convolutional Autoencoders

- Compression is achieved by applying convolutional layers to extract features from the input image and **downsampling** (reducing the resolution or size of data) them to reduce the dimensionality.
- Decoding process is just the reverse of this, where **upsampling** (increasing the resolution or size of data) layers are used to increase the dimensionality and convolutions are applied to reconstruct the image.
- Structure: Input Layer -> Encoder (Convolution 1, 2, 3) -> Flatten and Fully Connected (Hidden Layer) -> Decoder (Reshape -> DeConv 1 and 2) -> Output

<hr>

### Linear Autoencoders vs PCA

- What is PCA?
  - Linear transformation that finds the directions of maximum variance in the data, and projects the data onto a lower-dimensional space.
  - These directions are called **_principal components_**, and they are orthogonal to each other.
  - Way of compressing the data by discarding the components that have low variance and retain the most important ones.
  - Used for both supervised and unsupervised learning.
- When **Linear Autoencoder** is equivalent to PCA?
  - must use linear activation functions in both encoder and decoder.
  - both minimize the same reconstruction error (**_MSE_**).
  - latent dimensionality matches the number of principal components retained.
  - autoencoder’s weights align with the orthogonal subspace defined by PCA's eigenvectors.
- Differences:
  - PCA is a deterministic and linear method, while autoencoders are stochastic and nonlinear.
  - PCA always gives the same result for the same data, and it can only capture linear relationships.
  - Autoencoders, however, can give different results depending on the random initialization and optimization process, and they can capture nonlinear relationships.
  - PCA is a parametric method whereas autoencoders are non-parmetric.
  - PCA is global (whole dataset), autoencoders are local.

<hr>

### Variational Autoencoders

- Describe an observation in latent space in a probabilistic, rather than deterministic, manner.
- VAE outputs a **probability distribution** for each latent attribute, rather than a single value.
- Useful when we don't want to exactly replicate the input as output (like in standard autoencoder).
- Key Difference:
  - Normal Autoencoder: Compresses input into a fixed latent representation (a single point in space).
  - VAE: Learns a distribution (a range of possible values) in the latent space, adding randomness.
- Example: Imagine compressing images of handwritten digits:
  - A normal autoencoder maps each digit to a fixed point. If you interpolate between two digits, you might get blurry images.
  - A VAE ensures that similar digits have **overlapping distributions**. This allows smooth transitions between digits, making it great for generating new realistic digits.
- It is assumed that the distributions from each latent feature are Gaussian.
- As such, two vectors are output where one describes the mean and the other describes the variance of the distributions.
- Workflow:
  - **Probabilistic Encoder:**
    - Encode the data into a distribution.
    - Outputs: Mean (centre point), Standard deviation (spread), together form **Gaussian Distribution** [Z ~ N($\mu$, $\sigma$^2)].
    - **_Reparameterization Trick_**: To backpropagate through stochastic sampling, the latent vector Z is computed as: Z = $\mu$ + $\sigma$.$\epsilon$.
    - $\epsilon$ (0,1) is **random noise** sampled from a standard normal distribution, making the process differentiable for training.
  - **Latent Space (Z):**
    - Latent space is structured probabilistically.
    - Enabling the VAE to sample meaningful variations of the data by generating Z from the learned distribution.
  - **Probabilistic Decoder:**
    - Decoder reconstructs the input X from the latent variable Z:
    - Tries to model the probability of the original input X given Z.
    - Allows the VAE to generate new data points by sampling Z from the prior distribution.
- Loss functions:
  - **_Reconstruction Loss:_**
    - Measures how well the VAE reconstructs the input data.
    - For numerical data use MSE.
    - For Binary Data use Binary Cross Entropy.
  - **_KL Divergent Loss:_**
    - Regularizes the latent space by ensuring the learned latent distribution Q(Z∣X) is close to the prior distribution P(Z) (typically a standard gaussian).
    - The latent space is well-structured and facilitates sampling from the learned latent distribution.
    - $KL(q(z|x) \parallel p(z)) = \sum q(z|x) \log \frac{q(z|x)}{p(z)}$
    - $KL(q(z|x) \parallel p(z)) = \frac{1}{2} \sum \left( 1 + \log \sigma^2 - \mu^2 - \sigma^2 \right)$
  - Extra: **Evidence lower Bound:** $\mathcal{L} = \mathbb{E}_{q(z|x)} [\log p(x|z)] - KL(q(z|x) \parallel p(z))$
- Why use Probabilistic Encoder?
  - Learn a continuous, smooth latent space and ensure the ability to generate new data samples.
  - Uncertainty Quantification.
  - Smooth Latent Space for Generative Modeling.
  - Regularization with KL (**_Kullback-Leibler_**) Divergence.
  - Reparameterization Trick for randomness.
  - Robustness to Noise

<hr>
