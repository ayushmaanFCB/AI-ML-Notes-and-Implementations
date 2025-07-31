# GANs

### Deep Dream Algorithm

- Uses CNNs to over-interpret and enhance certain patterns in the images.
- Done by forwarding an image through network -> Calculating gradient w.r.t activations -> Activations are then boosted -> Image modified to increase these activations.
- Also called Inceptionism.
- Gradient Ascent: Maximize a function, moves in direction of positive ascent.
- Shallow Layers: Amplify low level features.
- Deeper Layers: Amplify high level features.
- Boosts pattern it sees in a given image, based on what it is trained to see.
- Working:
  - Take a pre-trained model like InceptionV3.
  - Take an Input image.
  - Select a layer to perform dreaming.
  - No backpropagation, instead Gradient Ascent -> Gradients are computed for input image w.r.t chosen activation.
  - Image iteratively adjusted to amplify activation.
  - Octave Processing: Image processed at multiple scales (octaves) to enhance both small and large patterns.

<hr>

### General Adversal Networks (GAN):

- Generator and Discriminator
- **Generator** (G): Takes random noise as input and generates synthetic data (e.g., images).
- **Discriminator** (D): Receives both real data and the synthetic data from the Generator, then
  classifies them as real or fake.
- Generator aims to improve its output to fool the Discriminator.
- Both the discriminator and the generator are trying to optimize a different and opposite fitness function (loss function).

<hr>

### Deep Convolutional GAN (DCGAN):

- Composes of convolution layers without max pooling or fully connected layers.
- In normal GANs, full connected are present at start of generator (to create from noise), and end of discriminator (flattening and binary classification).
- Uses convolutional stride and transposed convolution for the downsampling and the upsampling.
- Generator -> Transposed Convolution, Discriminator -> Strided Convolution.
- Overcomes **_Mode Collapse_** issue: generator got biased towards a few outputs and can’t able to produce outputs of every variation from the dataset.
- Replace all max pooling with convolutional stride.
- Use ReLU in generator, except output layer (use tanH).
- Use LeakyReU in discriminator.
- Use Batch normalization except the output layer for the generator and the input layer of the discriminator.

<hr>

### Conditional GAN (cGAN):

- Generate Data based on some conditions.
- Example: a car generation model is trained to generate only Hyundai cars.
- Class labels used for deliberate or targeted generation.
- Generator, Discriminator and Conditioning Variable.
- The label `y` is combined with generator input `p(z)` as joint hidden representation.
- Discriminator als receives input `x` and labels `y`.
- We use an embedding layer followed by a fully connected layer with a linear activation that scales the embedding to the size of the image before concatenating it in the model as an additional channel or feature map.
- Types: SRGAN, CycleGAN, Pix2Pix (image-to-image translation).

<hr>

### CycleGAN

- Advanced version of the Pix2Pix model.
- CycleGAN is a generative model that learns to translate images from one domain to another without paired training data, using a cycle consistency loss to ensure the translations are meaningful and reversible.
- Utilize **unpaired** data: Datasets in which the Inputs and the Labels are not directly Related.
- Does not need corresponding image pairs in the source and target domains.
- Ensures that a transformation from domain A to B domain and back to 𝐴 produces the original image.
- **GAN Framework**:
  - Two adversarial networks are used:
    - \( G \): Translates images from domain \( A \) to domain \( B \).
    - \( F \): Translates images from domain \( B \) to domain \( A \).
  - Each generator is paired with a discriminator to determine if the translated image looks real in the target domain.
- Loss Functions:

  - **Adversarial Loss**: Encourages the generators to produce realistic images.
  - **Cycle-Consistency Loss**: Preserves structural consistency between the original and reconstructed images. Let G and F be two generators:

    $$
    \mathcal{L}_{\text{cycle}} = \| G(F(x)) - x \| + \| F(G(y)) - y \|
    $$

  - **Identity Loss (optional)**: Helps retain color and other low-level details during translation.

- Example: transforming horses into zebras (and back) or converting summer landscapes to winter scenes (and vice-versa) without needing specific paired images of "horse next to zebra" or "summer scene next to winter scene."

<hr>

### Super Resoution GAN (SRGAN)

- Upsampling a low-resolution image into a higher resolution with minimal information distortion.
- combining the elements of efficient **sub-pixel nets**, as well as traditional GAN loss functions.
- Sub-pixel convolutional networks: upsample low-resolution images into high-resolution ones efficiently by rearranging feature maps into higher resolution space.
- SRGANs are combination of GANs and CNNs.
- Goal of it is to recover missing details, improve sharpness, and improve visual quality.
- Generator learns to capture the very fine details and overall visual characteristics in the
  image.
- The generator transforms LR images into HR images. It is composed of:
  - **Convolutional Layers**(64 filters, 9x9): Extract features from the input image.
  - **Residual Blocks** (16 total): A series of residual blocks with skip connections for stable training and preserving spatial details. Each block has:
    - 2 convolutional layers, 64 3x3 filters, stride 1.
    - Batch Normalization.
    - Parametric ReLU activation.
    - Elementwise Sum for the residual connection.
  - **PixelShuffle Layer** (Upscaling): Also known as sub-pixel convolution, this upscales the feature maps to a higher resolution without artifacts.
  - **Final Convolutional Layer**: Outputs the HR image.
- The discriminator classifies images as real (ground truth HR) or fake (generated HR). It typically has:
  - **Convolutional Layers**: Progressively deeper layers to capture features and differentiate between real and generated HR images.
  - **Leaky ReLU Activation**: Helps prevent vanishing gradients.
  - **Fully Connected Layers**: For binary classification (real vs. fake).
  - The discriminator is trained using a **binary cross-entropy loss**.
- The overall loss function is a combination of:
  1. **Adversarial Loss**: Encourages photo-realistic HR image generation.
  2. **Content Loss**: Ensures that generated images maintain perceptual quality.

<hr>
