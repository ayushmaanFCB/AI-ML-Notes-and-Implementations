
## Regularization Techniques, Overfitting & Dropout

### Overfitting

* Occurs when a model learns the noise and specific details of training data rather than general patterns.
* Symptoms: Very low training loss but high validation loss.
* Causes: Complex models, insufficient data, lack of constraints.

### Regularization Methods

1. **L1 Regularization (Lasso)**

   * Adds the absolute value of weights to the loss function.
   * Encourages sparsity by driving some weights to zero.

2. **L2 Regularization (Ridge)**

   * Adds squared weight values to the loss.
   * Prevents extremely large weights, promoting generalization.

3. **Early Stopping**

   * Stops training when validation loss stops improving, preventing overfitting.

4. **Dropout**

   * Randomly sets a fraction of neurons to zero during training, reducing co-adaptation and improving generalization.

#### Example with TensorFlow

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.5),  # Dropout layer
    layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

## Convolutional Neural Networks (CNNs)

### Why CNNs?

* CNNs are specialized for image processing.
* They automatically extract hierarchical features:

  * Low-level: edges, textures
  * High-level: objects, faces, patterns

### Key Components

* **Convolution Layer (Conv2D):** Applies filters (kernels) to detect features.
* **Activation (ReLU):** Adds non-linearity.
* **Pooling Layer:** Reduces dimensions (e.g., MaxPooling).
* **Fully Connected Layer (Dense):** Final decision making.

### CNN Flow

```
Input → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Dense → Output
```

#### Example with TensorFlow

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

## Recurrent Neural Networks (RNNs)

### Why RNNs?

* Designed to process sequential data (time series, text, speech).
* Maintains a hidden state to remember previous inputs.

### Limitations

* Standard RNNs suffer from vanishing gradients, making it hard to learn long-term dependencies.

#### Example with TensorFlow

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.SimpleRNN(64, activation='tanh', input_shape=(100, 1)),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## Long Short-Term Memory (LSTM)

### Why LSTMs?

* An improved RNN architecture that addresses vanishing gradient problems.
* Uses gates to control the flow of information:

  * Forget Gate: Discards irrelevant information.
  * Input Gate: Decides which new information to store.
  * Output Gate: Controls what information is output.

### Applications

* Natural Language Processing (NLP)
* Time Series Forecasting
* Speech Recognition

#### Example with TensorFlow

```python
model = models.Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=(100,1)),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## Encoders

### Concept

* Encoders compress input data into a latent representation.
* Useful in:

  * Autoencoders (dimensionality reduction, denoising)
  * Seq2Seq models (machine translation)
  * Variational Autoencoders (VAE) for generative tasks

### Architecture

```
Input → Encoder → Latent Space → Decoder → Output
```

#### Autoencoder Example

```python
from tensorflow.keras import layers, models

input_img = layers.Input(shape=(784,))
encoded = layers.Dense(64, activation='relu')(input_img)
decoded = layers.Dense(784, activation='sigmoid')(encoded)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

---

## Transfer Learning

### Concept

* Reuse pretrained models (e.g., VGG16, ResNet, MobileNet) trained on large datasets like ImageNet.
* Speeds up training and improves performance, especially on small datasets.

### Strategies

1. **Feature Extraction:** Freeze base model layers, only train added layers.
2. **Fine-Tuning:** Unfreeze selected layers and train with a small learning rate.

#### Example with TensorFlow

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Freeze base model layers

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## Data Augmentation

### Why Use Data Augmentation?

* Artificially increases dataset size by generating modified versions of images.
* Reduces overfitting and improves generalization.

### Common Augmentations

* Rotation, flipping, cropping
* Scaling, shifting
* Adjusting brightness, saturation, hue

#### TensorFlow Example

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.flow(x_train, y_train, batch_size=32)
```

---