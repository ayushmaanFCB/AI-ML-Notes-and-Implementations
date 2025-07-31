# TensforFlow Basics

### Introduction to Tensorflow

- Open-source Machine Learning and Deep Learning framework developed by Google.
- Developed in C++ and has its implementation in Python.
- Keras can now run on top of TensorFlow.
- Features:
  - **Tensors:** n-dimensional arrays (or data structures) that can hold data of any dimension, making it flexible for complex data.
  - **Computationalm Graphs:** a series of operations that define how data flows through the model.
  - High level **Keras API**.
  - **GPU** and **TPU Support**
  - **Tensorflow Serving**: for production environments.
  - **TensorBoard**: statistical visualization
  - **Tensorflow Hub:** repo of models
- Applications of TensorFlow in Healthcare
  - Medical Image Analysis
  - Predictive Analytics
  - Medical Image Synthesis
  - Drug Discovery and Genomics
  - Clinical Workflow Automation
  - Prosthetics and Robotic Surgery
  - Telemedicine and Remote Monitoring

<hr>

### Optimizers

- Minimize the loss function by updating the model's weights iteratively.
- Ensure optimal convergence.
- **Gradient Descent Optimization:**

  - Minimize the loss function by iteratively updating the weights in the direction of the negative gradient.
    | **Method** | **Description** | **Advantages** | **Disadvantages** |
    |--------------------------|-----------------------------------------------------------|---------------------------------------------|-------------------------------------------|
    | **Batch Gradient Descent** | Uses the entire dataset to compute the gradient. | Stable convergence, accurate gradient. | Slow for large datasets, high memory usage. |
    | **Mini-Batch Gradient Descent** | Uses a small random subset of the dataset. | Faster convergence, balances stability and speed. | Requires tuning of batch size. |
    | **Stochastic Gradient Descent (SGD)** | Uses one training example at a time. | Very fast, can escape local minima. | Noisy updates, less stable convergence. |
  - Example:

    ```text
    instances = 500, batches = 25

    How many times weight updated?

    1. Batch GD: 1 per epoch
    2. SGD: 500 per epoch
    3. 25 per epoch
    ```

- Other commonly used optimzers:

  - **ADAM**: Adapts the learning rate for each parameter individually, using estimates of both the mean and variance of the gradients.
  - **ADAGRAD**: Adapts the learning rate for each parameter individually, but scales the learning rate inversely proportional to the cumulative sum of squared gradients.
  - **ADADELTA**: An extension of ADAGRAD that addresses its aggressively decreasing learning rates by using a moving average of squared gradients.

  - | **Method**   | **Description**                                                               | **Advantages**                                                     | **Disadvantages**                             |
    | ------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------- |
    | **Adam**     | Combines momentum and adaptive learning rates.                                | Efficient, works well with sparse data, and adapts learning rates. | Can be sensitive to hyperparameters.          |
    | **Adagrad**  | Adapts the learning rate based on the frequency of updates.                   | Good for sparse data, no need to tune learning rate.               | Learning rate can become too small over time. |
    | **Adadelta** | An extension of Adagrad that maintains a moving average of squared gradients. | Addresses the diminishing learning rate issue of Adagrad.          | More complex to implement than Adagrad.       |

<hr>

### Tensorflow vs Keras vs PyTorch

| **Framework**  | **Description**                                                                                         | **Ease of Use**                         | **Flexibility**                      | **Community and Ecosystem**                      |
| -------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------- | ------------------------------------ | ------------------------------------------------ |
| **TensorFlow** | A comprehensive open-source platform for machine learning.                                              | Steeper learning curve, more complex.   | Highly flexible but can be verbose.  | Large community, extensive resources, and tools. |
| **Keras**      | A high-level API for building and training deep learning models, originally built on top of TensorFlow. | Very user-friendly, intuitive API.      | Less flexible than TensorFlow alone. | Growing community, integrated with TensorFlow.   |
| **PyTorch**    | An open-source deep learning framework that emphasizes dynamic computation graphs.                      | Easy to learn and use, Pythonic syntax. | Highly flexible and dynamic.         | Strong community, especially in research.        |

- **Caffe:** Oldest framework, least flexible, very low abstraction hence highly complex.

<hr>

### Transfer Learning and Imagenet

- **IMAGENET:**
  - An open source repository of images that consist of **1000 classes**, **22000 Categories (synset)** and over **14 million images**.
  - Labelled images.
  - Build according to **_WordNet Hierarchy_**:
    - Represented by a "**_synonym set_**" (**synset**).
    - Example, dog synset includes multiple breeds.
    - Labels are organized hierarchically (e.g., mammals → dogs → golden retriever).
  - In ImageNet, "categories" (synsets) are broader groupings of objects, while "classes are more specific subcategories within those broader groupings.
- **Transfer Learning:**
  - **Strategy 1:** Freeze pretrained model layers -> Only train newly added layers (random weights).
  - **Strategy 2:** Unfreeze layers (hence pre-trained model learns our weights) -> Now train using very small learning rate.

<hr>

### Data Augmentation

- Working:
  - Image fed to the pipeline.
  - Augmentation pipeline is defined by sequential steps of different augmentations.
  - Image is fed through the pipeline and processed through each step with a **probability**.
  - After the image is processed, the human expert randomly verifies the augmented results.
  - Augmented data is ready to use by the AI training process.
- Manual operations:

```python
import tensorflow as tf

tf.image.flip_left_right()
tf.image.rgb_to_grayscale()
tf.image.adjust_saturation()
tf.image.adjust_brightness()
tf.image.central_crop()          # Crop the image from center up to the image part you desire
tf.image.rot90()
```

- Using Datagenerator

```python
from keras.image.preprocessing import ImageDataGenerator

datagen = ImageDataGenerator(
  rotation_range = 90 ,
  height_shift_range= 0.2,
  shear_range= 0.2,
  horizontal_flip=True,
  fill_mode='nearest'
)
```

<hr>

### Repeated Topics

(notes excluded for all of these)

- Simple NN and **_Backpropagation_** (check numerical)
- Activation Functions
- CNN
  - Read about pooling types, strides, convolution, etc.
- RNN and LSTM
  - read about how LSTM solves RNN's vanishing gradient problem.
  - LSTMs use gates to control the flow of information, preventing gradients from vanishing as they propagate through time.
- Transfer Learning

<hr>
