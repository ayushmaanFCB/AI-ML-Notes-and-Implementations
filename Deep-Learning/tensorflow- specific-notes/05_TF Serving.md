# TF Serving

### TensorFlow Serving:

- Serving models in production environment.
- Scalable and Low-Latency.
- Deploy new algorithms keeping the same server Architecture and APIs.
- Python API - `tensorflow-serving-api` rely on `gRPC` (Google Remote Procedure Call) .
- Steps:
  - Install libraries including `gRPC`.
  - Train and Save the model.
  - Examine the saved model.
  - Serve your model with TensorFlow serving.
    - Install TensorFlow serving.
    - Start running.
  - Make request
  - Make REST Request.
- Sample Code:

```python
export_path = "./simple_model/1"
model.export(model, export_path)
print(f"Model saved to: {export_path}")
```

```bash
docker pull tensorflow/serving
docker run -p 8501:8501 --name=tf_serving --mount type=bind,source=$(pwd)/simple_model,target=/models/simple_model -e MODEL_NAME=simple_model -t tensorflow/serving
```

```python
import requests
import json

# Define the endpoint and data
url = "http://localhost:8501/v1/models/simple_model:predict"
data = {"instances": [[5.0], [6.0]]}

# Send the request
response = requests.post(url, json=data)

# Display the result
print("Response:")
print(response.json())
```

<hr>

### TensorBoard:

- Tool for measurement and visualization during ML workflow.
- Enables tracking metrics like Accuracy, Loss, Model Graph, etc.
- Project embeddings to a lower dimensional space.
- 4 components:
  - Scalars (for model metrics)
  - Graphs (model architecture)
  - Distributions (of weights and biases for various layers)
  - Histograms (for above distribution)
- Code:

```python
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a TensorBoard callback to log training metrics (such as loss and accuracy)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Launch TensorBoard directly within the Jupyter notebook
%load_ext tensorboard
%tensorboard --logdir logs/fit  # Start TensorBoard in the notebook to visualize training progress
```

<hr>

### Distributed Strategy:

- Distributed Training across multiple GPUs, Workers or Both.
- Components:
  - Data Parallelism (same model replicated, but Large Dataset is split)
  - Model Parallelism (Large model is split but dataset is replicated)
  - Hybrid (both)
  - Synchronization (Gradients aggregated and averaged across all replicas before updation)
- Strategies:
  - `tf.distributed.Strategy`: Primary, distribute your model training across machines, GPUs or TPUs.
  - `tf.distributed.MirroredStrategy`: Perform synchronous distributed training across multiple GPUs.
  - `tf.distributed.experimental.MultiWorkerMirroredStrategy`: Perform synchronous distributed training across GPUs in multiple workers.
  - `tf.distributed.experimental.TPUStrategy`: Synchronous Distributed Training across TPUs.
  - `tf.distributed.experimental.CentralStorageStrategy`: Perform synchronous training from a central CPU storage. Central storage (stores parameters) -> Worker Devices (data is divided into subsets and distributed training). -> Gradient Aggregation -> Parameter Updation.
  - `tf.distributed.experimental.ParameterStrategy`: Model Parameters are partioned and stored ins servers -> Worker Nodes -> Gradient Updates -> Parameter Retrieval -> Repeat

<hr>
