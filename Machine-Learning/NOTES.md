# 📚 Machine Learning – Brief Concept Notes

## Basics of Machine Learning

### What is Machine Learning?

Machine Learning is a subset of Artificial Intelligence (AI) where systems learn patterns from data and make predictions without being explicitly programmed.

### Types of Machine Learning

1. **Supervised Learning**

   - Learns from labeled data.
   - Examples: Regression, Classification.
   - Algorithms: Linear Regression, Decision Trees, Random Forest, SVM.

2. **Unsupervised Learning**

   - Works on unlabeled data to find patterns.
   - Examples: Clustering, Dimensionality Reduction.
   - Algorithms: K-Means, PCA.

3. **Reinforcement Learning**

   - Learns through trial-and-error using rewards and penalties.
   - Examples: Game AI, Robotics.

### Bias-Variance Tradeoff

- **High Bias (Underfitting):** Model too simple, misses patterns.
- **High Variance (Overfitting):** Model too complex, fits noise in data.
- Goal: Find a **balance** between bias and variance.

## Ensemble Learning

Ensemble methods combine predictions of multiple models to improve performance.

### Types:

- **Bagging (Bootstrap Aggregating):**
  Reduces variance by training models in parallel on random subsets of data (e.g., Random Forest).

- **Boosting:**
  Sequentially builds models where each new model focuses on errors made by the previous one (e.g., AdaBoost, Gradient Boosting, XGBoost).

- **Stacking:**
  Combines predictions from multiple base models using a meta-model.

## Gradient Boosting

Gradient Boosting is an **ensemble learning** technique that builds models sequentially, where each new model tries to **correct the errors** made by previous ones.

**How it works:**

1. Train an initial weak model (usually a decision tree).
2. Calculate residuals (errors) between predictions and actual targets.
3. Train a new model on these residuals.
4. Add the new model to the ensemble.
5. Repeat the process for several iterations.

**Pros:**
✔ High accuracy
✔ Handles non-linear relationships well

**Cons:**
✘ Computationally expensive
✘ Can overfit without regularization

## AdaBoost (Adaptive Boosting)

AdaBoost combines multiple weak learners by adjusting the **weights** of data points after each iteration.

**Steps:**

1. Start with equal weights for all samples.
2. Train a weak learner (e.g., decision stump).
3. Increase weights for misclassified samples.
4. Train the next learner using adjusted weights.
5. Combine all learners using weighted voting.

**Pros:**
✔ Reduces bias
✔ Simple to implement

**Cons:**
✘ Sensitive to outliers
✘ Requires careful tuning

## XGBoost (Extreme Gradient Boosting)

XGBoost is an advanced implementation of Gradient Boosting with **regularization** and **parallelization**.

**Key Features:**

- Optimized tree splitting with second-order derivatives.
- Regularization to prevent overfitting.
- Handles missing data internally.
- Much faster on large datasets.

**Applications:** Kaggle competitions, fraud detection, recommendation systems.

## Batch & Epoch

### Batch

- A **subset** of the training dataset processed in one iteration.
- **Batch Size:** Hyperparameter controlling learning speed and stability.

### Epoch

- **One complete pass** through the entire dataset.
- Usually, multiple epochs are required for convergence.

## Gradient Descent Variants

### Batch Gradient Descent (BGD)

- Uses the entire dataset for gradient computation.
- **Pros:** Accurate updates
- **Cons:** Slow for large datasets

### Stochastic Gradient Descent (SGD)

- Uses a single sample per update.
- **Pros:** Fast, can escape local minima
- **Cons:** Noisy updates

### Mini-Batch Gradient Descent

- Uses small batches of data for updates.
- **Pros:** Balances speed and accuracy
- **Cons:** Slightly more complex

## Bagging vs Boosting

| Feature        | Bagging         | Boosting                   |
| -------------- | --------------- | -------------------------- |
| Goal           | Reduce variance | Reduce bias                |
| Model Training | Parallel        | Sequential                 |
| Sample Weights | Equal           | Adaptive (focus on errors) |
| Examples       | Random Forest   | AdaBoost, XGBoost          |

## Stacking (Stacked Generalization)

Stacking combines multiple models using a **meta-model**:

1. Train several base models on the training data.
2. Generate predictions (meta-features) from these models.
3. Train a meta-model using meta-features.
4. Meta-model combines predictions for final output.

## Decision Trees

A Decision Tree is a flowchart-like structure where nodes represent decisions based on feature values.

- **Root Node:** Represents the entire dataset.
- **Internal Nodes:** Represent decisions based on attributes.
- **Leaf Nodes:** Represent outcomes.

**Splitting Criteria:**

- **ID3:** Information Gain (Entropy)
- **CART:** Gini Index
- **CHAID:** Chi-Square test
- **MARS:** Piecewise linear functions

**Pros:** Easy to interpret, handles both categorical & numerical data.
**Cons:** Prone to overfitting, unstable to small data changes.

## Binary Search Trees (BST)

- Left child < Parent
- Right child > Parent
- No duplicates
- Both left and right subtrees must also be BSTs.

## ID3 Algorithm

1. Select unused attributes.
2. Calculate **Entropy** and **Information Gain**.
3. Choose the attribute with maximum Information Gain.
4. Split dataset based on this attribute.
5. Repeat until all subsets are pure.

## Pruning (Avoiding Overfitting in Trees)

**Goal:** Prevent the model from becoming too complex.

### Types:

- **Pre-Pruning:** Stop tree growth early if split gain is small.
- **Post-Pruning:** Grow full tree, then remove unnecessary branches.

### Methods:

- **Reduced Error Pruning:** Keep pruning if validation accuracy improves.
- **Cost Complexity Pruning:** Trade-off between tree size and impurity.

## Handling Missing Values

- Impute with **mean**, **median**, or **mode**.
- Use predicted values (e.g., regression).
- Replace with placeholder or remove rows (if minimal).

## CART (Classification and Regression Trees)

- **Classification:** Uses **Gini Index** to measure impurity.
- **Regression:** Uses **variance reduction** (standard deviation).

## Random Forest

An ensemble of decision trees where:

- Each tree is trained on a **bootstrap sample**.
- At each split, only a random subset of features is considered.
- Predictions are aggregated (majority vote or average).

**Pros:**
✔ Reduces overfitting
✔ Works well with high-dimensional data

## Dimensionality Reduction

Reduces number of features while retaining most of the data’s variance.

**Benefits:**

- Prevents overfitting
- Improves computation speed
- Reduces storage requirements

### Feature Selection

- **Filter Methods:** Rank features by statistical metrics.
- **Wrapper Methods:** Evaluate subsets using models.
- **Embedded Methods:** Built into algorithms (e.g., Lasso).

### Feature Extraction

- **PCA:** Finds principal components capturing maximum variance.
- **LDA:** Maximizes class separability.
- **SVD:** Matrix factorization to reveal hidden patterns.

## Principal Component Analysis (PCA)

**Steps:**

1. Standardize data.
2. Compute covariance matrix.
3. Calculate eigenvalues & eigenvectors.
4. Select top eigenvectors (principal components).
5. Project data onto these components.

**Uses:**

- Data visualization
- Feature extraction
- Data compression

## Linear Discriminant Analysis (LDA)

**Steps:**

1. Compute class-wise means.
2. Compute within-class and between-class scatter matrices.
3. Solve eigenvalue problem for optimal projection.
4. Project data to maximize class separation.

**Uses:**

- Dimensionality reduction for classification
- Improving classifier performance

## Singular Value Decomposition (SVD)

Matrix factorization technique:

```
A = U Σ Vᵀ
```

- **U:** Left singular vectors
- **Σ:** Diagonal matrix with singular values
- **Vᵀ:** Right singular vectors

**Uses:**

- Dimensionality reduction
- Recommendation systems (e.g., Netflix)
- Image compression
- Anomaly detection

## PCA vs LDA vs SVD – **Analogy**

- **PCA:** Organizes coins by **size** (variance), ignoring heads/tails (class labels).
- **LDA:** Organizes coins by **heads/tails** (class separation), ignoring some size.
- **SVD:** Breaks coins into **size, weight, and composition factors**.

---
