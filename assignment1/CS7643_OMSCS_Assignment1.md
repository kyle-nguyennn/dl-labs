# CS7643: Deep Learning — Assignment 1

**Instructor:** Zsolt Kira  
**Deadline:** Feb 2, 2026, 8:00am ET

## Important notes

- This assignment is due on the date/time posted on Canvas. A **48-hour grace period** is available. **No questions** about the assignment are answered during the grace period.
- Discussion is encouraged, but each student must write their own answers and explicitly mention any collaborators.
- Respect the GT Honor Code. Anti-cheating software will be used; flags result in **0** and reporting to OSI.
- **Do not change filenames or function definitions** in the skeleton code; otherwise tests will fail.
- **Do not change import modules** or import additional modules.
- Ensure all deliverables are in correct format and code runs in the test environment. Non-runnable code may score **0**.
- Your programming parts will **NOT** be graded (score 0) if your code prints anything not asked in each question.

---

## Theory Problem Set

<div style="page-break-after: always;"></div>

### 1) Softmax gradient

In problem set 0, you derived the gradient of the log-sum-exp function. Now consider the softmax function **s(z)**, which maps a vector **z** (logits) to a vector with entries:

$$
s_i = \frac{e^{z_i}}{\sum_k e^{z_k}}
$$

**Task:** Derive the gradient of **s** with respect to logits **z**, i.e. derive $\frac{\partial s}{\partial z}$. Consider re-using your work from PS0.

#### Solution:
[placeholder for solution]

---
<div style="page-break-after: always;"></div>

### 2) Linear threshold neuron for AND / OR

Use a single linear threshold neuron with weights \(w\in\mathbb{R}^2\), bias \(b\in\mathbb{R}\), and \(x\in\{0,1\}^2\):

$$
f(x)=
\begin{cases}
1 & \text{if } w^T x + b \ge 0 \\
0 & \text{if } w^T x + b < 0
\end{cases}
$$

**Task A:** Find $w_{\text{AND}}$ and $b_{\text{AND}}$ such that:

| x1 | x2 | fAND(x) |
|---:|---:|--------:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

**Task B:** Find \(w_{\text{OR}}\) and \(b_{\text{OR}}\) such that:

| x1 | x2 | fOR(x) |
|---:|---:|-------:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

---
<div style="page-break-after: always;"></div>

### 3) XOR is not linearly separable

Consider XOR:

| x1 | x2 | fXOR(x) |
|---:|---:|--------:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Task:** Prove that XOR **cannot** be represented using a linear model of the same form as above.

> Hint: Plot the examples in a plane and reason about whether a linear decision boundary can separate the classes.

---
<div style="page-break-after: always;"></div>

## Paper Review

Choose **one** paper below and complete:

1. Provide a short review of the paper.
2. Answer paper-specific questions.

**Guidelines**
- Review: **≤ 350 words**
- Each question answer: **≤ 350 words**

Your review should include:
- The main contribution / key insights; strengths and weaknesses.
- Your personal takeaway (novelty vs other papers you’ve read, future directions, or anything noteworthy).

### Paper Choice 1: *Weight Agnostic Neural Networks* (NeurIPS 2019)

Questions:
- Traditional view: optimization searches weight space. How would you view this paper from the perspective of **search**?
- What does this paper say about representational power of **architectures** given a fixed method for determining weights?
  - Does the method for determining weights matter?
  - Do you think these two have equal representational power? Why or why not?

### Paper Choice 2: *Understanding deep learning requires rethinking generalization*

Questions:
- If neural networks can memorize random labels, why do they learn more meaningful, generalizable representations when there are meaningful patterns in the data?
- How does this finding align (or not) with your understanding of machine learning and generalization?

---

## Coding: Implement and train a network on MNIST

### Overview

You will build a pipeline to train neural networks on MNIST (handwritten digits). You will implement:
- Data loading + batching
- Two models (Softmax Regression and 2-layer MLP)
- Optimizer (vanilla SGD + L2 regularization)
- Learning-curve visualization
- Hyperparameter experiments + report write-up

Run the assignment via:

```bash
python main.py --config configs/<name_of_config_file>.yaml
```

There are **three** pre-defined config files under `./configs`:
- Two are default hyperparameters for the models (Softmax Regression and 2-layer MLP). **Do not modify** these; correctness is partially judged by performance on these defaults.
- `config_exp.yaml` is for hyperparameter tuning experiments; you **may** modify values in this file.

Training runs for the number of epochs specified in the config. At the end of each epoch, the script evaluates on the validation set. After training completes, it evaluates the best model on the test data.

---

### Python and dependencies

Use **Python 3**. Recommended: Anaconda (or Miniconda). Create the conda environment:

```bash
conda env create -f environment.yaml
```

If using your own environment, consult `environment.yaml` to match the coding/grading environment.

---

### Code test

Two ways to test:

1. **Python Unit Tests** (public tests in `tests/`):
   ```bash
   python -m unittest tests.<name_of_tests>
   ```
   Passing local tests does not guarantee full credit; Gradescope has additional tests.

2. **Gradescope Autograder**: submit as in Section 6. Returns public test results (not private). Not recommended as primary development loop.

---

## 1. Data Loading

Download MNIST with the provided script under `./data`:

```bash
cd data
sh get_data.sh
cd ../
```

**Windows 10:**
```bat
cd data
get_data.bat
cd ..
```

This downloads:
- `mnist_train.csv`
- `mnist_test.csv`
into `./data`.

### 1.1 Data Preparation

Split training data into train/validation to avoid overfitting hyperparameter choices.

- Use the first **80%** of the training set as training data.
- Use the remaining **20%** as validation data.
- Organize (train/val/test) into **batches**; shuffle batch combinations across epochs for training.

**Tasks**
- (a) Implement `load_mnist_trainval` in `./utils.py` for training/validation split
- (b) Implement `generate_batched_data` in `./utils.py` for batching

Test:
```bash
python -m unittest tests.test_loading
```

---

## 2. Model Implementation

Implement two networks from scratch (definitions in `./models`):
- Softmax Regression
- Two-layer MLP

Weights are randomly initialized and stored in a **weight dictionary**; gradients stored in a corresponding **gradient dictionary** (initialized to zeros).

Each model exposes a single public method `forward`, which:
- takes a batch of data + labels
- returns loss and accuracy
- **and** computes gradients of all weights based on the training batch (even though it’s called `forward`)

### 2.1 Utility Functions (`./models/_base_network.py`)

#### (a) Activation functions

Implement:
- `sigmoid`, `sigmoid_dev`
- `ReLU`, `ReLU_dev`

Test:
```bash
python -m unittest tests.test_activation
```

#### (b) Loss functions

Implement:
- Softmax function
- Cross Entropy Loss computation

Test:
```bash
python -m unittest tests.test_loss
```

#### (c) Accuracy

Implement `compute_accuracy` in `./models/_base_network.py`.

### 2.2 Model implementation

Implement `forward` in:
- `softmax_regression.py`
- `two_layer_nn.py`

Model specs:
- Softmax Regression: fully-connected layer followed by **ReLU** (no bias terms)
- Two-layer MLP: two fully-connected layers with **Sigmoid** activation in between (uses bias terms)

Remember to apply **softmax** before computing loss.

If `mode == "train"`: compute gradients and store them in gradient dict; else return loss + accuracy only.

Test:
```bash
python -m unittest tests.test_network
```

---

## 3. Optimizer

Implement:
- L2 regularization
- vanilla SGD updates

Before updating weights, apply L2 regularization:

\[
J = L_{CE} + \frac{1}{2}\lambda \sum_{i=1}^{N} w_i^2
\]

SGD update rule:

\[
\theta^{t+1} = \theta^t - \eta \nabla_{\theta} J(\theta)
\]

**Tasks**
- (a) Implement `apply_regularization` in `_base_optimizer.py`. **Do not** apply regularization to bias terms.
- (b) Implement `update` in `sgd.py`.

Test:
```bash
python -m unittest tests.test_training
```

---

## 4. Visualization

Plot learning curves using the stored averaged loss and accuracy values for training and validation at the end of each epoch.

**Task:** Implement `plot_curves` in `./utils.py`. Full marks as long as the plot makes sense.

---

## 5. Experiments

Use your **two-layer MLP** implementation.

Using `config_exp.yaml`, tune hyperparameters and report observations in the report template.

When tuning a specific hyperparameter (e.g., learning rate), leave other hyperparameters as-is in the default config file.

**Tasks**
- (a) Try different learning rates; report observations.
- (b) Try different regularization coefficients; report observations.
- (c) Tune hyperparameters for best accuracy.
- (d) For best accuracy: tune **at least 3 hyperparameters** (not including epochs). Changing epochs alone does not count.
- (e) Best model should show marked improvement vs default hyperparameters.
- (f) Use good scientific methods in reporting observations.

---

## 6. Deliverables

### 6.1 Coding (Gradescope)

Submit a zip file containing all code in the correct structure.

Run:

```bash
bash collect_submission.sh
```

**Windows 10:**
```bat
collect_submission.bat
```

Upload `assignment_1_submission.zip` to Gradescope.

### 6.2 Writeup

Submit a report summarizing experiments and findings (per Section 5) using the provided template.

- Include plots from all experiments where requested.
- Explain observations using ML intuition; justify hyperparameter choices.
- If you need more than one slide for a question, add slides after the given one.
- Export report as **PDF** and submit to Gradescope.
- Combine **theory answers + paper review + report** into **one PDF** and submit to “Assignment 1 Writeup”.

**Important:** When submitting, select **ALL corresponding slides** for each question. Incorrect tagging: **-1 point per incorrectly tagged question** (future assignments may be more severe).
