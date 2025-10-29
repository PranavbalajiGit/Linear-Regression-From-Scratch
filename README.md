# Linear Regression Implementation from Scratch

A complete implementation of Linear Regression algorithm from scratch using NumPy, demonstrating gradient descent optimization and model evaluation on salary prediction data.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Mathematical Foundation](#mathematical-foundation)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)

## Overview

This project implements a Linear Regression model from scratch without using any machine learning libraries like scikit-learn's LinearRegression. The implementation uses gradient descent optimization to learn the optimal weights and bias for predicting salaries based on years of experience.

## Dataset

The model is trained on the **Salary Dataset** containing:
- **Features**: Years of Experience
- **Target**: Salary
- **Size**: 30 samples
- **Train-Test Split**: 67-33 split (20 training, 10 testing samples)

Sample data:
| Years Experience | Salary |
|-----------------|--------|
| 1.1 | 39343 |
| 1.3 | 46205 |
| 10.5 | 121872 |

## Mathematical Foundation

### Linear Regression Model

The model predicts output using the linear equation:

\[ \hat{Y} = wX + b \]

Where:
- \( \hat{Y} \) = Predicted value
- \( w \) = Weight (slope)
- \( X \) = Input features
- \( b \) = Bias (intercept)

### Cost Function

The model uses Mean Squared Error (MSE) as the cost function:

\[ J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2 \]

Where:
- \( m \) = Number of training samples
- \( Y_i \) = Actual value
- \( \hat{Y}_i \) = Predicted value

### Gradient Descent Optimization

To minimize the cost function, we compute partial derivatives and update parameters iteratively:

**Partial Derivatives:**

\[ \frac{\partial J}{\partial w} = -\frac{2}{m} X^T (Y - \hat{Y}) \]

\[ \frac{\partial J}{\partial b} = -\frac{2}{m} \sum (Y - \hat{Y}) \]

**Parameter Updates:**

\[ w = w - \alpha \frac{\partial J}{\partial w} \]

\[ b = b - \alpha \frac{\partial J}{\partial b} \]

Where \( \alpha \) is the learning rate[file:1].

## Implementation Details

### Class Structure

class Linear_Regression:
def init(self, learning_rate, no_of_iterations):
self.learning_rate = learning_rate
self.no_of_iterations = no_of_iterations


### Key Methods

#### 1. `fit(X, Y)` Method

Trains the model by initializing weights and bias, then iteratively updating them using gradient descent[file:1].

def fit(self, X, Y):
# Get dimensions: m (samples), n (features)
self.m, self.n = X.shape # Returns (30, 2) for this dataset

# Initialize parameters to zero
self.w = np.zeros(self.n)  # Weight vector
self.b = 0                  # Bias term

# Store training data
self.X = X
self.Y = Y

# Gradient descent iterations
for i in range(self.no_of_iterations):
    self.update_weights()


**Initialization:**
- Weights (\(w\)) initialized to zeros with shape matching feature dimensions
- Bias (\(b\)) initialized to zero
- Training data stored for optimization[file:1]

#### 2. `update_weights()` Method

Implements the core gradient descent algorithm[file:1]:

def update_weights(self):
# Get predictions using current parameters
Y_prediction = self.predict(self.X)

# Calculate partial derivatives
dw = (-2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
db = (-2 * np.sum(self.Y - Y_prediction)) / self.m

# Update parameters using learning rate
self.w = self.w - self.learning_rate * dw
self.b = self.b - self.learning_rate * db


**Gradient Descent Steps:**
1. **Prediction**: Compute \(\hat{Y}\) using current parameters
2. **Calculate Gradients**: Compute \(\frac{\partial J}{\partial w}\) and \(\frac{\partial J}{\partial b}\)
3. **Update Parameters**: Adjust \(w\) and \(b\) in the direction that reduces cost[file:1]

#### 3. `predict(X)` Method

Makes predictions using the learned linear equation[file:1]:

def predict(self, X):
return X.dot(self.w) + self.b

Returns \(\hat{Y} = wX + b\) for any input \(X\)[file:1].

### Hyperparameters

model = Linear_Regression(learning_rate=0.02, no_of_iterations=1000)


- **Learning Rate**: 0.02 (controls step size in gradient descent)
- **Iterations**: 1000 (number of optimization steps)[file:1]

## Results

### Learned Parameters

After training for 1000 iterations with learning rate 0.02[file:1]:

Weights: 9514.400999035135
Bias: 23697.406507136307


### Final Equation

The learned linear regression model[file:1]:

\[ \text{Salary} = 9514.40 \times \text{YearsExperience} + 23697.41 \]

This means:
- For every additional year of experience, salary increases by approximately **$9,514**
- The base salary (0 years experience) is approximately **$23,697**[file:1]

### Model Visualization

The implementation includes a comparison plot showing[file:1]:
- **Red dots**: Actual test data (ground truth)
- **Blue line**: Model predictions (regression line)

The visualization demonstrates how well the learned linear model fits the test data, showing the relationship between years of experience and salary predictions[file:1].

## Requirements

numpy
pandas
scikit-learn (for train_test_split only)
matplotlib

Install dependencies:
pip install numpy pandas scikit-learn matplotlib

## Usage

### Training the Model

Load your data
salary_data = pd.read_csv('salary_data.csv')

Prepare features and target
X = salary_data.iloc[:, :-1].values
Y = salary_data.iloc[:, 1].values

Split data
X_train, X_test, Y_train, Y_test = train_test_split(
X, Y, test_size=0.33, random_state=2)

Create and train model
model = Linear_Regression(learning_rate=0.02, no_of_iterations=1000)
model.fit(X_train, Y_train)

View learned parameters
print("Weights:", model.w)
print("Bias:", model.b)


### Making Predictions

Predict on test data
test_predictions = model.predict(X_test)

Visualize results
plt.scatter(X_test, Y_test, color='red', label='Actual')
plt.plot(X_test, test_predictions, color='blue', label='Predicted')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Experience vs Salary')
plt.legend()
plt.show()


## Key Learning Points

1. **Gradient Descent**: Implemented from scratch showing how iterative optimization minimizes cost function
2. **Partial Derivatives**: Calculated gradients to determine parameter update direction
3. **Learning Rate**: Controls convergence speed (0.02 provides stable learning)
4. **Matrix Operations**: Used NumPy's efficient vectorization for calculations
5. **Model Evaluation**: Visual comparison of predictions vs actual values

## Project Structure

.
├── Linear_Regression_Model.ipynb # Main implementation notebook
├── salary_data.csv # Dataset
└── README.md # This file


## Acknowledgments

This implementation demonstrates the fundamental concepts of:
- Supervised machine learning
- Gradient descent optimization
- Linear regression mathematics
- NumPy for scientific computing

---

**Note**: This is an educational implementation. For production use, consider using optimized libraries like scikit-learn.
