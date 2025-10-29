# 🤖 Linear Regression from Scratch (Python/NumPy)

> A simple, custom implementation of the **Linear Regression** algorithm built purely with Python and **NumPy**. This project focuses on understanding the core mechanics of the model, particularly the **Gradient Descent** optimization process.

---

## 📁 Repository Structure

The core implementation is contained within the Jupyter Notebook:
* **`Linear_Regression_Model.ipynb`**: Contains the `Linear_Regression` class definition, dataset loading, training, and testing.

---

## 🔑 The Model: `Linear_Regression` Class

The model is defined by a class that encapsulates the initialization, training, and prediction logic.
```python
class Linear_Regression:
    def __init__(self, learning_rate, no_of_iterations):
        # Initializes hyperparameters
        
    def fit(self, X, Y):
        # Initializes weights and runs the training loop
        
    def update_weights(self):
        # Performs a single step of Gradient Descent
        
    def predict(self, X):
        # Calculates predictions using the final parameters
```

---

## ✨ Key Method Explanations

### 1. `fit(self, X, Y)`: Training the Model

The `fit` method orchestrates the training process:
1. **Initialization**: It first initializes the model's parameters—the weight vector (`self.w`) and the bias (`self.b`)—to zero.
2. **Iteration**: It then enters a loop defined by the `no_of_iterations` hyperparameter (epochs).
3. **Optimization**: In each iteration, it calculates the current prediction, determines the error, and calls `update_weights()` to adjust the parameters based on the error.

### 2. `update_weights()`: Gradient Descent Optimization

This is the core method for learning. It performs a single step of Gradient Descent to minimize the Cost Function (implicitly the Mean Squared Error - MSE).

#### 📉 Partial Derivatives of the Cost Function

To find the minimum cost, we must calculate the gradient (slope) of the cost function with respect to each parameter. These are the partial derivatives, which tell us the direction and magnitude of the steepest ascent.

* **Partial Derivative with respect to Weights (`dw`)**:
```
  dw = -2/m * Σ(Xi · (Yi - Ŷi))
```

* **Partial Derivative with respect to Bias (`db`)**:
```
  db = -2/m * Σ(Yi - Ŷi)
```

(where `m` is the number of training examples, `Y` is the true value, and `Ŷ` is the predicted value.)

#### 🎯 Parameter Update Rule

Once the gradients (`dw` and `db`) are calculated, the parameters are updated by moving in the opposite direction (descent). The size of this step is controlled by the Learning Rate (`self.learning_rate`).
```
w = w - learning_rate · dw
b = b - learning_rate · db
```

### 3. `predict(self, X)`: Making Predictions

This method uses the final, optimized weights (`self.w`) and bias (`self.b`) to calculate the predicted output (`Ŷ`) for new input data (`X`).

The prediction is based on the fundamental linear equation:
```
Ŷ = Xw + b
```

---

## 🚀 Example Usage

The notebook demonstrates initializing and training the model with specific hyperparameters:
```python
# Initialize the model with a learning rate and number of iterations
model = Linear_Regression(learning_rate=0.02, no_of_iterations=1000)

# Train the model
model.fit(X_train, Y_train)

# Make a prediction
predictions = model.predict(X_test)
```

---

## 🧑‍💻 Setup and Run

To run this implementation locally, follow these steps:

1. **Clone the repository**:
```bash
   git clone https://github.com/PranavbalajiGit/Linear-Regression-From-Scratch.git
   cd Linear-Regression-From-Scratch
```

2. **Install dependencies** (NumPy is the only required library):
```bash
   pip install numpy pandas matplotlib
```

3. **Open the notebook**: Open `Linear_Regression_Model.ipynb` in a Jupyter environment (like VS Code or Jupyter Notebook/Lab) and execute the cells.

---

## 📊 Features

- ✅ Pure NumPy implementation (no scikit-learn)
- ✅ Clear visualization of Gradient Descent
- ✅ Step-by-step parameter updates
- ✅ Educational focus on understanding the fundamentals

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/PranavbalajiGit/Linear-Regression-From-Scratch/issues).

---

## 👤 Author

**Your Name**
- GitHub: [@PranavbalajiGit](https://github.com/PranavbalajiGit)

---

**⭐ Star this repo if you find it helpful!**