import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
age = np.array([21, 22, 23, 24, 25,26,27,28])  # Independent variable
salary = np.array([20000, 40000, 50000,40000, 50000,60000,80000,100000])  # Dependent variable

# Compute coefficients
age_length = len(age)
mean_age = np.mean(age)
mean_salary = np.mean(salary)

numerator = np.sum((age - mean_age) * (salary - mean_salary))
denominator = np.sum((age - mean_age) ** 2)

slope = numerator / denominator
intercept = mean_salary - slope * mean_age

# Displasalary results
print(f"Slope (m): {slope}")
print(f"Intercept (b): {intercept}")

# Prediction function
def predict(age):
    return slope * age + intercept

# Predict for a new value
new_age = 6
print(f"Prediction for age={new_age}: {predict(new_age)}")

def plot_regression_line(age, salary, slope, intercept):
    plt.scatter(age, salary, color='blue', label='Data Points')
    regression_line = slope * age + intercept
    plt.plot(age, regression_line, color='red', label='Regression Line')
    plt.xlabel('age')
    plt.ylabel('salary')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid()
    plt.show()

# Plot the data and regression line
plot_regression_line(age, salary, slope, intercept)
