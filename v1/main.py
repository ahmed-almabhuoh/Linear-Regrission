import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import os

# Ensure output directory exists
output_dir = "C:/Users/Aseel Almabhouh/Desktop/Assignment/v1/plots"
os.makedirs(output_dir, exist_ok=True)

# Dataset
Y = np.array([43, 63, 71, 61, 81, 43, 58, 71, 72, 67, 64, 67, 69, 68, 77, 81, 74, 65, 65, 50, 50, 64, 53, 40, 63, 66, 78, 48, 85, 82])
X1 = np.array([51, 64, 70, 63, 78, 55, 67, 75, 82, 61, 53, 60, 62, 83, 77, 90, 85, 60, 70, 58, 40, 61, 66, 37, 54, 77, 75, 57, 85, 82])
X2 = np.array([30, 51, 68, 45, 56, 49, 42, 50, 72, 45, 53, 47, 57, 83, 54, 50, 64, 65, 46, 68, 33, 52, 52, 42, 42, 66, 58, 44, 71, 39])
X3 = np.array([39, 54, 69, 47, 66, 44, 56, 55, 67, 47, 58, 39, 42, 45, 72, 72, 69, 75, 57, 54, 34, 62, 50, 58, 48, 63, 74, 45, 71, 59])
X4 = np.array([61, 63, 76, 54, 71, 54, 66, 70, 71, 62, 58, 59, 55, 59, 79, 60, 79, 55, 75, 64, 43, 66, 63, 50, 66, 88, 80, 51, 77, 64])
X5 = np.array([92, 73, 86, 84, 83, 49, 68, 66, 83, 80, 77, 74, 63, 77, 77, 54, 79, 80, 85, 78, 64, 80, 80, 57, 75, 76, 78, 83, 74, 78])
X6 = np.array([45, 47, 48, 35, 47, 34, 35, 41, 31, 41, 34, 41, 25, 35, 46, 36, 63, 60, 46, 52, 33, 41, 37, 49, 33, 72, 49, 38, 55, 39])

X = np.column_stack((X1, X2, X3, X4, X5, X6))
variables = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

results = []

# Question (a): Construct Linear Models
print("Question (a): Linear Models")
print("=" * 50)
for i, (X_var, var_name) in enumerate(zip([X1, X2, X3, X4, X5, X6], variables)):
    X_var_2d = X_var.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X_var_2d, Y)
    y_pred = model.predict(X_var_2d)
    
    r2 = model.score(X_var_2d, Y)
    mse = mean_squared_error(Y, y_pred)
    adj_r2 = adjusted_r2(r2, len(Y), 1)
    
    results.append({
        'Model': f'Y vs {var_name}',
        'R²': r2,
        'Adjusted R²': adj_r2,
        'MSE': mse,
        'Slope': model.coef_[0],
        'Intercept': model.intercept_
    })
    
    print(f"\nModel {i+1}: Y vs {var_name}")
    print(f"Linear Model: Y = {model.coef_[0]:.2f} * {var_name} + {model.intercept_:.2f}")
    print(f"Slope: {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"Adjusted R²: {adj_r2:.4f}")
    print(f"MSE: {mse:.2f}")
    
    # Generate and save plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_var, Y, color='blue', label='Data points')
    plt.plot(X_var, y_pred, color='red', label='Regression line')
    plt.xlabel(var_name)
    plt.ylabel('Y')
    plt.title(f'Simple Linear Regression: Y vs {var_name}\nR² = {r2:.4f}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, f'regression_{var_name}.png')
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    # plt.show()  # Uncomment to display plots interactively
    plt.close()

model_multi = LinearRegression()
model_multi.fit(X, Y)
y_pred_multi = model_multi.predict(X)

r2_multi = model_multi.score(X, Y)
mse_multi = mean_squared_error(Y, y_pred_multi)
adj_r2_multi = adjusted_r2(r2_multi, len(Y), X.shape[1])

results.append({
    'Model': 'Y vs X1,X2,X3,X4,X5,X6',
    'R²': r2_multi,
    'Adjusted R²': adj_r2_multi,
    'MSE': mse_multi,
    'Slope': model_multi.coef_,
    'Intercept': model_multi.intercept_
})

print(f"\nModel 7: Y vs X1,X2,X3,X4,X5,X6")
print(f"Multiple Linear Model: Y = {model_multi.intercept_:.2f} + "
      f"{model_multi.coef_[0]:.2f}*X1 + {model_multi.coef_[1]:.2f}*X2 + "
      f"{model_multi.coef_[2]:.2f}*X3 + {model_multi.coef_[3]:.2f}*X4 + "
      f"{model_multi.coef_[4]:.2f}*X5 + {model_multi.coef_[5]:.2f}*X6")
print(f"Intercept: {model_multi.intercept_:.2f}")
print(f"Coefficients: X1={model_multi.coef_[0]:.2f}, X2={model_multi.coef_[1]:.2f}, "
      f"X3={model_multi.coef_[2]:.2f}, X4={model_multi.coef_[3]:.2f}, "
      f"X5={model_multi.coef_[4]:.2f}, X6={model_multi.coef_[5]:.2f}")
print(f"R²: {r2_multi:.4f}")
print(f"Adjusted R²: {adj_r2_multi:.4f}")
print(f"MSE: {mse_multi:.2f}")

# Generate and save plot for multiple regression
plt.figure(figsize=(8, 6))
plt.scatter(Y, y_pred_multi, color='blue', label='Predicted vs Actual')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', linestyle='--', label='Ideal fit')
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title(f'Multiple Linear Regression: Predicted vs Actual\nR² = {r2_multi:.4f}')
plt.legend()
plt.grid(True)
plot_path = os.path.join(output_dir, 'regression_multiple.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
# plt.show()  # Uncomment to display plots interactively
plt.close()

# Question (b): Hypothesis Testing
print("\nQuestion (b): Hypothesis Testing")
print("=" * 50)
for i, (X_var, var_name) in enumerate(zip([X1, X2, X3, X4, X5, X6], variables)):
    X_var_sm = sm.add_constant(X_var)
    model_sm = sm.OLS(Y, X_var_sm).fit()
    
    p_value = model_sm.pvalues[1]
    
    print(f"\nModel {i+1}: Y vs {var_name}")
    print(f"H₀: β_{var_name} = 0 (Predictor {var_name} has no effect on Y)")
    print(f"H₁: β_{var_name} ≠ 0 (Predictor {var_name} has a significant effect on Y)")
    print(f"P-value for {var_name}: {p_value:.4f}")
    print(f"Conclusion: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'} "
          f"(α = 0.05)")

X_sm = sm.add_constant(X)
model_multi_sm = sm.OLS(Y, X_sm).fit()
p_values_multi = model_multi_sm.pvalues[1:]

print(f"\nModel 7: Y vs X1,X2,X3,X4,X5,X6")
for var, p_val in zip(variables, p_values_multi):
    print(f"H₀: β_{var} = 0 (Predictor {var} has no effect on Y)")
    print(f"H₁: β_{var} ≠ 0 (Predictor {var} has a significant effect on Y)")
    print(f"P-value for {var}: {p_val:.4f}")
    print(f"Conclusion: {'Reject H₀' if p_val < 0.05 else 'Fail to reject H₀'} "
          f"(α = 0.05)")

# Question (c): Model Comparison
print("\nQuestion (c): Model Comparison")
print("=" * 50)
print(f"{'Model':<20} {'R²':<8} {'Adjusted R²':<12} {'MSE':<8}")
print("-" * 50)
for res in results:
    print(f"{res['Model']:<20} {res['R²']:.4f} {res['Adjusted R²']:.4f} {res['MSE']:.2f}")
print("-" * 50)

best_model = max(results, key=lambda x: x['Adjusted R²'])
print(f"\nBest Model: {best_model['Model']}")
print(f"Why: The model has the highest Adjusted R² ({best_model['Adjusted R²']:.4f}), "
      f"indicating it explains the most variance in Y while accounting for the number of predictors.")
print(f"How: This model achieves a better balance of explanatory power (R² = {best_model['R²']:.4f}) "
      f"and prediction accuracy (MSE = {best_model['MSE']:.2f}) compared to others.")
print("Additional Insights:")
print("- Simple linear models with low R² suggest weak linear relationships.")
print("- The multiple regression model may include insignificant predictors (check p-values from Question b).")
print("- Adjusted R² is preferred for model selection as it penalizes unnecessary predictors.")