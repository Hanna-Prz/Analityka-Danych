import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import scipy.stats as stats
data = pd.read_csv(
    'survey_results_public.csv',
    engine='python',
    on_bad_lines='skip',
    encoding_errors='replace'
)
df = data[["YearsCode", "ConvertedCompYearly"]].copy()
df["YearsCode"] = df["YearsCode"].replace({
    "Less than 1 year": 0.5,
    "More than 50 years": 50
})
df = df.apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)
df = df[(df["ConvertedCompYearly"] > 0) & (df["ConvertedCompYearly"] < 500000)]
X = df[["YearsCode"]]
y = df["ConvertedCompYearly"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
print("Intercept:", lin_reg.intercept_)
print("Coef:", lin_reg.coef_)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression — MSE: {mse:.2f}, R²: {r2:.4f}")
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color="gray", alpha=0.6, label="Actual values")
plt.plot(X_test, y_pred, color="blue", linewidth=2, label="Predicted regression line")
plt.title("Linear Regression: Years of Coding Experience vs. Annual Compensation")
plt.xlabel("Years of Coding Experience")
plt.ylabel("Annual Compensation (USD)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
degrees = range(1, 8)
for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    poly_reg = LinearRegression()
    poly_reg.fit(X_train_p, y_train_p)
    y_pred_p = poly_reg.predict(X_test_p)
    mse_p = mean_squared_error(y_test_p, y_pred_p)
    r2_p = r2_score(y_test_p, y_pred_p)
    print(f"Polynomial Regression (degree={degree}) — MSE: {mse_p:.2f}, R²: {r2_p:.4f}")
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color="lightgray", alpha=0.5, label="Actual data")
    X_sorted = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_poly_pred = poly_reg.predict(poly.transform(X_sorted))
    plt.plot(X_sorted, y_poly_pred, color="red", linewidth=2, label=f"Degree {degree}")
    plt.title(f"Polynomial Regression (degree={degree})")
    plt.xlabel("Years of Coding Experience")
    plt.ylabel("Annual Compensation (USD)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
    residuals = y_test_p - y_pred_p  # Upewniamy się, że to *te same* dane
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(y_pred_p, residuals, alpha=0.5, color='blue')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residuals vs Predicted (degree={degree})')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.subplot(1,2,2)
    plt.hist(residuals, bins=30, color='gray', alpha=0.7)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(5,5))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'QQ Plot of Residuals (degree={degree})')
    plt.show()
degrees = range(1, 9)
mse_list = []
for order in degrees:
    poly = PolynomialFeatures(order, include_bias=False)
    X_order = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_order, y)
    y_pred = poly_model.predict(X_order)
    mse_list.append(mean_squared_error(y, y_pred))
plt.figure(figsize=(8,5))
plt.plot(degrees, mse_list, marker='o', linestyle='-', color='blue')
plt.xlabel('degree')
plt.ylabel('error')
plt.title('Models performance')
plt.xticks(degrees)
plt.grid(True)
plt.show()
train_errors = []
test_errors = []
for order in range(1, 9):
    poly = PolynomialFeatures(order, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=44)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))
plt.figure(figsize=(8, 5))
plt.plot(range(1, 9), train_errors, marker='o', label='Train MSE')
plt.plot(range(1, 9), test_errors, marker='o', label='Test MSE')
plt.xlabel('Polynomial Order')
plt.ylabel('Mean Squared Error')
plt.title('Train and Test Errors vs Polynomial Order')
plt.xticks(range(1, 9))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
X = df[["YearsCode"]]
y = df["ConvertedCompYearly"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
y_pred_simple = lin_reg.predict(X)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Simple Linear Regression — MSE: {mse:.2f}, R²: {r2:.4f}")
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual values')
plt.plot(X, y_pred_simple, color='red', linewidth=2, label='Linear regression')
plt.xlabel("Years of Coding Experience")
plt.ylabel("Annual Compensation (USD)")
plt.title("Simple Linear Regression: YearsCode → ConvertedCompYearly")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
features = ["YearsCode", "WorkExp", "ToolCountWork"]
available_features = [col for col in features if col in data.columns]
df_model = data[available_features + ["ConvertedCompYearly"]].copy()
df_model["YearsCode"] = df_model["YearsCode"].replace({
    "Less than 1 year": 0.5,
    "More than 50 years": 50
})
df_model = df_model[
    (df_model["ConvertedCompYearly"] > 5000) &
    (df_model["ConvertedCompYearly"] < 500000)
]
df_model.dropna(inplace=True)
df_encoded = pd.get_dummies(
    df_model,
    columns=[col for col in available_features if df_model[col].dtype == 'object'],
    drop_first=True
)
X = df_encoded.drop("ConvertedCompYearly", axis=1)
y = df_encoded["ConvertedCompYearly"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Improved model — MSE: {mse:.2f}, R²: {r2:.4f}")
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.4, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Compensation")
plt.ylabel("Predicted Compensation")
plt.title("Improved Linear Regression Model — Actual vs Predicted")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
coeffs = pd.Series(lin_reg.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print("\nMost influential features:")
display(coeffs.head(10))
#Exercise 3 (extended)
#a) Predict YearsCode based on Compensation
X_rev = df[['ConvertedCompYearly']]
y_rev = df['YearsCode']
X_train, X_test, y_train, y_test = train_test_split(X_rev, y_rev, test_size=0.2, random_state=42)
rev_model = LinearRegression()
rev_model.fit(X_train, y_train)
y_pred_rev = rev_model.predict(X_test)
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, alpha=0.4, color='gray')
plt.plot(X_test, y_pred_rev, color='red')
plt.xlabel('Annual Compensation (USD)')
plt.ylabel('Years of Coding Experience')
plt.title('Predicting YearsCode from Compensation')
plt.show()



print("R^2 score:", lin_reg.score(X_test, y_test))
