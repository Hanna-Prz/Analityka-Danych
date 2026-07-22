import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures,  StandardScaler
house_df = pd.read_csv('kc_house_data.csv')
# Exercise 1
#a) Create and train simple regression model of sqft_living to price.  
#b) Evaluate and visualise created model.  
#c) Create and plot higher order polynomial regression models.
#a)
X = house_df.sqft_living.values.reshape(-1,1)[:1000]
y = house_df.price.values.reshape(-1)[:1000]
model = LinearRegression()
model
model.fit(X,y)
#b)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)
print("R² Score:", model.score(X, y))
plt.scatter(X, y, color='blue', alpha=1)
plt.plot(X, y_pred, color='red')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.title('Linear Regression: sqft_living vs price')
plt.show()
#b)
for order in range(1, 9):
    poly = PolynomialFeatures(order, include_bias=False)
    X_order = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_order, y)
    X_min, X_max = X.min(), X.max()
    X_range = np.linspace(X_min, X_max, 1000).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_poly_pred = poly_model.predict(X_range_poly)
    mse = mean_squared_error(y, poly_model.predict(X_order))
    r2 = poly_model.score(X_order, y)
    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, color='lightblue', alpha=0.5, label='Actual data')
    plt.plot(X_range, y_poly_pred, color='red', linewidth=2, label=f'Order {order}')
    plt.xlabel('sqft_living')
    plt.ylabel('price')
    plt.title(f'Polynomial Regression (Order {order})\nMSE={mse:.2e}, R²={r2:.3f}')
    plt.legend()
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
# Exercise 2
#In reality, we are not given the full dataset, so we need to create models which are able to handle unknown data.  
# Split your data into training sample and test sample and repeat exercise. Plot training and test errors for all models.
train_errors = []
test_errors = []
for order in range(1,9):
    X_train, X_test, y_train, y_test = train_test_split(X_order, y, test_size=0.3, random_state=44)
    poly = PolynomialFeatures(order, include_bias=False)
    X_order = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_order, y, test_size=0.3, random_state=44)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))
plt.figure(figsize=(8,5))
plt.plot(range(1, 9), train_errors, marker='o', label='Train MSE')
plt.plot(range(1, 9), test_errors, marker='o', label='Test MSE')
plt.xlabel('Polynomial Order')
plt.ylabel('Mean Squared Error')
plt.title('Train and Test Errors vs Polynomial Order')
plt.xticks(range(1, 9))
plt.legend()
plt.show()
#Exercise 3
#a) Train model to predict sqft_living for a given house price. Plot predictions.  
#b) Apply more features and propose better model for predicting house prices. Try to beat proposed one.
#a)
X = house_df.price.values.reshape(-1,1)[:1000]
y = house_df.sqft_living.values.reshape(-1)[:1000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual sqft_living')
plt.ylabel('Predicted sqft_living')
plt.title('Prediction of sqft_living from price')
plt.show()
print("MSE:", mean_squared_error(y_test, y_pred))
X = house_df[['sqft_living', 'sqft_lot', 'grade', 'view']].values.reshape(-1,4)
y = house_df.price.values.reshape(-1)
poly = PolynomialFeatures(3, include_bias = False)
X = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=43)
model = LinearRegression()
_= model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean squared error equals: {0}".format(mean_squared_error(y_pred, y_test)))
X = house_df[['sqft_living', 'sqft_lot', 'grade', 'view']].values
y = house_df['price'].values
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)
print("Number of features after polynomial transformation:", X_poly.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.5, random_state=43)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error equals: {mse}")
features = [
    "YearsCode", "WorkExp", "Employment", "MainBranch", "Age", "EdLevel","DevType","OrgSize","Country"
]
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
cat_cols = [col for col in available_features if df_model[col].dtype == 'object']
df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
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


#Hint: model.predict(), model.intercept_,  model.coef_, model.score()
