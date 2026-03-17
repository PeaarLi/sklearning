#加载数据集
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X = X[:,[2]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, shuffle=False)

#训练模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(X_train, y_train)

#评估模型
from sklearn.metrics import mean_squared_error , r2_score
y_pred = regressor.predict(X_test)
print("----------------------------------评估模型-------------------------------------")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2: {r2_score(y_test, y_pred):.2f}")
print("------------------------------------------------------------------------------")

#可视化
from matplotlib import pyplot as plt

fig ,ax = plt.subplots(ncols=2, figsize=(10,5), sharey=True, sharex=True)
ax[0].scatter(X_train, y_train, alpha=0.5, label="Training Data")
ax[0].plot(
    X_train,
    regressor.predict(X_train),
    color="red",
    label="Prediction",
    linewidth=3,
)
ax[0].set(xlabel="Feature", ylabel="Target", title="Training Data and Prediction")
ax[0].legend()

ax[1].scatter(X_test, y_test, alpha=0.5, label="Test Data")
ax[1].plot(
    X_test,
    y_pred,
    color="red",
    label="Prediction",
    linewidth=3,
)
ax[1].set(xlabel="Feature", ylabel="Target", title="Test Data and Prediction")
ax[1].legend()

fig.suptitle("Linear Regression Model")
plt.show()