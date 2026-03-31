import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import SplineTransformer
from sklearn.kernel_approximation import Nystroem

def generate_data(n_sample = 100):
    rng = np.random.RandomState(325)

    n_sample = n_sample
    data_max, data_min = 1.4, -1.4
    len_data = data_max - data_min

    data = np.sort(rng.rand(n_sample) * len_data - len_data / 2)
    noise = rng.randn(n_sample) * 0.3
    target = data**3 - 0.5 * data**2 + noise
    return data, target

def fit_score_plot_regression(model, data, target, ax, *, title=None):
    """在给定的 ax 上绘制模型拟合结果"""
    full_data = pd.DataFrame({"input_feature": data, "target": target})
    data = data.reshape(-1, 1)

    model.fit(data, target)
    target_predicted = model.predict(data)
    mse = mean_squared_error(target, target_predicted)

    # 在传入的 ax 上绘图
    sns.scatterplot(
        data=full_data, x="input_feature", y="target", color="black", alpha=0.5, ax=ax
    )
    ax.plot(data, target_predicted, color='blue')
    if title is not None:
        ax.set_title(title + f" (MSE = {mse:.2f})")
    else:
        ax.set_title(f"Mean squared error = {mse:.2f}")

def main():
    data, target = generate_data(100)
    
    tree = DecisionTreeRegressor(max_depth=3)
    polynomial_regression = make_pipeline(
        PolynomialFeatures(degree=3, include_bias=False),
        LinearRegression(),
    )
    linear_svr = SVR(kernel="linear")
    poly_svr = SVR(kernel="poly", degree=3)
    binned_regression = make_pipeline(
        KBinsDiscretizer(n_bins=8),
        LinearRegression(),
        )
    spline_regression = make_pipeline(
        SplineTransformer(degree=3, include_bias=False),
        LinearRegression(),
        )
    nystroem_regression = make_pipeline(
        Nystroem(kernel="poly", degree=3, n_components=5, random_state=0),
        LinearRegression(),
        )
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))  

    fit_score_plot_regression(LinearRegression(), data, target, ax=axes[0][0], title="Linear Regression")
    fit_score_plot_regression(tree, data, target, ax=axes[0][1], title="Decision Tree")
    fit_score_plot_regression(polynomial_regression, data, target, ax=axes[0][2], title="Polynomial Regression")
    fit_score_plot_regression(linear_svr, data, target, ax=axes[0][3], title="Linear Support Vector Machine")
    fit_score_plot_regression(poly_svr, data, target, ax=axes[1][0], title="Polynomial Support Vector Machine")
    fit_score_plot_regression(binned_regression, data, target, ax=axes[1][1], title="Binned Linear Regression")
    fit_score_plot_regression(spline_regression, data, target, ax=axes[1][2], title="Spline Regression")
    fit_score_plot_regression(nystroem_regression, data, target, ax=axes[1][3], title="Nystroem Approximated Polynomial Regression")

    plt.suptitle("Regression models comparison", fontsize=16)
    plt.tight_layout()  # 自动调整子图间距，避免重叠
    plt.show()

if __name__ == "__main__":
    main()