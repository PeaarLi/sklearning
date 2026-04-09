import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_moons,make_gaussian_quantiles
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer,SplineTransformer,PolynomialFeatures,StandardScaler
from sklearn.kernel_approximation import Nystroem

def plot_datasets(datasets, *, feature_names=None, subtitles=None, title=None):
    n = len(datasets)
    fig, axs = plt.subplots(ncols=n, figsize=(8*n, 8), constrained_layout=True)

    common_scatter_plot_params = dict(
        cmap=ListedColormap(["tab:red", "tab:blue"]),
        edgecolor="white",
        linewidth=1,
    )

    for i, ax, (data, target) in zip(
        range(len(datasets)),
        axs,
        datasets,
    ):
        ax.scatter(
            data.iloc[:, 0],
            data.iloc[:, 1],
            c=target,
            s=100,
            **common_scatter_plot_params,
        )
        if feature_names is not None:
            ax.set_xlabel(feature_names[0])
            if i == 0:
                ax.set_ylabel(feature_names[1])
            else:
                ax.set_ylabel(None)
        if subtitles is not None:
            ax.set_title(subtitles[i])
    if title is not None:
        fig.suptitle(title)
    return fig

def plot_decision_boundary(model, datasets, title=None):
    n = len(datasets)
    common_scatter_plot_params = dict(
        cmap=ListedColormap(["tab:red", "tab:blue"]),
        edgecolor="white",
        linewidth=1,
    )

    fig, axs = plt.subplots(
        ncols=n,
        figsize=(8*n, 8),
        constrained_layout=True,
    )

    for i, ax, (data, target) in zip(
        range(len(datasets)),
        axs,
        datasets,
    ):
        model.fit(data, target)
        DecisionBoundaryDisplay.from_estimator(
            model,
            data,
            response_method="predict_proba",
            plot_method="pcolormesh",
            cmap="RdBu",
            alpha=0.8,
            # Setting vmin and vmax to the extreme values of the probability to
            # ensure that 0.5 is mapped to white (the middle) of the blue-red
            # colormap.
            vmin=0,
            vmax=1,
            ax=ax,
        )
        DecisionBoundaryDisplay.from_estimator(
            model,
            data,
            response_method="predict_proba",
            plot_method="contour",
            alpha=0.8,
            levels=[0.5],  # 0.5 probability contour line
            linestyles="--",
            linewidths=2,
            ax=ax,
        )
        ax.scatter(
            data.iloc[:, 0],
            data.iloc[:, 1],
            c=target,
            s=100,
            **common_scatter_plot_params,
        )
        if i > 0:
            ax.set_ylabel(None)
    if title is not None:
        fig.suptitle(title, fontsize=16)

    return fig

def main():
    feature_names = ["Feature #0", "Feature #1"]
    target_name = "class"

    # Moons dataset
    X, y = make_moons(n_samples=100, noise=0.13, random_state=42)
    moons = pd.DataFrame(
        np.concatenate([X, y[:, np.newaxis]], axis=1),
        columns=feature_names + [target_name],
    )
    data_moons, target_moons = moons[feature_names], moons[target_name]

    # Gaussian quantiles dataset
    X, y = make_gaussian_quantiles(
        n_samples=100, n_features=2, n_classes=2, random_state=42
    )
    gauss = pd.DataFrame(
        np.concatenate([X, y[:, np.newaxis]], axis=1),
        columns=feature_names + [target_name],
    )
    data_gauss, target_gauss = gauss[feature_names], gauss[target_name]

    # XOR dataset
    xor = pd.DataFrame(
        np.random.RandomState(0).uniform(low=-1, high=1, size=(200, 2)),
        columns=feature_names,
    )
    target_xor = np.logical_xor(xor["Feature #0"] > 0, xor["Feature #1"] > 0)
    target_xor = target_xor.astype(np.int32)
    xor["class"] = target_xor
    data_xor = xor[feature_names]
    target_xor = xor[target_name]

    datasets = [
        (data_moons, target_moons),
        (data_gauss, target_gauss),
        (data_xor, target_xor),
    ]
    subtitles = [
        "Moon dataset",
        "Gaussian quantiles dataset",
        "XOR dataset",
    ]

    #plot_datasets(datasets, feature_names=feature_names, subtitles=subtitles, title="Datasets")
    
    fig1 = plot_decision_boundary(
        model=LogisticRegression(max_iter=1000),
        datasets=datasets,
        title="Logistic regression decision boundaries",
    )
    
    KBins_classifer = Pipeline(
        steps=[
            ("preprocessor", KBinsDiscretizer(n_bins=5, encode="onehot")),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    fig2 = plot_decision_boundary(
        model=KBins_classifer,
        datasets=datasets,
        title="Logistic regression decision boundaries with binning",
    )
    
    
    Spline_classifer = Pipeline(
        steps=[
            ("preprocessor", SplineTransformer(n_knots=5, degree=3)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    fig3 = plot_decision_boundary(
        model=Spline_classifer,
        datasets=datasets,
        title="Logistic regression decision boundaries with spline",
    )
    
    Poly_classifer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("preprocessor", PolynomialFeatures(degree=3,include_bias=False)),
            ("classifier", LogisticRegression(C=10, max_iter=1000)),
        ]
    )
    fig4 = plot_decision_boundary(
        model=Poly_classifer,
        datasets=datasets,
        title="Logistic regression decision boundaries with polynomial",
    )

    Nystroem_poly_classifer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("preprocessor", Nystroem(kernel="poly", degree=3, coef0=1, n_components=100)),
            ("classifier", LogisticRegression(C=10, max_iter=1000)),
        ]
    )
    fig5 = plot_decision_boundary(
        model=Nystroem_poly_classifer,
        datasets=datasets,
        title="Logistic regression decision boundaries with Nystroem(Polynomial)",
    )

    Nystroem_rbf_classifer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("preprocessor", Nystroem(kernel="rbf", gamma=1, n_components=100)),
            ("classifier", LogisticRegression(C=10, max_iter=1000)),
        ]
    )
    fig6 = plot_decision_boundary(
        model=Nystroem_rbf_classifer,
        datasets=datasets,
        title="Logistic regression decision boundaries with Nystroem(RBF)",
    )

    Kbins_Nystroem_rbf_classifer = Pipeline(
        steps=[
            ("preprocessor", KBinsDiscretizer(n_bins=5)),
            ("preprocessor2", Nystroem(kernel="rbf", gamma=1.0, n_components=100)),
            ("classifier", LogisticRegression(C=10, max_iter=1000)),
        ]
    )
    fig7 = plot_decision_boundary(
        model=Kbins_Nystroem_rbf_classifer,
        datasets=datasets,
        title="Logistic regression decision boundaries with KBins+Nystroem(RBF)",
    )

    Spline_Nystroem_rbf_classifer = Pipeline(
        steps=[
            ("preprocessor", SplineTransformer(n_knots=5, degree=3)),
            ("preprocessor2", Nystroem(kernel="rbf", gamma=1.0, n_components=100)),
            ("classifier", LogisticRegression(C=10, max_iter=1000)),
        ]
    )
    fig8 = plot_decision_boundary(
        model=Spline_Nystroem_rbf_classifer,
        datasets=datasets,
        title="Logistic regression decision boundaries with Spline+Nystroem(RBF)",
    )
    
    #plt.close(fig1)
    #plt.close(fig2)
    #plt.close(fig3)
    #plt.close(fig4)
    #plt.close(fig5)
    #plt.close(fig6)
    #plt.close(fig7)
    #plt.close(fig8)
    plt.show()

    output_folder = "my_proj/Courses/Non-linear feature engineering for Logistic Regression/OUTPUT/"
    fig1.savefig(output_folder + "Logistic_regression_decision_boundaries.png")
    fig2.savefig(output_folder + "Logistic_regression_decision_boundaries_with_binning.png")
    fig3.savefig(output_folder + "Logistic_regression_decision_boundaries_with_spline.png")
    fig4.savefig(output_folder + "Logistic_regression_decision_boundaries_with_polynomial.png")
    fig5.savefig(output_folder + "Logistic_regression_decision_boundaries_with_Nystroem(Polynomial).png")
    fig6.savefig(output_folder + "Logistic_regression_decision_boundaries_with_Nystroem(RBF).png")
    fig7.savefig(output_folder + "Logistic_regression_decision_boundaries_with_KBins+Nystroem(RBF).png")
    fig8.savefig(output_folder + "Logistic_regression_decision_boundaries_with_Spline+Nystroem(RBF).png")
    
if __name__ == "__main__":
    main()