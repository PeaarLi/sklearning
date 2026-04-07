import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ValidationCurveDisplay
import joblib

data_raw = pd.read_csv("my_proj/Exercises/ExerciseM4.02/data/penguins.csv")
target_name = "Body Mass (g)"
columns = ["Flipper Length (mm)", "Culmen Length (mm)", "Culmen Depth (mm)"]
data_raw = data_raw.dropna(subset=columns + [target_name])
target = data_raw[target_name]
data = data_raw[columns]

model1 = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
    LinearRegression()
)

#scores = cross_val_score(model1, data, target, cv=10 ,scoring="neg_mean_absolute_error")
#print(f"The mean accuracy is: {scores.mean():.2f} +/- {scores.std():.2f}")

param_range = np.array([1, 2, 5, 10, 20, 50, 100, 200])
model2 = make_pipeline(
    Nystroem(kernel="poly", degree=2, random_state=0),
    LinearRegression()
)
with joblib.parallel_backend("threading", n_jobs=-1):
    disp = ValidationCurveDisplay.from_estimator(
        model2,
        data,
        target,
        cv=10,
        param_name="nystroem__n_components",
        param_range=param_range,
        scoring="neg_mean_absolute_error",
    )

disp.ax_.set(
    xlabel="Number of components",
    ylabel="Mean absolute error (k$)",
    title="Validation curve for Nystroem",
)
plt.show()