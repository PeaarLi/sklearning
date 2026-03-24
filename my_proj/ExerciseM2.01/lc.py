import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_validate,ValidationCurveDisplay,LearningCurveDisplay
import joblib
import time

def main():
    start_time = time.time()
    data_raw = pd.read_csv("my_proj/ExerciseM2.01/data/blood_transfusion.csv")
    target_name = "Class"
    target = data_raw[target_name]
    train = data_raw.drop(columns=target_name)

    preprocessor = Pipeline(
        steps=[("Scaler",StandardScaler())]
    )

    model = Pipeline(
        steps=[
            ("preprocessor",preprocessor),
            ("Regressor",SVC(kernel="rbf"))
        ]
    )

    cv_generator = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)
    """
    cv_results = cross_validate(
        estimator=model,
        X=train,
        y=target,
        scoring="neg_mean_absolute_error",
        cv=cv_generator,
        return_train_score=True,
    )
    """
    gammas=np.linspace(10e-3, 10e2, num=30, endpoint=True)
    display1 = ValidationCurveDisplay.from_estimator(
        model, train, target,
        param_name="Regressor__gamma",
        param_range=gammas,
        cv=cv_generator,
        scoring="accuracy",
        score_name="Accuracy",
        std_display_style="errorbar",
        errorbar_kw={"alpha": 0.7},
    )

    display1.ax_.set(xscale="log",title="ValidationCurve For SVC")

    train_size = np.linspace(0.1, 1, num=10, endpoint=True)
    display2 = LearningCurveDisplay.from_estimator(
        model, train, target,
        train_sizes=train_size,
        cv=cv_generator,
        score_type="both",
        scoring="accuracy",
        score_name="Accuracy",
        std_display_style="errorbar",
        errorbar_kw={"alpha": 0.7},
    )

    display2.ax_.set(title="LearningCurve For SVC")
    print(time.time()-start_time)
    plt.show()

if __name__ == "__main__":
    with joblib.parallel_backend(backend='threading', n_jobs=1):
        main()