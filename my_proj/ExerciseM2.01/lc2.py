import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, ValidationCurveDisplay, LearningCurveDisplay
import joblib  # ← 新增导入
import time

def main():
    # 使用绝对路径加载数据
    start_time = time.time()
    N_jobs = -1

    data_path = "my_proj/ExerciseM2.01/data/blood_transfusion.csv"
    data_raw = pd.read_csv(data_path)
    
    target_name = "Class"
    target = data_raw[target_name]
    train = data_raw.drop(columns=target_name)

    preprocessor = Pipeline(
        steps=[("Scaler", StandardScaler())]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("Regressor", SVC(kernel="rbf"))
        ]
    )

    cv_generator = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)

    gammas = np.linspace(10e-3, 10e2, num=30, endpoint=True)
    
    # 使用 threading 后端进行并行
    with joblib.parallel_backend('threading', n_jobs=N_jobs):
        display1 = ValidationCurveDisplay.from_estimator(
            model, train, target,
            param_name="Regressor__gamma",
            param_range=gammas,
            cv=cv_generator,
            scoring="accuracy",
            score_name="Accuracy",
            std_display_style="errorbar",
            errorbar_kw={"alpha": 0.7},
            # n_jobs 不再需要（由 parallel_backend 控制）
        )

    display1.ax_.set(xscale="log", title="Validation Curve For SVC")

    train_size = np.linspace(0.1, 1, num=10, endpoint=True)
    
    with joblib.parallel_backend('threading', n_jobs=N_jobs):
        display2 = LearningCurveDisplay.from_estimator(
            model, train, target,
            train_sizes=train_size,
            cv=cv_generator,
            score_type="both",
            scoring="accuracy",
            score_name="Accuracy",
            std_display_style="errorbar",
            errorbar_kw={"alpha": 0.7},
            # n_jobs 不再需要
        )

    display2.ax_.set(title="Learning Curve For SVC")
    print(time.time()-start_time)
    plt.show()

if __name__ == '__main__':
    main()