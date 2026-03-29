import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import joblib
def main():
    data,target = fetch_california_housing(return_X_y=True, as_frame=True)
    target = target*100
    data_train,data_test,target_train,target_test = train_test_split(data, target, train_size=0.8, random_state= 325)

    model = Pipeline(
        steps=[
            ("preprocessor",StandardScaler()),
            ("regressor",KNeighborsRegressor())
        ]
    )

    params = {
        "regressor__n_neighbors":np.logspace(0, 3, num=10).astype(np.int32),
        "preprocessor__with_mean":[True,False],
        "preprocessor__with_std":[True,False],
    }
    
    with joblib.parallel_backend("threading", n_jobs=-1):
        model_random_search = RandomizedSearchCV(
            model, 
            param_distributions=params, 
            cv=5, 
            n_iter=30, 
            scoring="neg_mean_absolute_error",
            verbose=1,
            random_state=1,
        )

    model_random_search.fit(data_train, target_train)
    print(model_random_search.best_params_)

    score = model_random_search.score(data_test, target_test)
    print(score)

if __name__=="__main__":
    main()