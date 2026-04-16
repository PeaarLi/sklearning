import pathlib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

def plot_cv_coefs(cv_result:dict, process_name:str) -> None:
    feature_names = cv_result['estimator'][0].named_steps[process_name].feature_names_in_
    
def main():
    script_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = script_dir.parent / "data" / "ames_housing_no_missing.csv"
    data_raw = pd.read_csv(data_dir)
    
    target_name = "SalePrice"
    features_of_interest = [
        "LotFrontage",
        "LotArea",
        "PoolArea",
        "YearBuilt",
        "YrSold",
    ]
    data = data_raw[features_of_interest]
    target = data_raw[target_name]

    model = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("logistic_regression", LogisticRegression(max_iter=1000)),
        ]
    )

    cv_results = cross_validate(
        model, data, target, 
        cv=10, 
        scoring="neg_mean_squared_error",
        return_train_score=True,
        return_estimator=True
        )
    
    train_scores = -cv_results["train_score"]
    test_scores = -cv_results["test_score"]

    print(f"Train score: {train_scores.mean():.3f} ± {train_scores.std():.3f}")
    print(f"Test score: {test_scores.mean():.3f} ± {test_scores.std():.3f}")


if __name__ == "__main__":
    main()