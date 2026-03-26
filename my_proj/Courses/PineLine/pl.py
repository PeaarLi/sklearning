import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

raw_data = pd.read_csv("my_proj/PineLine/data/house_prices.csv", na_values="?")
target_name = "SalePrice"
target = raw_data[target_name]
target = (target > 200_000).astype(int)

numeric_features = ["LotArea", "FullBath", "HalfBath"]
categorical_features = ["Neighborhood", "HouseStyle"]
train = raw_data[numeric_features + categorical_features]

num_transformer = Pipeline(
    steps=[
        ("I mputer",SimpleImputer(strategy="median")),
        ("Scaler",StandardScaler())
    ]
)
cat_transformer = Pipeline(
    steps=[
        ("OneHotEncoder",OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numeric_features),
        ("cat", cat_transformer, categorical_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression())
    ]
)
def print_cv(cv_result:dict):
    print(f"{"fit time":<12}{cv_result["fit_time"].mean():.3f}±{cv_result["fit_time"].std():.3f}")
    print(f"{"score time":<12}{cv_result["score_time"].mean():.3f}±{cv_result["score_time"].std():.3f}")
    print(f"{"score":<12}{cv_result["test_score"].mean():.3f}±{cv_result["test_score"].std():.3f}")

cv_results = cross_validate(model, train, target)
print_cv(cv_results)