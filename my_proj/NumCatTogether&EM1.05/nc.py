import pandas as pd
from sklearn.compose import make_column_transformer,make_column_selector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate

def print_cv(cv_result:dict):
    print(f"{"fit time":<12}{cv_result["fit_time"].mean():.3f}±{cv_result["fit_time"].std():.3f}")
    print(f"{"score time":<12}{cv_result["score_time"].mean():.3f}±{cv_result["score_time"].std():.3f}")
    print(f"{"score":<12}{cv_result["test_score"].mean():.3f}±{cv_result["test_score"].std():.3f}")

raw_data = pd.read_csv("my_proj/NumCatTogether/data/adult-census.csv")
target_name = "class"
target = raw_data[target_name]
data = raw_data.drop(columns=target_name)

cat_selecter = make_column_selector(dtype_include="object")
cat_preprocesser =OrdinalEncoder(
    handle_unknown="use_encoded_value",unknown_value=-1
)

precesser = make_column_transformer(
    (cat_preprocesser,cat_selecter(data)),
    remainder="passthrough",
)

model = make_pipeline(
    precesser,HistGradientBoostingClassifier()
)

cv_results = cross_validate(model, data, target)
print_cv(cv_results)