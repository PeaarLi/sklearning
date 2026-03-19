import pandas as pd
from sklearn.compose import make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

def print_cv(cv_result:dict):
    print(f"{"fit time":<12}{cv_result["fit_time"].mean():.3f}±{cv_result["fit_time"].std():.3f}")
    print(f"{"score time":<12}{cv_result["score_time"].mean():.3f}±{cv_result["score_time"].std():.3f}")
    print(f"{"score":<12}{cv_result["test_score"].mean():.3f}±{cv_result["test_score"].std():.3f}")

data = pd.read_csv("my_proj/ExerciseM1.04/data/adult-census.csv")
target_name = "class"
target = data[target_name]
cat_selecter = make_column_selector(dtype_include="object")
cat_data = data[cat_selecter(data)].drop(columns=target_name)

model1 = make_pipeline(
    OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),
    LogisticRegression(max_iter=500)
    )
cv_results1 = cross_validate(model1,cat_data,target)
model2 = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    LogisticRegression(max_iter=500)
)
cv_results2 = cross_validate(model2,cat_data,target)
print("OrigianlEnoder:")
print_cv(cv_results1) 
print("OneHotEncoder:")
print_cv(cv_results2)