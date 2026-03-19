import pandas as pd
from sklearn.compose import make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier

data = pd.read_csv("my_proj/ExerciseM1.03/data/adult-census.csv")
target_name = "class"
target = data[target_name]
num_selecter = make_column_selector(dtype_include="int64")
num_data = data.drop(columns=target_name)[num_selecter(data)]

model = make_pipeline(StandardScaler(),DummyClassifier())

cv_result = cross_validate(model,num_data,target,cv=5)

print(cv_result)