import pandas as pd
import numpy as np
from sklearn.compose import make_column_selector,make_column_transformer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_validate

data_raw = pd.read_csv("my_proj/ExerciseM3.01/data/adult-census.csv")
target_name = "class"
target = data_raw[target_name]
data = data_raw.drop(columns=[target_name,"education-num"])

data_train, data_test, target_train, target_test = train_test_split(data, target, train_size=0.2,random_state=325)

preprocessor = make_column_transformer(
    (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), make_column_selector(dtype_include="object")),
    remainder="passthrough"
)

model = Pipeline(
    steps=[
        ("preprocessor",preprocessor),
        ("classifier",HistGradientBoostingClassifier(random_state=325))
    ]
)

learning_rates = [0.01, 0.1, 1, 10]
max_leaf_nodes = [3, 10, 30]

max_score = 0
best_lr = 0
best_mln = 0
print(f"{"Learning rate":<15}{"Max Leaf Rate":<15}{"score"}")
for lr in learning_rates:
    for mln in max_leaf_nodes:
        model.set_params(classifier__learning_rate=lr,classifier__max_leaf_nodes=mln)
        model.fit(data_train, target_train)
        score = model.score(data_test, target_test)
        print(f"{lr:<15}{mln:<15}{score:.3f}")
        if(score>max_score):
            max_score = score
            best_lr = lr
            best_mln = mln
print(f"Best param: Learning rate= {best_lr} ;Max Leaf Rate= {best_mln} ")

model.set_params(classifier__learning_rate=best_lr,classifier__max_leaf_nodes=best_mln)
cv_result = cross_validate(model, data, target)

print(f"{"fit time":<12}{cv_result["fit_time"].mean():.3f}±{cv_result["fit_time"].std():.3f}")
print(f"{"score time":<12}{cv_result["score_time"].mean():.3f}±{cv_result["score_time"].std():.3f}")
print(f"{"score":<12}{cv_result["test_score"].mean():.3f}±{cv_result["test_score"].std():.3f}")