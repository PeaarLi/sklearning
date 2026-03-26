import pandas as pd
from sklearn.compose import make_column_selector,make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_validate,RandomizedSearchCV
from scipy.stats import loguniform

class loguniform_int:
    """Integer valued version of the log-uniform distribution"""

    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)
def shorten_param(param_name):
        if "__" in param_name:
            return param_name.rsplit("__", 1)[1]
        return param_name
def main():
    data_raw = pd.read_csv("my_proj/Courses/VisualizeParams/data/adult-census.csv")
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

    param_distributions = {
        "classifier__l2_regularization": loguniform(1e-6, 1e3),
        "classifier__learning_rate": loguniform(0.001, 10),
        "classifier__max_leaf_nodes": loguniform_int(2, 256),
        "classifier__min_samples_leaf": loguniform_int(1, 100),
        "classifier__max_bins": loguniform_int(2, 255),
    }

    model_random_search = RandomizedSearchCV(
        model, 
        param_distributions=param_distributions,
        n_iter=30,
        cv=5,
        verbose=1
    )

    model_random_search.fit(data_train, target_train)

    # get the parameter names
    column_results = [f"param_{name}" for name in param_distributions.keys()]
    column_results += ["mean_test_score", "std_test_score", "rank_test_score"]

    cv_results = pd.DataFrame(model_random_search.cv_results_)
    cv_results = cv_results[column_results].sort_values(
        "mean_test_score", ascending=False
    )

    cv_results = cv_results.rename(shorten_param, axis=1)
    print(cv_results)

if __name__ == "__main__":
     main()