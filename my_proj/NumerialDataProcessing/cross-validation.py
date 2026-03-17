from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pandas as pd

data = pd.read_csv("my_proj/NumerialDataProcessing/data/adult-census.csv")

num_columns = ["age","capital-gain","capital-loss","hours-per-week"]
numerial_data = data[num_columns]
target = data["class"]

model = make_pipeline(StandardScaler(),LogisticRegression())
cv_result = cross_validate(model,numerial_data,target,cv=5)

print(f"Test Score:{cv_result["test_score"].mean():.3f}±{cv_result["test_score"].std():.3f}")