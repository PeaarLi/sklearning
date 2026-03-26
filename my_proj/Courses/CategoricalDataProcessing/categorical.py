import pandas as pd
from sklearn.compose import make_column_selector
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("my_proj/CategoricalDataProcessing/data/adult-census.csv")
data = data.drop(columns="education-num")
target = data["class"]
train = data.drop(columns="class")
cat_selecter = make_column_selector(dtype_include="object")
num_selecter = make_column_selector(dtype_include="int64")
cat_train = train[cat_selecter(train)]
num_train = train[num_selecter(train)]

def OEncode(cat_train:pd.DataFrame,col=None):
    if col==None:
        col = cat_train.columns[0]
    OEncoder = OrdinalEncoder().set_output(transform="pandas")
    cat_train_encoded = OEncoder.fit_transform(cat_train)
    orig = cat_train[col].rename(col+"-original")
    encoded = cat_train_encoded[col].rename(col+"-encoded")
    c = pd.concat([orig,encoded],axis=1)
    print(c.value_counts().sort_index())
    return cat_train_encoded

def main():
    #OEncode(cat_train,col=cat_train.columns[-1])
    model =  make_pipeline(OneHotEncoder(handle_unknown="ignore"),LogisticRegression(max_iter=500))
    cv_results = cross_validate(model,cat_train,target)
    print(cv_results)
    print(f"{cv_results["test_score"].mean():.3f}±{cv_results["test_score"].std():.3f}")

if __name__ == "__main__":
    main()