import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import time
#数据预处理

data = pd.read_csv("my_proj/NumerialDataScaling/data/adult-census.csv")

num_columns = ["age","capital-gain","capital-loss","hours-per-week"]
numerial_data = data[num_columns]
target = data["class"]

data_train, data_test, target_train, target_test = train_test_split(numerial_data, target, test_size=0.2, random_state=325)
scaler = StandardScaler().set_output(transform="pandas")
data_train_scaled = scaler.fit_transform(data_train)

#可视化
def Jplot(data_train,data_train_scaled,num):
    sns.jointplot(
        data=data_train[:num],
        x="age",
        y="hours-per-week",
        marginal_kws=dict(bins=15),
    )
    plt.suptitle("Before", y=1.1)
    sns.jointplot(
        data=data_train_scaled[:num],
        x="age",
        y="hours-per-week",
        marginal_kws=dict(bins=15)
    )
    plt.suptitle("After", y=1.1)
    plt.show()
def main():
    #Jplot(data_train,data_train_scaled,500)
    model = make_pipeline(StandardScaler(),LogisticRegression())

    start_time = time.time()
    model.fit(data_train,target_train)
    elapse_time = time.time() - start_time
    print("训练用时:",elapse_time)

    start_time = time.time()
    target_predicted = model.predict(data_test)
    elapse_time = time.time() - start_time
    print("预测用时",elapse_time)

    print("准确率：",model.score(data_test,target_test))
    print("迭代次数",model[-1].n_iter_[0])

if __name__ == "__main__":
    main()