import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

raw_data = pd.read_csv("my_proj/ExerciseM1.02/data/adult-census-numeric.csv")
data = raw_data.drop(columns="class")
target = raw_data["class"]

model = KNeighborsClassifier(n_neighbors=50)
model.fit(data, target)

target_predicted = model.predict(data)

for i in range(10):
    print(
        f"raw: {target[i]} predicted: {target_predicted[i]} {target[i]==target_predicted[i]}"
    )

print(
    "Number of correct prediction: "
    f"{(target[:10] == target_predicted[:10]).sum()} / 10"
)

print(
    "Accuracy:",
    (target == target_predicted).mean(),
)