import pandas as pd
import matplotlib.pyplot as plt
import seaborn

data = pd.read_csv("my_proj/ExerciseM1.01/data/penguins_classification.csv")

for i in data.columns:
    print(data[i].value_counts())

#hist = data.hist(figsize=(10, 10))
seaborn.pairplot(data, hue="Species") 
plt.show()