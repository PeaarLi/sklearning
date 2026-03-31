import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from tqdm import tqdm

class LinearModel:
    def __init__(self, *,a=1, b=1):
        self.a = a
        self.b = b

    def predict(self, X):
        results = self.a*X +self.b
        return results
    
    def score(self, X, y_true):
        y_pred = self.predict(X)
        rmse = sqrt(((y_pred.values - y_true.values) ** 2).mean())
        return rmse
        
    def set_param(self, a, b):
        self.a = a
        self.b = b

def main():
    data_raw = pd.read_csv("my_proj/Exercises/ExerciseM4.01/data/penguins_regression.csv")
    target_name = "Body Mass (g)"
    target = data_raw[target_name]
    data = data_raw.drop(columns=target_name)

    model = LinearModel()
    min_rmse = float('inf')
    best_a=1
    best_b=1
    for i in tqdm(range(1,100), desc="Searching"):
        for j in range(-1000000,0,10000):
            model.set_param(i,j)
            if model.score(data,target)<min_rmse:
                min_rmse = model.score(data,target)
                best_a = i
                best_b = j
    model.set_param(best_a, best_b)

    print(best_a,"  ",best_b)
    print(model.score(data,target))

    fig = plt.figure(figsize=(16,9))
    plt.scatter(data, target, alpha=0.5)

    x_line = np.linspace(data.min(), data.max(), 100)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, 'r-', linewidth=2, label='Fitted line')
    
    plt.legend()
    plt.show()

    print(f"RMSE:{model.score(data,target):.2f}")

if __name__ == "__main__":
    main()