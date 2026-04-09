from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer,make_column_selector
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

def print_cv(cv_result:dict):
    print(f"{"fit time":<12}{cv_result["fit_time"].mean():.3f}±{cv_result["fit_time"].std():.3f}")
    print(f"{"score time":<12}{cv_result["score_time"].mean():.3f}±{cv_result["score_time"].std():.3f}")
    print(f"{"score":<12}{cv_result["test_score"].mean():.3f}±{cv_result["test_score"].std():.3f}")

def plot_coefs(cv_result:dict, data:pd.DataFrame, color:str="darkorange", process_name:str="classifier"):
    estimators = cv_result['estimator']
    
    # 1. 获取特征名称
    first_estimator = estimators[0]
    preprocessor = first_estimator.named_steps['preprocessor']
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = preprocessor.get_feature_names()

    coefs_list = []
    for est in estimators:
        lr_model = est.named_steps[process_name]
        coefs_list.append(lr_model.coef_[0])
        
    coefs_df = pd.DataFrame(coefs_list, columns=feature_names)
    
    # 2. 简化特征名称：移除常见的前缀，使标签更简洁
    def simplify_name(name):
        # 如果名称中包含下划线，通常格式为 "featureName_category"
        if '_' in name:
            parts = name.split('_', 1)
            # 如果前缀很长，我们只保留后半部分，或者可以根据需要保留缩写
            # 这里简单起见，如果第二部分存在则返回第二部分，否则返回原名
            # 也可以改为返回 f"{parts[0][:3]}_{parts[1]}" 来保留一点前缀信息
            return parts[1] if len(parts) > 1 else name
        return name

    coefs_df.columns = [simplify_name(col) for col in coefs_df.columns]
    
    coefs_melted = coefs_df.melt(var_name='Feature', value_name='Coefficient')
    
    # 3. 按系数绝对值的均值排序，让重要的特征排在一起
    feature_order = coefs_melted.groupby('Feature')['Coefficient'].mean().abs().sort_values(ascending=False).index
    
    plt.figure(figsize=(12, 10)) # 调整尺寸以适应排序后的布局
    sns.set_theme(style="whitegrid") # 设置美观的主题
    
    sns.boxplot(
        x='Coefficient', 
        y='Feature', 
        data=coefs_melted, 
        order=feature_order, # 应用排序
        color=color,
        width=0.6,
        showfliers=True,
        linewidth=0.8,       # 箱体边框线宽
        flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5} # 美化异常值点
    )
    
    plt.title('Distribution of Logistic Regression Coefficients (Sorted by Importance)', fontsize=14, pad=15)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.axvline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.7) # 虚线零参考线
    plt.tick_params(axis='y', labelsize=10) # 调整 Y 轴标签字体大小
    plt.tight_layout()
def main():
    script_dir = Path(__file__).parent.resolve()
    data_path = script_dir / "data" / "adult-census.csv"
    data_raw = pd.read_csv(data_path)

    target_name = "class"
    target = data_raw[target_name]
    data = data_raw.drop(columns=[target_name, "education-num"])

    num_processor = Pipeline(
        steps=[
            ("poly",PolynomialFeatures(degree=2, interaction_only=True)),
            ("scaler",StandardScaler())
        ]
    )

    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore",min_frequency=0.01), make_column_selector(dtype_include="object")),
        (num_processor, make_column_selector(dtype_exclude="object")),
        remainder="passthrough"
    )

    model = Pipeline(
        steps=[
            ("preprocessor",preprocessor),
            ("classifier",LogisticRegression(C=0.01, max_iter=1000))
        ]
    )

    cv_results = cross_validate(model, data, target, cv=10, return_estimator=True)
    print_cv(cv_results)
    plot_coefs(cv_results, data, process_name="classifier")
    plt.show()

if __name__ == "__main__":
    main()