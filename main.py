import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import os

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("results", exist_ok=True)

def load_and_clean_data():
    # 读取教师预划分的训练/测试集
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    print("===== 原始数据探查 =====")
    print(f"训练集原始样本量：{len(train_df)}")
    print(f"测试集原始样本量：{len(test_df)}")
    print(f"训练集缺失值数量：{train_df.isna().sum().sum()}")
    print(f"测试集缺失值数量：{test_df.isna().sum().sum()}")

    # 修正数据类型：把Solidity、Compactness转为数值，无法转换的设为NaN
    train_df["Solidity"] = pd.to_numeric(train_df["Solidity"], errors="coerce")
    train_df["Compactness"] = pd.to_numeric(train_df["Compactness"], errors="coerce")
    test_df["Solidity"] = pd.to_numeric(test_df["Solidity"], errors="coerce")
    test_df["Compactness"] = pd.to_numeric(test_df["Compactness"], errors="coerce")

    # 标签映射，统一为7个标准类别
    label_mapping = {
        "HOROZ": "HOROZ", "H0R0Z": "HOROZ", "horoz": "HOROZ", "Horoz": "HOROZ",
        "DERMASON": "DERMASON", "D3RMAS0N": "DERMASON", "dermason": "DERMASON", "Dermason": "DERMASON",
        "SEKER": "SEKER", "S3K3R": "SEKER", "seker": "SEKER", "Seker": "SEKER",
        "SIRA": "SIRA", "sira": "SIRA", "Sira": "SIRA",
        "BARBUNYA": "BARBUNYA", "barbunya": "BARBUNYA", "Barbunya": "BARBUNYA",
        "CALI": "CALI", "cali": "CALI", "Cali": "CALI",
        "BOMBAY": "BOMBAY", "B0MBAY": "BOMBAY", "bombay": "BOMBAY", "Bombay": "BOMBAY"
    }
    train_df["Class"] = train_df["Class"].map(label_mapping)
    test_df["Class"] = test_df["Class"].map(label_mapping)

    # 分别剔除缺失值，不合并后再切分，避免打乱原有划分
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    print(f"\n剔除缺失值后：训练集 {len(train_df)}，测试集 {len(test_df)}")

    # 离群值过滤：仅在训练集上统计IQR，分别过滤，避免测试集被清空
    if len(train_df) > 0:
        train_feat = train_df.drop("Class", axis=1)
        Q1 = train_feat.quantile(0.25)
        Q3 = train_feat.quantile(0.75)
        IQR = Q3 - Q1
        # 训练集过滤
        train_mask = ~((train_feat < Q1 - 1.5*IQR) | (train_feat > Q3 + 1.5*IQR)).any(axis=1)
        train_df = train_df[train_mask]
        # 测试集用同一阈值过滤
        test_feat = test_df.drop("Class", axis=1)
        test_mask = ~((test_feat < Q1 - 1.5*IQR) | (test_feat > Q3 + 1.5*IQR)).any(axis=1)
        test_df = test_df[test_mask]
        print(f"离群值过滤后：训练集 {len(train_df)}，测试集 {len(test_df)}")

    # 兜底：如果过滤后仍为空，保留原始清洗后数据
    if len(train_df) == 0 or len(test_df) == 0:
        print("警告：过滤后样本为空，跳过离群值过滤")
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
        train_df["Solidity"] = pd.to_numeric(train_df["Solidity"], errors="coerce")
        train_df["Compactness"] = pd.to_numeric(train_df["Compactness"], errors="coerce")
        test_df["Solidity"] = pd.to_numeric(test_df["Solidity"], errors="coerce")
        test_df["Compactness"] = pd.to_numeric(test_df["Compactness"], errors="coerce")
        train_df["Class"] = train_df["Class"].map(label_mapping)
        test_df["Class"] = test_df["Class"].map(label_mapping)
        train_df = train_df.dropna()
        test_df = test_df.dropna()

    # 分离特征与标签
    X_train = train_df.drop("Class", axis=1)
    y_train = train_df["Class"]
    X_test = test_df.drop("Class", axis=1)
    y_test = test_df["Class"]

    # 标签编码
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n===== 最终清洗后数据 =====")
    print(f"训练集样本：{len(X_train)}，测试集样本：{len(X_test)}")
    print(f"类别分布：\n{pd.Series(y_train).value_counts()}")
    return X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, le, X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test, le):
    models = {
        "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
        "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost(课外拓展)": xgb.XGBClassifier(n_estimators=100, random_state=42, objective="multi:softmax", num_class=7)
    }
    results = []
    trained_models = {}
    for model_name, model in models.items():
        print(f"\n===== 正在训练：{model_name} =====")
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        pred_start = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - pred_start
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        results.append({
            "模型名称": model_name,
            "训练集准确率": round(train_acc, 4),
            "测试集准确率": round(test_acc, 4),
            "训练耗时(s)": round(train_time, 4),
            "推理耗时(s)": round(pred_time, 4),
            "宏平均F1": round(report["macro avg"]["f1-score"], 4),
            "加权平均F1": round(report["weighted avg"]["f1-score"], 4)
        })
        trained_models[model_name] = {"model": model, "y_pred": y_pred, "y_test": y_test}
        print(f"{model_name} 训练完成，测试集准确率：{round(test_acc, 4)}")
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/模型精度汇总.csv", index=False, encoding="utf-8-sig")
    print("\n===== 模型精度汇总 =====")
    print(results_df)
    return results_df, trained_models

def robustness_test(X_train, X_test, y_train, y_test, le):
    noise_levels = [0.1, 0.3, 0.5]
    robustness_results = []
    for model_name, model in {
        "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
        "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost(课外拓展)": xgb.XGBClassifier(n_estimators=100, random_state=42, objective="multi:softmax", num_class=7)
    }.items():
        model.fit(X_train, y_train)
        base_acc = accuracy_score(y_test, model.predict(X_test))
        for noise_std in noise_levels:
            rng = np.random.default_rng(42)
            X_train_noisy = X_train + rng.normal(0, noise_std, X_train.shape)
            model.fit(X_train_noisy, y_train)
            noisy_acc = accuracy_score(y_test, model.predict(X_test))
            decay_rate = round((base_acc - noisy_acc) / base_acc * 100, 2)
            robustness_results.append({
                "模型名称": model_name, "噪声强度": noise_std,
                "基准准确率": round(base_acc, 4), "噪声后准确率": round(noisy_acc, 4),
                "精度衰减率(%)": decay_rate
            })
    robustness_df = pd.DataFrame(robustness_results)
    robustness_df.to_csv("results/模型鲁棒性测试结果.csv", index=False, encoding="utf-8-sig")
    print("\n===== 鲁棒性测试结果 =====")
    print(robustness_df)
    return robustness_df

def plot_all(results_df, trained_models, robustness_df, le, X_train, y_train):
    plt.figure(figsize=(10,6))
    sns.barplot(x="模型名称", y="测试集准确率", data=results_df, palette="Blues_d")
    plt.title("不同模型测试集准确率对比", fontsize=14)
    plt.ylim(0.8,1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/模型精度对比.png", dpi=300, bbox_inches="tight")
    plt.close()

    xgb_pred = trained_models["XGBoost(课外拓展)"]["y_pred"]
    y_test = trained_models["XGBoost(课外拓展)"]["y_test"]
    cm = confusion_matrix(y_test, xgb_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("XGBoost混淆矩阵", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/XGBoost混淆矩阵.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10,6))
    for m in robustness_df["模型名称"].unique():
        d = robustness_df[robustness_df["模型名称"]==m]
        plt.plot(d["噪声强度"], d["噪声后准确率"], marker="o", label=m, linewidth=2)
    plt.title("不同噪声强度下模型准确率变化", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/模型鲁棒性对比.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12,10))
    corr = X_train.corr()
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("特征相关性热力图", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/特征相关性热力图.png", dpi=300, bbox_inches="tight")
    plt.close()

    class_counts = le.inverse_transform(y_train).value_counts()
    plt.figure(figsize=(10,6))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="Blues_d")
    plt.title("训练集类别分布", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/训练集类别分布.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("\n===== 所有图表已生成至results文件夹 =====")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le, X_train_org, X_test_org, y_train_org, y_test_org = load_and_clean_data()
    results_df, trained_models = train_and_evaluate(X_train, X_test, y_train, y_test, le)
    robustness_df = robustness_test(X_train, X_test, y_train, y_test, le)
    plot_all(results_df, trained_models, robustness_df, le, X_train_org, y_train_org)
    print("\n===== 全流程执行完成！所有结果已保存到results文件夹 =====")