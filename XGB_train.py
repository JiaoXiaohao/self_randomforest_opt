# -*- coding:utf-8 -*-
# Author: jiaoxiaohao
# E-mail: jiaoxiaohao876@gmail.com
# Time: 2024-02-16 23:53:46
# File name: xgboost_train.py
# Nothing is true, everything is permitted.

from utils import *

import numpy as np

def XGB_train(sample_path, SavePath, space, msg, test_size=0.5):
    msg.emit("*" * 30 + " 开始训练 " + "*" * 30)
    msg.emit("模型：XGB_train")
    msg.emit("样本路径：" + sample_path)
    msg.emit("保存路径：" + SavePath)
    msg.emit("测试集比例：" + str(test_size))
    msg.emit("*" * 30 + " 获取样本中... " + "*" * 30)
    # 读取第一行以获取列名
    if sample_path.endswith(".txt"):
        with open(sample_path, "r") as file:
            header_line = file.readline().strip()
    elif sample_path.endswith(".csv"):
        with open(sample_path, "r") as file:
            header_line = file.readline().strip()

    # 获取列名
    column_names = header_line.split(",")
    # 重新定义converters字典，跳过第一行（表头）
    converters = {i: label_dict if column_names[i] == "class" else int for i in range(len(column_names))}
    # 从第二行开始读取数据
    data = np.loadtxt(sample_path, dtype=int, delimiter=",", skiprows=1, converters=converters)

    x, y = np.split(data, indices_or_sections=(len(column_names) - 1,), axis=1)  # x为数据，y为标签
    # 判断y数据是否符合xgboost的输入要求
    y = CheckY(y)
    # msg.emit(x.shape, y.shape)

    # 计算16个波段的相关系数矩阵，上三角矩阵
    # PlotHeatmap(np.corrcoef(x.T))

    msg.emit("*" * 30 + " 获取样本完成 " + "*" * 30)
    msg.emit("*" * 30 + " 分割训练集和测试集 " + "*" * 30)
    # 分割训练集和测试集
    train_data, test_data, train_label, test_label = Split_train_test_dataset(x, y, test_size=test_size)
    msg.emit("*" * 30 + " 分割完成 " + "*" * 30)
    msg.emit(f"训练集样本数：{train_data.shape[0]}")
    msg.emit(f"测试集样本数：{test_data.shape[0]}")
    msg.emit(f"样本特征数：{train_data.shape[1]}")
    msg.emit(f"类别数：{len(np.unique(train_label))}")
    
    msg.emit("*" * 30 + " 正在进行超参数优化... " + "*" * 30)

    # 超参数优化
    def HyperOptimize(train_data, train_label, test_data, test_label, max_evals=100):
        a = []
        b = []

        def hyperopt_train_test(params):
            skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            f1_scores = []

            for train_index, val_index in skf.split(train_data, train_label):
                X_train, X_val = train_data[train_index], train_data[val_index]
                y_train, y_val = train_label[train_index], train_label[val_index]

                clf = xgb.XGBClassifier(**params, random_state=42)
                clf.fit(X_train, y_train.ravel())
                pred_label = clf.predict(X_val)
                f1 = f1_score(y_val, pred_label, average="macro")
                f1_scores.append(f1)

            avg_f1 = sum(f1_scores) / len(f1_scores)
            a.append(avg_f1)
            b.append(params)
            return -avg_f1  # 注意返回负的 F1 score，因为hyperopt是最小化目标

        def f(params):
            acc = hyperopt_train_test(params)
            return {"loss": acc, "status": STATUS_OK}

        if len(np.unique(train_label)) == 2:
            space["objective"] = "binary:logistic"
            space["eval_metric"] = "logloss"
        else:
            space["objective"] = "multi:softmax"
            space["eval_metric"] = "mlogloss"
            space["num_class"] = len(np.unique(train_label))
        trials = Trials()
        fmin(f, space, algo=tpe.suggest, max_evals=max_evals, trials=trials, show_progressbar=False)
        max_f1_dict = b[a.index(max(a))]
        msg.emit(f"最佳超参数：{max_f1_dict}")
        return max_f1_dict

    max_f1_dict = HyperOptimize(train_data, train_label, test_data, test_label)
    # 训练模型
    classifier = xgb.XGBClassifier(**max_f1_dict, random_state=42)
    classifier.fit(train_data, train_label.ravel())
    msg.emit("*" * 30 + " 超参数优化完成 " + "*" * 30)

    msg.emit("*" * 30 + " 训练结果 " + "*" * 30)
    msg.emit("训练集精度：" + str(classifier.score(train_data, train_label)))
    msg.emit("测试集精度：" + str(classifier.score(test_data, test_label)))
    # 预测输出
    predict = classifier.predict(test_data)
    msg.emit("平均绝对误差：" + str(mean_absolute_error(test_label, predict)))
    msg.emit("召回率：" + str(recall_score(test_label, predict, average="macro")))
    msg.emit("准确率：" + str(accuracy_score(test_label, predict)))
    msg.emit("F1值：" + str(f1_score(test_label, predict, average="macro")))
# -------------------------------------------------change-------------------------------------
    # 交叉验证参数
    shuffle = model_selection.ShuffleSplit(n_splits=10, test_size=0.5, random_state=42)
    # 绘制学习曲线
    plot_learning_curve(
        classifier,
        "XGB Learning Curve",
        x,
        y.ravel(),
        model_name="XGB",
        # ylim=(0.7, 1.01),
        cv=shuffle,
    )

    # 绘制特征重要性
    PlotImportance_XGB(classifier.feature_importances_)
    msg.emit("*" * 5 + " 学习曲线已生成 " + "*" * 5)
# -------------------------------------------------------------------------------------------
    # 保存模型
    SavePickle(classifier, SavePath)
    msg.emit("*" * 5 + f" 训练完成,模型已保存至 {SavePath} " + "*" * 5)


if __name__ == "__main__":
    sample_path = r"data\\sample_demo.txt"
    SavePath = r"data\\model_xgb.pickle"
    test_size = 0.2
    space = {
        "n_estimators": hp.choice("n_estimators", range(50, 150, 5)),
        "max_depth": hp.choice("max_depth", range(100, 500)),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        "gamma": hp.uniform("gamma", 0, 10),
        "reg_alpha": hp.uniform("reg_alpha", 0, 10),
        "reg_lambda": hp.uniform("reg_lambda", 0, 10),
        "min_child_weight": hp.uniform("min_child_weight", 0, 10),
    }
    msg = ""
    XGB_train(sample_path, SavePath, space, msg, test_size=test_size)
