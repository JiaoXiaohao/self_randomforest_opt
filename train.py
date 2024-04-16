# -*- coding: utf-8 -*-
# File Name：train.py
# description: 训练模型函数
from utils import *
import numpy as np
def RF_train(sample_path, SavePath, space, msg, test_size=0.5):
    msg.emit("*" * 30 + " 开始训练 " + "*" * 30)
    msg.emit("模型：RandomForest")
    msg.emit("样本路径：" + sample_path)
    msg.emit("保存路径：" + SavePath)
    msg.emit("验证集比例：" + str(test_size))
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
    # msg.emit(x.shape, y.shape)
    # ------------------------------- change -------------------------------
    # 计算16个波段的相关系数矩阵，上三角矩阵
    # PlotHeatmap(np.corrcoef(x.T))
    msg.emit("*" * 30 + " 获取样本完成 " + "*" * 30)
    msg.emit("*" * 30 + " 分割训练集和验证集 " + "*" * 30)
    # 分割训练集和验证集
    train_data, test_data, train_label, test_label = Split_train_test_dataset(
        x, y, val_size=test_size, random_state=42
    )
    msg.emit("*" * 30 + " 分割完成 " + "*" * 30)
    msg.emit(f"训练集样本数：{train_data.shape[0]}")
    msg.emit(f"验证集样本数：{test_data.shape[0]}")
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
                clf = RandomForestClassifier(**params, random_state=42)
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
        trials = Trials()
        fmin(f, space, algo=tpe.suggest, max_evals=max_evals, trials=trials, show_progressbar=False)
        max_f1_dict = b[a.index(max(a))]
        msg.emit("最佳超参数：" + str(max_f1_dict))
        return max_f1_dict
    # 超参数优化
    max_f1_dict = HyperOptimize(train_data, train_label, test_data, test_label, max_evals=20)
    msg.emit("*" * 30 + " 超参数优化完成 " + "*" * 30)
    msg.emit("*" * 30 + " 训练中... " + "*" * 30)
    # 训练模型
    classifier = RandomForestClassifier(**max_f1_dict, random_state=42)
    classifier.fit(train_data, train_label.ravel())
    msg.emit("*" * 30 + " 超参数优化完成 " + "*" * 30)
    msg.emit("*" * 30 + " 训练结果 " + "*" * 30)
    msg.emit("训练集精度：" + str(classifier.score(train_data, train_label)))
    msg.emit("验证集精度：" + str(classifier.score(test_data, test_label)))
    # 预测输出
    predict = classifier.predict(test_data)
    msg.emit("平均绝对误差：" + str(mean_absolute_error(test_label, predict)))
    msg.emit("召回率：" + str(recall_score(test_label, predict, average="macro")))
    msg.emit("准确率：" + str(accuracy_score(test_label, predict)))
    msg.emit("F1值：" + str(f1_score(test_label, predict, average="macro")))
    # ------------------------------- change -------------------------------
    # 绘制学习曲线
    # 交叉验证参数
    shuffle = model_selection.ShuffleSplit(n_splits=10, test_size=0.5, random_state=42)
    plot_learning_curve(
        classifier,
        "RandomForest Learning Curve",
        x,
        y.ravel(),
        model_name="RandomForest",
        # ylim=(0.7, 1.01),
        cv=shuffle,
    )
    # ------------------------------- change -------------------------------
    # 绘制特征重要性
    PlotImportance_RF(classifier.feature_importances_)
    msg.emit("*" * 5 + " 学习曲线已生成 " + "*" * 5)
    # 保存模型
    SavePickle(classifier, SavePath)
    msg.emit("*" * 5 + f" 训练完成,模型已保存至 {SavePath} " + "*" * 5)
def XGB_train(sample_path, SavePath, space, msg, test_size=0.5):
    msg.emit("*" * 30 + " 开始训练 " + "*" * 30)
    msg.emit("模型：XGB_train")
    msg.emit("样本路径：" + sample_path)
    msg.emit("保存路径：" + SavePath)
    msg.emit("验证集比例：" + str(test_size))
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
    msg.emit("*" * 30 + " 分割训练集和验证集 " + "*" * 30)
    # 分割训练集和验证集
    train_data, test_data, train_label, test_label = Split_train_test_dataset(x, y, val_size=test_size)
    msg.emit("*" * 30 + " 分割完成 " + "*" * 30)
    msg.emit(f"训练集样本数：{train_data.shape[0]}")
    msg.emit(f"验证集样本数：{test_data.shape[0]}")
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
    msg.emit("验证集精度：" + str(classifier.score(test_data, test_label)))
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
def LightGBM_train(
    sample_path,
    SavePath,
    space,
    msg,
    test_size=0.5,
):
    msg.emit("*" * 30 + " 开始训练 " + "*" * 30)
    msg.emit("模型：LightGBM")
    msg.emit("样本路径：" + sample_path)
    msg.emit("保存路径：" + SavePath)
    msg.emit("验证集比例：" + str(test_size))
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
    y = CheckY(y.ravel())
    # print(x.shape, y.shape)
    msg.emit("*" * 30 + " 获取样本完成 " + "*" * 30)
    msg.emit("*" * 30 + " 分割训练集和验证集 " + "*" * 30)
    # 分割训练集和验证集
    train_data, test_data, train_label, test_label = Split_train_test_dataset(x, y, val_size=0.5)
    msg.emit("*" * 30 + " 分割完成 " + "*" * 30)
    msg.emit(f"训练集样本数：{train_data.shape[0]}")
    msg.emit(f"验证集样本数：{test_data.shape[0]}")
    msg.emit(f"样本特征数：{train_data.shape[1]}")
    msg.emit(f"类别数：{len(np.unique(train_label))}")
    msg.emit("*" * 30 + " 正在进行超参数优化... " + "*" * 30)
    # 定义超参数优化函数
    def HyperOptimize(train_data, train_label, test_data, test_label, max_evals):
        a = []
        b = []
        def hyperopt_train_test(params):
            skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            f1_scores = []
            callbacks = [
                lgb.early_stopping(stopping_rounds=10, verbose=False),
            ]
            for train_index, val_index in skf.split(train_data, train_label):
                X_train, X_val = train_data[train_index], train_data[val_index]
                y_train, y_val = train_label[train_index], train_label[val_index]
                clf = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
                clf.fit(
                    X_train,
                    y_train.ravel(),
                    callbacks=callbacks,
                    eval_set=[(X_val, y_val.ravel())],
                    eval_metric="multi_logloss",
                )
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
        # 判断任务是否为二分类
        if len(np.unique(train_label)) == 2:
            space["objective"] = "binary"
            space["metric"] = "binary_logloss"
            space["num_class"] = 1
        else:
            space["objective"] = "multiclass"
            space["metric"] = "multi_logloss"
            space["num_class"] = len(np.unique(train_label))
        trials = Trials()
        fmin(f, space, algo=tpe.suggest, max_evals=max_evals, trials=trials, show_progressbar=False)
        max_f1_dict = b[a.index(max(a))]
        msg.emit("最佳参数：" + str(max_f1_dict))
        return max_f1_dict
    # 超参数优化
    max_f1_dict = HyperOptimize(train_data, train_label, test_data, test_label, max_evals=100)
    msg.emit("*" * 30 + " 超参数优化完成 " + "*" * 30)
    msg.emit("*" * 30 + " 训练中... " + "*" * 30)
    # 训练模型
    clf = lgb.LGBMClassifier(**max_f1_dict, random_state=42, verbose=-1)
    callbacks = [
        lgb.early_stopping(stopping_rounds=10, verbose=False),
    ]
    clf.fit(
        train_data,
        train_label.ravel(),
        callbacks=callbacks,
        eval_set=[(test_data, test_label.ravel())],
    )
    msg.emit("*" * 30 + " 超参数优化完成 " + "*" * 30)
    msg.emit("*" * 30 + " 训练结果 " + "*" * 30)
    msg.emit("训练集精度：" + str(clf.score(train_data, train_label)))
    msg.emit("验证集精度：" + str(clf.score(test_data, test_label)))
    # 预测输出
    predict = clf.predict(test_data)
    msg.emit("平均绝对误差：" + str(mean_absolute_error(test_label, predict)))
    msg.emit("召回率：" + str(recall_score(test_label, predict, average="macro")))
    msg.emit("准确率：" + str(accuracy_score(test_label, predict)))
    msg.emit("F1值：" + str(f1_score(test_label, predict, average="macro")))
    # -------------------------------------------change------------------------------------------
    # # 交叉验证参数
    shuffle = model_selection.ShuffleSplit(n_splits=10, test_size=0.5, random_state=42)
    # 绘制学习曲线
    plot_learning_curve(
        clf,
        "LightGBM Learning Curve",
        train_data,
        train_label.ravel(),
        model_name="LightGBM",
        cv=shuffle,
        # ylim=(0.7,1.01),
    )
    # 绘制特征重要性
    PlotImportance_LGBM(clf.feature_importances_, column_names[:-1])
    msg.emit("*" * 5 + " 学习曲线已生成 " + "*" * 5)
    # msg.emit("特征重要性：", clf.feature_importances_)
    # --------------------------------------------------------------------------------------
    # msg.emit("*" * 30 + " 训练完成 " + "*" * 30)
    # 保存模型
    SavePickle(clf, SavePath)
    msg.emit("*" * 5 + f" 训练完成,模型已保存至 {SavePath} " + "*" * 5)
