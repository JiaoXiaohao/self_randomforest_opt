# -*- coding:utf-8 -*-
# File name: utils.py
# 设置全局变量
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
# GUI所需库
from PyQt5 import QtCore, QtGui, QtWidgets
import time
# 运行所需库
from osgeo import gdal
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import model_selection
import csv
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import learning_curve
from sklearn.cluster import KMeans
from hyperopt import hp, fmin, tpe, Trials, space_eval, STATUS_OK
# 模型保存所需要库
import pickle
# 绘图所需库
import scienceplots
import matplotlib
import matplotlib.pyplot as plt
from seaborn import heatmap
from warnings import simplefilter
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
simplefilter(action="ignore", category=FutureWarning)
plt.style.use(["science", "grid", "no-latex"])
# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 设置负号正常显示
plt.rcParams["axes.unicode_minus"] = False
plt.figure()
# ===================== get_sample.py =====================
def printLog(msg, blog_textBrowser):
    try:
        blog_textBrowser.append(msg)
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Error", str(e), QtWidgets.QMessageBox.Ok)
def get_info(file_name, msg):
    def read_tif(file_name):
        dataset = gdal.Open(file_name)
        if dataset is None:
            msg.emit(f"{file_name}文件无法打开")
        return dataset
    dataset = read_tif(file_name)
    msg.emit("Dataset Name:" + file_name)
    msg.emit("Driver: " + dataset.GetDriver().ShortName + "/" + dataset.GetDriver().LongName)
    msg.emit(
        "Size is："
        + "W"
        + str(dataset.RasterXSize)
        + "*"
        + "H"
        + str(dataset.RasterYSize)
        + "*"
        + "B"
        + str(dataset.RasterCount)
    )
    msg.emit("Projection is：" + dataset.GetProjection())
    width, height, bands = dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount
    geotrans = dataset.GetGeoTransform()
    if geotrans is not None:
        msg.emit(f"Origin = ({geotrans[0]}, {geotrans[3]})")
        msg.emit(f"Pixel Size = ({geotrans[1]}, {-geotrans[5]})")
    _data_ = dataset.ReadAsArray(0, 0, width, height)
    return width, height, bands, geotrans, _data_, dataset.GetProjection()
def ReadNumClass(numclass_path):
    # 规定一个字典
    num_class = {}
    if numclass_path.endswith(".txt"):
        # 文件第一行为表头，{类别 数量}
        with open(numclass_path, "r") as file_read_obj:
            # 跳过第一行
            file_read_obj.readline()
            # 读取其余行
            for line in file_read_obj.readlines():
                # 去除空格
                line = line.strip()
                # 以空格分割
                line = line.split(",")
                # 以字典形式存储
                num_class[int(line[0])] = int(line[1])
        return num_class
    elif numclass_path.endswith(".csv"):
        # 文件第一行为表头，{类别 数量}
        with open(numclass_path, "r") as file_read_obj:
            csv_read_obj = csv.reader(file_read_obj)
            # 跳过第一行
            next(csv_read_obj)
            # 读取其余行
            for line in csv_read_obj:
                # 以字典形式存储
                num_class[int(line[0])] = int(line[1])
        return num_class
def adjustLocationByGeoTransform(img, label, img_geotrans, label_geotrans, msg):
    # 判断两个影像的坐标是否一致
    if img_geotrans == label_geotrans:
        return img
    msg.emit("Warning: _img_ and _label_ are not in the same location")
    try:
        # 获取_label_图像的地理坐标范围
        label_min_x, label_max_y = gdal.ApplyGeoTransform(label_geotrans, 0, 0)
        label_max_x, label_min_y = gdal.ApplyGeoTransform(label_geotrans, label.shape[1], label.shape[0])
        # 获取_img_图像的地理坐标范围
        img_min_x, img_max_y = gdal.ApplyGeoTransform(img_geotrans, 0, 0)
        img_max_x, img_min_y = gdal.ApplyGeoTransform(img_geotrans, img.shape[2], img.shape[1])
    except Exception as e:
        msg.emit(f"Error calculating geo transforms: {e}")
        return img
    # 确保_label_包含在_img_内
    if (
        label_min_x >= img_min_x
        and label_max_x <= img_max_x
        and label_min_y >= img_min_y
        and label_max_y <= img_max_y
    ):
        # 计算在_img_中对应_label_的区域
        label_start_col = int((label_min_x - img_geotrans[0]) / img_geotrans[1])
        label_start_row = int((img_max_y - label_max_y) / abs(img_geotrans[5]))
        label_end_col = int((label_max_x - img_geotrans[0]) / img_geotrans[1])
        label_end_row = int((img_max_y - label_min_y) / abs(img_geotrans[5]))
        # 提取对应区域的_img_
        adjusted_img = img[
            :,
            np.s_[label_start_row:label_end_row],
            np.s_[label_start_col:label_end_col],
        ]
        return adjusted_img
    else:
        msg.emit("Error: _label_ is not in _img_")
        return img
def WriteSample(_img_, _label_, sample_path, class_num, msg):
    # 读取_label_数据中的唯一值确定类别
    classes = np.unique(_label_)
    msg.emit("当前标签数组所有值：")
    for i, cls in enumerate(classes):
        if cls != 0:
            msg.emit("Value:" + str(cls) + " Class:" + str(i))
        else:
            msg.emit("Value:" + str(cls) + " Class:" + "Nan")
    msg.emit("目标类别及对应数量：")
    for key, value in class_num.items():
        msg.emit("Class:" + str(key) + " Num:" + str(value))
    # 提前检查文件存在性，存在就删除
    if os.path.exists(sample_path):
        os.remove(sample_path)
    # 统计每个类别的样本数量
    class_counts = {cls: 0 for cls in classes}
    # 读取_img_的波段数
    bands = _img_.shape[0]
    # 打开文件，如果文件不存在，就会自动创建,
    # 如果sample_path是csv文件，就用csv模块打开，如果是txt文件，就用open打开
    if sample_path.endswith(".csv"):
        file_write_obj = open(sample_path, "a+", newline="", encoding="utf-8")
        csv_write_obj = csv.writer(file_write_obj)
        # 将波段数以及类别写入文件
        csv_write_obj.writerow(list(str("Band") + str(i) for i in range(bands)) + ["class"])
        for i, row in enumerate(_label_):
            for j, _class in enumerate(row):
                if _class != 0:
                    # 判断每个类别的样本数量是否达到要求
                    if class_counts[_class] >= class_num[_class]:
                        break
                    else:
                        # 获取该像元的所有波段像素值
                        pixel_values = [_img_[k, i, j] for k in range(_img_.shape[0])]
                        # 将像素值和类别写入文件
                        csv_write_obj.writerow(pixel_values + [_class])
                        # 统计每个类别的样本数量
                        class_counts[_class] += 1
    elif sample_path.endswith(".txt"):
        with open(sample_path, "a+", encoding="utf-8") as file_write_obj:
            # 将波段数以及类别写入文件
            file_write_obj.writelines(
                ",".join(
                    map(
                        str,
                        list(str("Band") + str(i) for i in range(bands)) + ["class"],
                    )
                )
                + "\n"
            )
            for i, row in enumerate(_label_):
                for j, _class in enumerate(row):
                    if _class != 0:
                        # 判断每个类别的样本数量是否达到要求
                        if class_counts[_class] >= class_num[_class]:
                            break
                        else:
                            # 获取该像元的所有波段像素值
                            pixel_values = [_img_[k, i, j] for k in range(_img_.shape[0])]
                            # 将像素值和类别写入文件
                            line = ",".join(map(str, pixel_values + [_class]))
                            file_write_obj.writelines(line + "\n")
                            # 统计每个类别的样本数量
                            class_counts[_class] += 1
def get_samples(img_path, label_path, sample_path, numclass_path, msg):
    try:
        start_time = time.time()
        # 判断img_path和label_path是否为栅格数据
        if not img_path.endswith(".tif") and not img_path.endswith(".tiff") and not img_path.endswith(".pix"):
            msg.emit("Error: img_path is not a raster data")
            return
        if (
            not label_path.endswith(".tif")
            and not label_path.endswith(".tiff")
            and not label_path.endswith(".pix")
        ):
            msg.emit("Error: label_path is not a raster data")
            return
        # 读取影像数据
        try:
            msg.emit("*" * 10 + " 读取影像数据 " + "*" * 10)
            (
                img_width,
                img_height,
                img_bands,
                img_geotrans,
                _img_,
                img_projection,
            ) = get_info(img_path, msg)
            msg.emit("原始影像数据形状：" + str(_img_.shape))
            msg.emit("*" * 10 + " 读取完毕 " + "*" * 10)
        except Exception as e:
            msg.emit("Error: " + str(e) + " 读取影像数据失败，检查数据!!!")
            return False
        # 读取标签数据
        try:
            msg.emit("*" * 10 + " 读取标签数据 " + "*" * 10)
            # 读取标签数据
            (
                label_width,
                label_height,
                label_bands,
                label_geotrans,
                _label_,
                label_projection,
            ) = get_info(label_path, msg)
            msg.emit("原始标签数据形状：" + str(_label_.shape))
            msg.emit("*" * 10 + " 读取完毕 " + "*" * 10)
        except Exception as e:
            msg.emit("Error: " + str(e) + " 读取标签数据失败，检查数据!!!")
            return False
        # 调整label和img的大小一致
        msg.emit("*" * 10 + " 调整数据 " + "*" * 10)
        _img_ = adjustLocationByGeoTransform(_img_, _label_, img_geotrans, label_geotrans, msg)
        msg.emit("调整后影像数据形状：" + str(_img_.shape))
        msg.emit("*" * 10 + " 调整完毕 " + "*" * 10)
        # 写入数据
        try:
            msg.emit("*" * 10 + " 写入数据 " + "*" * 10)
            class_num = ReadNumClass(numclass_path)
            WriteSample(_img_, _label_, sample_path, class_num, msg)
            msg.emit("*" * 10 + " 写入完毕 " + "*" * 10)
        except Exception as e:
            msg.emit("Error: " + str(e) + " 写入数据失败，检查数据!!!")
            return False
        end_time = time.time()
        msg.emit("*" * 10 + " 样本获取运行时间: " + str(end_time - start_time) + " 秒 " + "*" * 10)
        return True
    except Exception as e:
        msg.emit("Error: " + str(e))
        return False
# ===================== RF_train.py =====================
# 定义字典，便于解析样本数据集txt
def label_dict(s):
    # 如果是数字，直接返回
    if s.isdigit():
        return int(s)
    # 如果是字符串，返回对应的数字
    else:
        it = {"Vegetation": 1, "Non-Vegetation": 2}
        return it[s]
def PlotHeatmap(corr):
    # 保留下三角矩阵
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = False
    # 绘制热力图
    heatmap(corr, mask=mask, cmap="RdBu_r", annot=True, fmt=".2f")
    plt.savefig("corr.png", dpi=500)
    # plt.show()
def Split_train_test_dataset(x, y, val_size=0.2, random_state=1):
    # 分割训练集和验证集
    train_data, test_data, train_label, test_label = model_selection.train_test_split(
        x,
        y,
        random_state=random_state,
        test_size=val_size,
    )
    # 输出训练集和测试集的样本数量和特征数量
    # print("训练集样本数量：", train_data.shape[0])
    # print("测试集样本数量：", test_data.shape[0])
    return train_data, test_data, train_label, test_label
def plot_learning_curve(estimator, title, X, y, model_name, ylim=None, cv=None, n_jobs=1):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training sample size")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)  # 计算测试集上的平均得分
    test_scores_std = np.std(test_scores, axis=1)  # 计算测试集上的标准差
    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="b",
    )
    plt.plot(
        train_sizes,
        train_scores_mean,
        "o-",
        color="g",
        label="the score on the training set",
    )
    plt.plot(
        train_sizes,
        test_scores_mean,
        "o-",
        color="b",
        label="the score on the validation set",
    )
    plt.legend(loc="lower right")
    plt.savefig(f"learning curve of {model_name}.png", dpi=800)
    # 清除图像
    plt.clf()
    # plt.savefig("学习曲线_XBM.png", dpi=800)
def PlotImportance_RF(importances):
    indices = np.argsort(importances)[::-1]
    plt.title("Feature importances by RandomTreeClassifier")
    plt.bar(range(len(indices)), importances[indices], color="darkorange", align="center")
    plt.xticks(range(len(indices)), indices)
    plt.xlim([-1, len(indices)])
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.savefig("Feature importances by RandomTreeClassifier.png", dpi=500)
    # 清楚图像
    plt.clf()
    # plt.show()
def SavePickle(classifier, SavePath):
    # 以二进制的方式打开文件：
    file = open(SavePath, "wb")
    # 将模型写入文件：
    pickle.dump(classifier, file)
    # 最后关闭文件：
    file.close()
# ===================== RF_predict.py =====================
def Predict_RF_func(model_path, _img_):
    # 以读二进制的方式打开文件
    file = open(model_path, "rb")
    # 把模型从文件中读取出来
    rf_model = pickle.load(file)
    # 关闭文件
    file.close()
    # 用读入的模型进行预测
    # 在与测试前要调整一下数据的格式
    data = np.zeros((_img_.shape[0], _img_.shape[1] * _img_.shape[2]))
    # 使用tqdm显示进度
    for i in range(_img_.shape[0]):
        data[i] = _img_[i].flatten()
    data = data.swapaxes(0, 1)
    #  对调整好格式的数据进行预测
    pred = rf_model.predict(data)
    #  同样地，我们对预测好的数据调整为我们图像的格式
    pred = pred.reshape(_img_.shape[1], _img_.shape[2])
    pred = pred.astype(np.uint8)
    return pred
def PlotPredictResult(pred, model, classPath=r"data\\ClassDefine.txt"):
    labels = []
    colors = []
    # 设置颜色和标签
    with open(classPath, "r") as file_read_obj:
        # 跳过第一行
        file_read_obj.readline()
        # 读取其余行
        for line in file_read_obj.readlines():
            # 去除空格
            line = line.strip()
            # 以空格分割
            line = line.split(",")
            labels.append(line[2])
            colors.append(line[3])
    # 值为1的像素点为绿色，值为2的像素点为白色
    cmap = ListedColormap(colors)
    # 为图例添加标签和样式
    legend_elements = [Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
    # 添加图例
    plt.legend(handles=legend_elements, loc="best")
    plt.title("预测结果")
    # 保存
    plt.imshow(pred, cmap=cmap)
    plt.savefig(f"PredictResult_{model}.png", dpi=500)
    # plt.show()
# ===================== K_Means_Classify.py =====================
def KMeansExecute(data, K, img_width, img_height, n_init=10, random_state=0):
    # K-Means
    kmeans = KMeans(n_clusters=K, n_init=n_init, random_state=random_state)
    kmeans.fit(data)
    idx = kmeans.labels_
    labels = idx.reshape(img_width, img_height)
    return labels
def PlotKMeansResult(labels):
    # 显示聚类结果
    plt.imshow(labels, cmap="rainbow")
    # plt.show()
# ===================== SVM_Classify.py =====================
def Predict_SVM(model_path, _img_):
    # 以读二进制的方式打开文件
    file = open(model_path, "rb")
    # 把模型从文件中读取出来
    svm_model = pickle.load(file)
    # 关闭文件
    file.close()
    # 用读入的模型进行预测
    # 在与测试前要调整一下数据的格式
    data = np.zeros((_img_.shape[0], _img_.shape[1] * _img_.shape[2]))
    # 使用tqdm显示进度
    for i in range(_img_.shape[0]):
        data[i] = _img_[i].flatten()
    data = data.swapaxes(0, 1)
    # 对调整好格式的数据进行预测
    pred = svm_model.predict(data)
    # 同样地，我们对预测好的数据调整为我们图像的格式
    pred = pred.reshape(_img_.shape[1], _img_.shape[2])
    pred = pred.astype(np.uint8)
    return pred
# 保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if "int8" in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif "int16" in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if dataset != None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
# ===================== XGB_train.py =====================
def CheckY(y):
    # XGBoost要求标签从0开始
    if y.min() == 1:
        y = y - 1
    return y
def PlotImportance_XGB(importances):
    indices = np.argsort(importances)[::-1]
    # print("Feature ranking:")
    # for f in range(len(indices)):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    plt.title("Feature importances by XGBoost")
    plt.bar(range(len(indices)), importances[indices], color="darkorange", align="center")
    plt.xticks(range(len(indices)), indices)
    plt.xlim([-1, len(indices)])
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.legend().set_visible(False)
    plt.savefig("Feature importances by XGBoost.png", dpi=500)
    # plt.show()
def Predict_XGB_func(model_path, _img_):
    # 以读二进制的方式打开文件
    file = open(model_path, "rb")
    # 把模型从文件中读取出来
    xgb_model = pickle.load(file)
    # 关闭文件
    file.close()
    # 用读入的模型进行预测
    # 在与测试前要调整一下数据的格式
    data = np.zeros((_img_.shape[0], _img_.shape[1] * _img_.shape[2]))
    # 使用tqdm显示进度
    for i in range(_img_.shape[0]):
        data[i] = _img_[i].flatten()
    data = data.swapaxes(0, 1)
    # 对调整好格式的数据进行预测
    pred = xgb_model.predict(data)
    # 同样地，我们对预测好的数据调整为我们图像的格式
    pred = pred.reshape(_img_.shape[1], _img_.shape[2])
    pred = pred.astype(np.uint8)
    return pred
# ===================== lightGBM.py =====================
def PlotImportance_LGBM(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    # print("Feature ranking:")
    # for f in range(len(indices)):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    plt.title("Feature importances by LightGBM")
    plt.bar(range(len(indices)), importances[indices], color="darkorange", align="center")
    plt.xticks(range(len(indices)), list(np.array(feature_names)[indices]), rotation="vertical")
    plt.xlim([-1, len(indices)])
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.legend().set_visible(False)
    plt.savefig("Feature importances by LightGBM.png", dpi=500)
    # plt.show()
def Predict_LGBM_func(model_path, _img_):
    # 以读二进制的方式打开文件
    file = open(model_path, "rb")
    # 把模型从文件中读取出来
    lgbm_model = pickle.load(file)
    # 关闭文件
    file.close()
    # 用读入的模型进行预测
    # 在与测试前要调整一下数据的格式
    data = np.zeros((_img_.shape[0], _img_.shape[1] * _img_.shape[2]))
    # 使用tqdm显示进度
    for i in range(_img_.shape[0]):
        data[i] = _img_[i].flatten()
    data = data.swapaxes(0, 1)
    # 对调整好格式的数据进行预测
    pred = lgbm_model.predict(data)
    # 同样地，我们对预测好的数据调整为我们图像的格式
    pred = pred.reshape(_img_.shape[1], _img_.shape[2])
    pred = pred.astype(np.uint8)
    return pred
# ===================== ui_functions.py =====================
# 选择的遥感影像数据路径
def selectImgPath(lineEdit, log_textBrowser):
    try:
        img, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select Image", "", "Image Files(*.tif *.tiff *.pix)"
        )
        if img:
            lineEdit.setText(str(img))
            printLog("选择的遥感影像数据路径:" + str(img), log_textBrowser)
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Error", str(e), QtWidgets.QMessageBox.Ok)
# 选择的标签数据路径
def selectLabelPath(lineEdit, log_textBrowser):
    try:
        label, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select Label", "", "Image Files(*.tif *.tiff *.pix)"
        )
        if label:
            lineEdit.setText(str(label))
            printLog("选择的标签数据路径:" + str(label), log_textBrowser)
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Error", str(e), QtWidgets.QMessageBox.Ok)
# 选择的输出样本数据路径
def selectOutSamplePath(lineEdit, log_textBrowser):
    try:
        # 新建文件的输出路径文件名
        out_sample, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "Save File", "", "Image Files(*.csv *.txt)"
        )
        if out_sample:
            lineEdit.setText(str(out_sample))
            printLog("选择的输出样本数据路径:" + str(out_sample), log_textBrowser)
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Error", str(e), QtWidgets.QMessageBox.Ok)
# 选择的NumClass数据路径
def selectNumClassPath(lineEdit, log_textBrowser):
    try:
        NumClassPath, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select NumClass", "", "Image Files(*.csv *.txt)"
        )
        if NumClassPath:
            lineEdit.setText(str(NumClassPath))
            printLog("选择的NumClass数据路径:" + str(NumClassPath), log_textBrowser)
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Error", str(e), QtWidgets.QMessageBox.Ok)
# 选择的模型保存路径
def selectSaveModelPath(lineEdit, log_textBrowser):
    try:
        model, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "Save Model", "", "Model Files(*.pkl *.pickle)"
        )
        if model:
            lineEdit.setText(str(model))
            printLog("选择的模型保存路径:" + str(model), log_textBrowser)
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Error", str(e), QtWidgets.QMessageBox.Ok)
# 选择的模型路径
def selectModelPath(lineEdit, log_textBrowser):
    try:
        model, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select Model", "", "Model Files(*.pkl *.pickle)"
        )
        if model:
            lineEdit.setText(str(model))
            printLog("选择的模型路径:" + str(model), log_textBrowser)
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Error", str(e), QtWidgets.QMessageBox.Ok)
# 选择的保存影像数据路径
def selectSaveImgPath(lineEdit, log_textBrowser):
    try:
        img, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "Save Image", "", "Image Files(*.tif *.tiff *.pix)"
        )
        if img:
            lineEdit.setText(str(img))
            printLog("选择的保存影像数据路径:" + str(img), log_textBrowser)
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Error", str(e), QtWidgets.QMessageBox.Ok)
# get samples clicked
# 获取样本
def GetSamplesF(
    img_lineEdit,
    label_lineEdit,
    out_sample_lineEdit,
    numclass_lineEdit,
    msg,
):
    try:
        img_path = img_lineEdit
        label_path = label_lineEdit
        sample_path = out_sample_lineEdit
        numclass_path = numclass_lineEdit
        if os.path.exists(img_path) and os.path.exists(label_path):
            msg.emit("*" * 10 + " 开始获取样本数据 " + "*" * 10)
            res = get_samples(img_path, label_path, sample_path, numclass_path, msg)
            if res:
                msg.emit("*" * 10 + " 样本数据获取成功! " + "*" * 10)
            else:
                msg.emit("*" * 10 + " 样本数据获取失败! " + "*" * 10)
        else:
            msg.emit("*" * 10 + " 请检查输入路径是否正确 " + "*" * 10)
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Error", str(e), QtWidgets.QMessageBox.Ok)
def float_range(start, end, step):
    return [start + step * i for i in range(int((end - start) / step))]
