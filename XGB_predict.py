# -*- coding:utf-8 -*-
# Author: jiaoxiaohao
# E-mail: jiaoxiaohao876@gmail.com
# Time: 2024-02-17 01:11:35
# File name: XGB_predict.py
# Nothing is true, everything is permitted.

from utils import *

def XGB_Predict(XGB_model_path, img_Path, SavePath, classPath,msg):
    def get_inf(Landset_Path):
        # 读取影像数据
        dataset = gdal.Open(Landset_Path)
        img_width = dataset.RasterXSize
        img_height = dataset.RasterYSize
        img_bands = dataset.RasterCount
        img_geotrans = dataset.GetGeoTransform()
        img_projection = dataset.GetProjection()
        _img_ = dataset.ReadAsArray(0, 0, img_width, img_height)
        return img_width, img_height, img_bands, img_geotrans, _img_, img_projection

    start_time = time.time()
    msg.emit("*" * 30 + "读取影像数据" + "*" * 30)
    img_width, img_height, img_bands, img_geotrans, _img_, img_projection = get_inf(img_Path)
    msg.emit("*" * 30 + "读取完毕" + "*" * 30)
    # 预测
    msg.emit("*" * 30 + "预测" + "*" * 30)
    pred = Predict_XGB_func(XGB_model_path, _img_)
    msg.emit("*" * 30 + "预测完毕" + "*" * 30)
    # Plot展示
    # PlotPredictResult(pred, classPath)
    #  将结果写到tif图像里
    msg.emit("*" * 30 + "写入" + "*" * 30)
    writeTiff(pred, img_geotrans, img_projection, SavePath)
    msg.emit("*" * 30 + "写入完毕" + "*" * 30)
    end_time = time.time()
    msg.emit("*" * 25 + "程序运行时间：" + str(end_time - start_time) +"s"+"*" * 25)

if __name__ == '__main__':
    XGB_model_path = "data\\model.pickle"
    img_Path = "data\\img.tif"
    SavePath = "data\\save02.tif"
    classPath = "data\\ClassDefine.txt"
    msg = ""
    XGB_Predict(XGB_model_path, img_Path, SavePath, classPath,msg)