# -*- coding:utf-8 -*-
# Author: jiaoxiaohao
# E-mail: jiaoxiaohao876@gmail.com
# Time: 2024-02-20 03:20:00
# File name: LightGBM_predict.py
# Nothing is true, everything is permitted.

from utils import *
import time
def LightGBM_Predict(LGBM_model_path, img_path, save_path, class_path, msg):
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
    img_width, img_height, img_bands, img_geotrans, _img_, img_projection = get_inf(img_path)
    msg.emit("*" * 30 + "读取完毕" + "*" * 30)
    # 预测
    msg.emit("*" * 30 + "预测中..." + "*" * 30)
    pred = Predict_LGBM_func(LGBM_model_path, _img_)
    # pred结果加1，还原为真实类别
    pred = pred + 1
    msg.emit("*" * 30 + "预测完毕" + "*" * 30)
    # Plot展示
    # PlotPredictResult(pred, classPath)
    #  将结果写到tif图像里
    msg.emit("*" * 30 + "写入" + "*" * 30)
    writeTiff(pred, img_geotrans, img_projection, save_path)
    msg.emit("*" * 30 + "写入完毕" + "*" * 30)
    end_time = time.time()
    msg.emit("*" * 25 + "程序运行时间：" + str(end_time - start_time) +"s"+"*" * 25)

    if __name__ == "__main__":
        LGBM_model_path = "data/model_gbm.pickle"
        img_path = "data/img.tif"
        save_path = "data/predict.tif"
        class_path = "data/ClassDefine.txt"
        msg = ""
        LightGBM_Predict(LGBM_model_path, img_path, save_path, class_path, msg)