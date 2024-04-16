from utils import *


def RF_Predict(RF_model_path, img_path, save_path, class_path, msg):
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
    pred = Predict_RF_func(RF_model_path, _img_)
    msg.emit("*" * 30 + "预测完毕" + "*" * 30)
    # Plot展示
    PlotPredictResult(pred, "RandomForest", class_path)
    #  将结果写到tif图像里
    msg.emit("*" * 30 + "写入" + "*" * 30)
    writeTiff(pred, img_geotrans, img_projection, save_path)
    msg.emit("*" * 30 + "写入完毕" + "*" * 30)
    end_time = time.time()
    msg.emit("*" * 25 + " 预测运行时间:" + str(end_time - start_time) + " s " + "*" * 25)


def XGB_Predict(XGB_model_path, img_Path, SavePath, classPath, msg):
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
    PlotPredictResult(pred, "XGBoost", classPath)
    #  将结果写到tif图像里
    msg.emit("*" * 30 + "写入" + "*" * 30)
    writeTiff(pred, img_geotrans, img_projection, SavePath)
    msg.emit("*" * 30 + "写入完毕" + "*" * 30)
    end_time = time.time()
    msg.emit("*" * 25 + "程序运行时间：" + str(end_time - start_time) + "s" + "*" * 25)


def LightGBM_Predict(LGBM_model_path, img_path, save_path, classPath, msg):
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
    PlotPredictResult(pred, "lightGBM", classPath)
    #  将结果写到tif图像里
    msg.emit("*" * 30 + "写入" + "*" * 30)
    writeTiff(pred, img_geotrans, img_projection, save_path)
    msg.emit("*" * 30 + "写入完毕" + "*" * 30)
    end_time = time.time()
    msg.emit("*" * 25 + "程序运行时间：" + str(end_time - start_time) + "s" + "*" * 25)
