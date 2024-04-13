from utils import *
from LightGBM_train import LightGBM_train
from XGB_train import XGB_train
from RF_train import RF_train
from RF_predict import RF_Predict
from XGB_predict import XGB_Predict
from LightGBM_predict import LightGBM_Predict

# 获取样本数据的线程
class GetSamplesFThread(QtCore.QThread):
    error = QtCore.pyqtSignal(str)
    msg = QtCore.pyqtSignal(str)

    def __init__(self, img_path, mark_path, out_sample_path, numclass_path):
        super(GetSamplesFThread, self).__init__()
        self.img_path = img_path
        self.mark_path = mark_path
        self.out_sample_path = out_sample_path
        self.numclass_path = numclass_path

    def run(self):
        try:
            GetSamplesF(self.img_path, self.mark_path, self.out_sample_path, self.numclass_path, self.msg)
        except Exception as e:
            self.error.emit(str(e))


class Train(QtCore.QThread):
    error = QtCore.pyqtSignal(str)
    msg = QtCore.pyqtSignal(str)

    def __init__(self, sample_path, SavePath, space, test_size, model):
        super(Train, self).__init__()
        self.sample_path = sample_path
        self.SavePath = SavePath
        self.test_size = test_size
        self.space = space
        self.model = model

    def run(self):
        try:
            if self.model == "LightGBM":
                LightGBM_train(
                    self.sample_path,
                    self.SavePath,
                    self.space,
                    self.msg,
                    test_size=self.test_size,
                )
            elif self.model == "XGBoost":
                XGB_train(
                    self.sample_path,
                    self.SavePath,
                    self.space,
                    self.msg,
                    test_size=self.test_size,
                )
            elif self.model == "RandomForest":
                RF_train(
                    self.sample_path,
                    self.SavePath,
                    self.space,
                    self.msg,
                    test_size=self.test_size,
                )
        except Exception as e:
            self.error.emit(str(e))


class Predict(QtCore.QThread):
    error = QtCore.pyqtSignal(str)
    msg = QtCore.pyqtSignal(str)

    def __init__(self, RF_model_path, img_path, save_path, class_path, model):
        super(Predict, self).__init__()
        self.RF_model_path = RF_model_path
        self.img_path = img_path
        self.save_path = save_path
        self.class_path = class_path
        self.model = model

    def run(self):
        try:
            if self.model == "RandomForest":
                RF_Predict(
                    self.RF_model_path, 
                    self.img_path, 
                    self.save_path, 
                    self.class_path, 
                    self.msg)
            elif self.model == "XGBoost":
                XGB_Predict(
                    self.RF_model_path,
                    self.img_path,
                    self.save_path,
                    self.class_path,
                    self.msg
                )
            elif self.model == "LightGBM":
                LightGBM_Predict(
                    self.RF_model_path,
                    self.img_path,
                    self.save_path,
                    self.class_path,
                    self.msg
                )
        except Exception as e:
            self.error.emit(str(e))
