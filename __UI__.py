# -*- coding: utf-8 -*-
# Author: jiaoxiaohao
# E-mail: jiaoxiaohao876@gmail.com
# Time: 2024-03-10 19:19:15
# File name: Ui___UI__.py
# Nothing is true, everything is permitted.

from PyQt5 import QtCore, QtGui, QtWidgets
from classes import *


class ClassificationUI(object):
    def ThreadLog(self, msg):
        self.log_text_browser.append(msg)
    def SaveLogButton_clicked(self):
        # 获取当前text browser的文本

        # 打开一个保存文件选择的框
        try:
            # 获取当前text browser的文本
            text = self.log_text_browser.toPlainText()
            # 新建文件的输出路径文件名
            out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                None, "Save File", "", "Text Files(*.txt)"
            )
            open(out_path, "w").write(text)
            QtWidgets.QMessageBox.information(None, "成功", "成功保存日志文件!", QtWidgets.QMessageBox.Ok)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", str(e), QtWidgets.QMessageBox.Ok)
    def get_page_space(self):
        if self.tabWidgetPages.currentIndex() == 0:
            space = {
                "n_estimators": hp.choice(
                    "n_estimators",
                    range(
                        int(self.RF_n_estimators_min.text()),
                        int(self.RF_n_estimators_max.text()),
                        int(self.RF_n_estimators_interval.text()),
                    ),
                ),
                "max_depth": hp.choice(
                    "max_depth",
                    range(
                        int(self.RF_max_depth_min.text()),
                        int(self.RF_max_depth_max.text()),
                        int(self.RF_max_depth_interval.text()),
                    ),
                ),
                "min_samples_leaf": hp.choice(
                    "min_samples_leaf",
                    range(
                        int(self.RF_min_samples_leaf_min.text()),
                        int(self.RF_min_samples_leaf_max.text()),
                        int(self.RF_min_samples_leaf_interval.text()),
                    ),
                ),
                "min_samples_split": hp.choice(
                    "min_samples_split",
                    range(
                        int(self.RF_min_samples_split_min.text()),
                        int(self.RF_min_samples_split_max.text()),
                        int(self.RF_min_samples_split_interval.text()),
                    ),
                ),
            }
            max_features = []
            if self.sqrt_cb.isChecked():
                max_features.append("sqrt")
            if self.log2_cb.isChecked():
                max_features.append("log2")
            if self.None_cb.isChecked():
                max_features.append(None)
            space["max_features"] = hp.choice("max_features", max_features)
            bootstrap = []
            if self.True_cb.isChecked():
                bootstrap.append(True)
            if self.False_cb.isChecked():
                bootstrap.append(False)
            space["bootstrap"] = hp.choice("bootstrap", bootstrap)
            return space
        elif self.tabWidgetPages.currentIndex() == 1:
            space = {
                "n_estimators": hp.choice(
                    "n_estimators",
                    range(
                        int(self.LGB_n_estimators_min.text()),
                        int(self.LGB_n_estimators_max.text()),
                        int(self.LGB_n_estimators_interval.text()),
                    ),
                ),
                "max_depth": hp.choice(
                    "max_depth",
                    range(
                        int(self.LGB_max_depth_min.text()),
                        int(self.LGB_max_depth_max.text()),
                        int(self.LGB_max_depth_interval.text()),
                    ),
                ),
                "learning_rate": hp.choice(
                    "learning_rate",
                    float_range(
                        float(self.LGB_learning_rate_min.text()),
                        float(self.LGB_learning_rate_Max.text()),
                        float(self.LGB_learning_rate_Interval.text()),
                    ),
                ),
                "min_child_weight": hp.choice(
                    "min_child_weight",
                    range(
                        int(self.LGB_min_child_weight_min.text()),
                        int(self.LGB_min_child_weight_max.text()),
                        int(self.LGB_min_child_weight_interval.text()),
                    ),
                ),
                "subsample": hp.choice(
                    "subsample",
                    float_range(
                        float(self.LGB_subsample_min.text()),
                        float(self.LGB_subsample_max.text()),
                        float(self.LGB_subsample_interval.text()),
                    ),
                ),
                "colsample_bytree": hp.choice(
                    "colsample_bytree",
                    float_range(
                        float(self.LGB_colsample_bytree_min.text()),
                        float(self.LGB_colsample_bytree_max.text()),
                        float(self.LGB_colsample_bytree_interval.text()),
                    ),
                ),
                "reg_alpha": hp.choice(
                    "reg_alpha",
                    range(
                        int(self.LGB_reg_alpha_min.text()),
                        int(self.LGB_reg_alpha_max.text()),
                        int(self.LGB_reg_alpha_interval.text()),
                    ),
                ),
                "reg_lambda": hp.choice(
                    "reg_lambda",
                    range(
                        int(self.LGB_reg_lambda_min.text()),
                        int(self.LGB_reg_lambda_max.text()),
                        int(self.LGB_reg_lambda_interval.text()),
                    ),
                ),
                "objective": "multiclass",
                # "metric": "multi_logloss",
                # "boosting_type": "gbdt",
            }
            # print(space['learning_rate'])
            return space
        elif self.tabWidgetPages.currentIndex() == 2:
            space = {
                "n_estimators": hp.choice(
                    "n_estimators",
                    range(
                        int(self.XGB_n_estimators_min.text()),
                        int(self.XGB_n_estimators_max.text()),
                        int(self.XGB_n_estimators_interval.text()),
                    ),
                ),
                "max_depth": hp.choice(
                    "max_depth",
                    range(
                        int( self.XGB_max_depth_min.text()),
                        int(self.XGB_max_depth_max.text()),
                        int(self.XGB_max_depth_interval.text()),
                    ),
                ),
                "learning_rate": hp.choice(
                    "learning_rate",
                    float_range(
                        float(self.XGB_learning_rate_min.text()),
                        float(self.XGB_learning_rate_Max.text()),
                        float(self.XGB_learning_rate_Interval.text()),
                    ),
                ),
                "min_child_weight": hp.choice(
                    "min_child_weight",
                    range(
                        int(self.XGB_min_child_weight_min.text()),
                        int(self.XGB_min_child_weight_max.text()),
                        int(self.XGB_min_child_weight_interval.text()),
                    ),
                ),
                "subsample": hp.choice(
                    "subsample",
                    float_range(
                        float(self.XGB_subsample_min.text()),
                        float(self.XGB_subsample_max.text()),
                        float(self.XGB_subsample_interval.text()),
                    ),
                ),
                "colsample_bytree": hp.choice(
                    "colsample_bytree",
                    float_range(
                        float(self.XGB_colsample_bytree_min.text()),
                        float(self.XGB_colsample_bytree_max.text()),
                        float(self.XGB_colsample_bytree_interval.text()),
                    ),
                ),
                "gamma": hp.choice(
                    "gamma",
                    range(
                        int(self.XGB_gamma_min.text()),
                        int(self.XGB_gamma_max.text()),
                        int(self.XGB_gamma_interval.text()),
                    ),
                ),
                "reg_alpha": hp.choice(
                    "reg_alpha",
                    range(
                        int(self.XGB_reg_alpha_min.text()),
                        int(self.XGB_reg_alpha_max.text()),
                        int(self.XGB_reg_alpha_interval.text()),
                    ),
                ),
                "reg_lambda": hp.choice(
                    "reg_lambda",
                    range(
                        int(self.XGB_reg_lambda_min.text()),
                        int(self.XGB_reg_lambda_max.text()),
                        int(self.XGB_reg_lambda_interval.text()),
                    ),
                ),
                "objective": "multiclass",
                # "metric": "multi_logloss",
                # "boosting_type": "gbdt",
            }
            return space

    # 随机森林数据获取部分的按钮点击链接
    def RF_select_image_pushbutton_clicked(self):
        selectImgPath(self.RF_img_path, self.log_text_browser)

    def RF_select_label_pushbutton_clicked(self):
        selectLabelPath(self.RF_label_path, self.log_text_browser)

    def RF_select_numclass_path_pushButton_clicked(self):
        selectNumClassPath(self.RF_class_num_path, self.log_text_browser)

    def RF_select_sample_path_pushButton_clicked(self):
        selectOutSamplePath(self.RF_output_samples_path, self.log_text_browser)

    def select_RF_train_Save_model_path_pushButton_clicked(self):
        selectSaveModelPath(self.RF_train_model_save_path, self.log_text_browser)

    def select_RF_model_path_pushButton_clicked(self):
        selectModelPath(self.RF_model_path, self.log_text_browser)

    def select_RF_class_num_pushButton_clicked(self):
        selectNumClassPath(self.RF_class_num_path_2, self.log_text_browser)

    def select_RF_predict_result_output_path_pushButton_clicked(self):
        selectSaveImgPath(self.RF_output_predict_result_path, self.log_text_browser)

    def select_RF_img_predict_path_pushButton_clicked(self):
        selectImgPath(self.RF_img_predict_path, self.log_text_browser)

    def RF_get_sample_pushButton_clicked(self):
        # 判断是否有路径
        if (
            not self.RF_img_path.text()
            or not self.RF_label_path.text()
            or not self.RF_output_samples_path.text()
            or not self.RF_class_num_path.text()
        ):
            self.log_text_browser.append("请检查路径!!!!")
            return
        # 开启线程
        self.getSamplesFThread = GetSamplesFThread(
            self.RF_img_path.text(),
            self.RF_label_path.text(),
            self.RF_output_samples_path.text(),
            self.RF_class_num_path.text(),
        )
        self.getSamplesFThread.error.connect(self.ThreadLog)
        self.getSamplesFThread.msg.connect(self.ThreadLog)
        self.getSamplesFThread.start()

    def RF_train_pushButton_clicked(self):
        # 判断是否有路径
        if (
            not self.RF_output_samples_path.text()
            or not self.RF_train_model_save_path.text()
            or not self.RF_val_datasets_size.text()
        ):
            self.log_text_browser.append("请检查路径!!!!")
            return
        # 判断data_size是否为浮点数字
        try:
            float(self.RF_val_datasets_size.text())
        except:
            self.log_text_browser.append("请输入正确的数据集大小!!!!")
            return
        # 开启线程
        self.trainThread = Train(
            self.RF_output_samples_path.text(),
            self.RF_train_model_save_path.text(),
            self.get_page_space(),
            float(self.RF_val_datasets_size.text()),
            "RandomForest",
        )
        self.trainThread.error.connect(self.ThreadLog)
        self.trainThread.msg.connect(self.ThreadLog)
        self.trainThread.start()

    def RF_predict_pushButton_clicked(self):
        # 判断是否有路径
        if (
            not self.RF_model_path.text()
            or not self.RF_img_predict_path.text()
            or not self.RF_output_predict_result_path.text()
            or not self.RF_class_num_path_2.text()
        ):
            self.log_text_browser.append("请检查路径!!!!")
            return
        # 开启线程
        self.predictThread = Predict(
            self.RF_model_path.text(),
            self.RF_img_predict_path.text(),
            self.RF_output_predict_result_path.text(),
            self.RF_class_num_path_2.text(),
            "RandomForest",
        )
        self.predictThread.error.connect(self.ThreadLog)
        self.predictThread.msg.connect(self.ThreadLog)
        self.predictThread.start()

    # XGB数据获取部分的按钮点击链接
    def XGB_select_image_pushbutton_clicked(self):
        selectImgPath(self.XGB_img_path, self.log_text_browser)

    def XGB_select_label_pushbutton_clicked(self):
        selectLabelPath(self.XGB_label_path, self.log_text_browser)

    def XGB_select_numclass_path_pushbutton_clicked(self):
        selectNumClassPath(self.XGB_class_num_path, self.log_text_browser)

    def XGB_select_sample_path_pushbutton_clicked(self):
        selectOutSamplePath(self.XGB_output_samples_path, self.log_text_browser)

    def select_XGB_train_Save_model_path_pushButton_clicked(self):
        selectSaveModelPath(self.XGB_train_model_save_path, self.log_text_browser)

    def select_XGB_model_path_pushButton_clicked(self):
        selectModelPath(self.XGB_model_path, self.log_text_browser)

    def select_XGB_class_num_button_2_clicked(self):
        selectNumClassPath(self.XGB_class_num_path_2, self.log_text_browser)

    def XGB_select_predict_result_output_path_pushButton_clicked(self):
        selectSaveImgPath(self.XGB_output_predict_result_path, self.log_text_browser)

    def select_XGB_img_predict_path_pushButton_clicked(self):
        selectImgPath(self.XGB_img_predict_path, self.log_text_browser)

    def XGB_get_sample_pushButton_clicked(self):
        # 判断是否有路径
        if (
            not self.XGB_img_path.text()
            or not self.XGB_label_path.text()
            or not self.XGB_output_samples_path.text()
            or not self.XGB_class_num_path.text()
        ):
            self.log_text_browser.append("请检查路径!!!!")
            return
        # 开启线程
        self.getSamplesFThread = GetSamplesFThread(
            self.XGB_img_path.text(),
            self.XGB_label_path.text(),
            self.XGB_output_samples_path.text(),
            self.XGB_class_num_path.text(),
        )
        self.getSamplesFThread.error.connect(self.ThreadLog)
        self.getSamplesFThread.msg.connect(self.ThreadLog)
        self.getSamplesFThread.start()

    def XGB_train_pushButton_clicked(self):
        # 判断是否有路径
        if (
            not self.XGB_output_samples_path.text()
            or not self.XGB_train_model_save_path.text()
            or not self.XGB_val_datasets_size.text()
        ):
            self.log_text_browser.append("请检查路径!!!!")
            return
        # 判断data_size是否为浮点数字
        try:
            float(self.XGB_val_datasets_size.text())
        except:
            self.log_text_browser.append("请输入正确的数据集大小!!!!")
            return
        # 开启线程
        self.trainThread = Train(
            self.XGB_output_samples_path.text(),
            self.XGB_train_model_save_path.text(),
            self.get_page_space(),
            float(self.XGB_val_datasets_size.text()),
            "XGBoost",
        )
        self.trainThread.error.connect(self.ThreadLog)
        self.trainThread.msg.connect(self.ThreadLog)
        self.trainThread.start()

    def XGB_predict_pushButton_clicked(self):
        # 判断是否有路径
        if (
            not self.XGB_model_path.text()
            or not self.XGB_img_predict_path.text()
            or not self.XGB_output_predict_result_path.text()
            or not self.XGB_class_num_path_2.text()
        ):
            self.log_text_browser.append("请检查路径!!!!")
            return
        # 开启线程
        self.predictThread = Predict(
            self.XGB_model_path.text(),
            self.XGB_img_predict_path.text(),
            self.XGB_output_predict_result_path.text(),
            self.XGB_class_num_path_2.text(),
            "XGBoost",
        )
        self.predictThread.error.connect(self.ThreadLog)
        self.predictThread.msg.connect(self.ThreadLog)
        self.predictThread.start()
    # LGB数据获取部分的按钮点击链接
    def LGB_select_image_pushbutton_clicked(self):
        selectImgPath(self.LGB_img_path, self.log_text_browser)

    def LGB_select_label_pushbutton_clicked(self):
        selectLabelPath(self.LGB_label_path, self.log_text_browser)

    def LGB_select_numclass_path_pushbutton_clicked(self):
        selectNumClassPath(self.LGB_class_num_path, self.log_text_browser)

    def LGB_select_sample_path_pushbutton_clicked(self):
        selectOutSamplePath(self.LGB_output_samples_path, self.log_text_browser)

    def select_LGB_train_Save_model_path_pushButton_clicked(self):
        selectSaveModelPath(self.LGB_train_model_save_path, self.log_text_browser)

    def select_LGB_model_path_pushbutton_clicked(self):
        selectModelPath(self.LGB_model_path, self.log_text_browser)

    def select_LGB_class_num_button_2_clicked(self):
        selectNumClassPath(self.LGB_class_num_path_2, self.log_text_browser)
    
    def LGB_select_predict_result_output_path_pushbutton_clicked(self):
        selectSaveImgPath(self.LGB_output_predict_result_path, self.log_text_browser)

    def select_LGB_img_predict_path_button_clicked(self):
        selectImgPath(self.LGB_img_predict_path, self.log_text_browser)

    def LGB_get_sample_pushButton_clicked(self):
        # 判断是否有路径
        if (
            not self.LGB_img_path.text()
            or not self.LGB_label_path.text()
            or not self.LGB_output_samples_path.text()
            or not self.LGB_class_num_path.text()
        ):
            self.log_text_browser.append("请检查路径!!!!")
            return
        # 开启线程
        self.getSamplesFThread = GetSamplesFThread(
            self.LGB_img_path.text(),
            self.LGB_label_path.text(),
            self.LGB_output_samples_path.text(),
            self.LGB_class_num_path.text(),
        )
        self.getSamplesFThread.error.connect(self.ThreadLog)
        self.getSamplesFThread.msg.connect(self.ThreadLog)
        self.getSamplesFThread.start()

    def LGB_train_pushButton_clicked(self):
        # 判断是否有路径
        if (
            not self.LGB_output_samples_path.text()
            or not self.LGB_train_model_save_path.text()
            or not self.LGB_val_datasets_size.text()
        ):
            self.log_text_browser.append("请检查路径!!!!")
            return
        # 判断data_size是否为浮点数字
        try:
            float(self.LGB_val_datasets_size.text())
        except:
            self.log_text_browser.append("请输入正确的数据集大小!!!!")
            return
        # 开启线程
        self.trainThread = Train(
            self.LGB_output_samples_path.text(),
            self.LGB_train_model_save_path.text(),
            self.get_page_space(),
            float(self.LGB_val_datasets_size.text()),
            "LightGBM",
        )
        self.trainThread.error.connect(self.ThreadLog)
        self.trainThread.msg.connect(self.ThreadLog)
        self.trainThread.start()

    def LGB_predict_pushButton_clicked(self):
        # 判断是否有路径
        if (
            not self.LGB_model_path.text()
            or not self.LGB_img_predict_path.text()
            or not self.LGB_output_predict_result_path.text()
            or not self.LGB_class_num_path_2.text()
        ):
            self.log_text_browser.append("请检查路径!!!!")
            return
        # 开启线程
        self.predictThread = Predict(
            self.LGB_model_path.text(),
            self.LGB_img_predict_path.text(),
            self.LGB_output_predict_result_path.text(),
            self.LGB_class_num_path_2.text(),
            "LightGBM",
        )
        self.predictThread.error.connect(self.ThreadLog)
        self.predictThread.msg.connect(self.ThreadLog)
        self.predictThread.start()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1300, 910)
        # 固定窗口大小
        MainWindow.setFixedSize(1300, 910)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.centralWidget.setObjectName("centralWidget")
        self.tabWidgetPages = QtWidgets.QTabWidget(self.centralWidget)
        self.tabWidgetPages.setGeometry(QtCore.QRect(20, 10, 480, 890))
        self.tabWidgetPages.setObjectName("tabWidgetPages")

        # RF_tab_page
        self.RF_tab_page = QtWidgets.QWidget()
        self.RF_tab_page.setObjectName("RF_tab_page")
        self.RF_tab_page_init()
        self.tabWidgetPages.addTab(self.RF_tab_page, "")

        # LGBM_tab_page
        self.LGB_tab_page = QtWidgets.QWidget()
        self.LGB_tab_page.setObjectName("LGB_tab_page")
        self.LightGBM_tab_page_init()
        self.tabWidgetPages.addTab(self.LGB_tab_page, "")

        #### ======================================== ========================================

        # XGB_tab_page
        self.XGB_tab_page = QtWidgets.QWidget()
        self.XGB_tab_page.setObjectName("XGB_tab_page")
        self.XGB_tab_page_init()
        self.tabWidgetPages.addTab(self.XGB_tab_page, "")

        # 设置日志界面
        self.log_group = QtWidgets.QGroupBox(self.centralWidget)
        self.log_group.setGeometry(QtCore.QRect(530, 10, 750, 890))
        self.log_group.setObjectName("log_group")
        self.log_text_browser = QtWidgets.QTextBrowser(self.log_group)
        self.log_text_browser.setGeometry(QtCore.QRect(10, 20, 730, 830))
        self.log_text_browser.setObjectName("log_text_browser")
        self.SaveLogButton = QtWidgets.QPushButton(self.log_group)
        self.SaveLogButton.setGeometry(QtCore.QRect(330, 855, 90, 30))
        self.SaveLogButton.setObjectName("SaveLogButton")
        self.SaveLogButton.clicked.connect(self.SaveLogButton_clicked)

        MainWindow.setCentralWidget(self.centralWidget)

        # 绑定tabWidgetPagesChanged
        self.tabWidgetPages.setCurrentIndex(0)
        self.tabWidgetPages.currentChanged.connect(self.tabWidgetPagesChanged)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def RF_tab_page_init(self):
        # ======================================== RF_get_samples_group ========================================
        self.RF_get_samples_group = QtWidgets.QGroupBox(self.RF_tab_page)
        self.RF_get_samples_group.setGeometry(QtCore.QRect(10, 10, 460, 225))
        self.RF_get_samples_group.setObjectName("RF_get_samples_group")
        self.RF_img_path_label = QtWidgets.QLabel(self.RF_get_samples_group)
        self.RF_img_path_label.setGeometry(QtCore.QRect(10, 20, 80, 30))
        self.RF_img_path_label.setObjectName("RF_img_path_label")
        self.RF_img_path = QtWidgets.QLineEdit(self.RF_get_samples_group)
        self.RF_img_path.setGeometry(QtCore.QRect(90, 24, 235, 22))
        self.RF_img_path.setObjectName("RF_img_path")
        self.RF_img_path.setReadOnly(True)
        self.select_RF_img_button = QtWidgets.QPushButton(self.RF_get_samples_group)
        self.select_RF_img_button.setGeometry(QtCore.QRect(350, 20, 100, 30))
        self.select_RF_img_button.setObjectName("select_RF_img_button")
        self.select_RF_img_button.clicked.connect(self.RF_select_image_pushbutton_clicked)

        self.RF_label_path_label = QtWidgets.QLabel(self.RF_get_samples_group)
        self.RF_label_path_label.setGeometry(QtCore.QRect(10, 60, 80, 30))
        self.RF_label_path_label.setObjectName("RF_label_path_label")
        self.RF_label_path = QtWidgets.QLineEdit(self.RF_get_samples_group)
        self.RF_label_path.setGeometry(QtCore.QRect(90, 64, 235, 22))
        self.RF_label_path.setObjectName("RF_label_path")
        self.RF_label_path.setReadOnly(True)
        self.select_RF_label_button = QtWidgets.QPushButton(self.RF_get_samples_group)
        self.select_RF_label_button.setGeometry(QtCore.QRect(350, 60, 100, 30))
        self.select_RF_label_button.setObjectName("select_RF_label_button")
        self.select_RF_label_button.clicked.connect(self.RF_select_label_pushbutton_clicked)

        self.RF_class_num_label = QtWidgets.QLabel(self.RF_get_samples_group)
        self.RF_class_num_label.setGeometry(QtCore.QRect(10, 100, 80, 30))
        self.RF_class_num_label.setObjectName("RF_class_num_label")
        self.RF_class_num_path = QtWidgets.QLineEdit(self.RF_get_samples_group)
        self.RF_class_num_path.setGeometry(QtCore.QRect(90, 104, 235, 22))
        self.RF_class_num_path.setObjectName("RF_class_num_path")
        self.RF_class_num_path.setReadOnly(True)
        self.select_RF_class_num_button = QtWidgets.QPushButton(self.RF_get_samples_group)
        self.select_RF_class_num_button.setGeometry(QtCore.QRect(350, 100, 100, 30))
        self.select_RF_class_num_button.setObjectName("select_RF_class_num_button")
        self.select_RF_class_num_button.clicked.connect(self.RF_select_numclass_path_pushButton_clicked)

        self.RF_output_samples_path_label = QtWidgets.QLabel(self.RF_get_samples_group)
        self.RF_output_samples_path_label.setGeometry(QtCore.QRect(10, 140, 80, 30))
        self.RF_output_samples_path_label.setObjectName("RF_output_samples_path_label")
        self.RF_output_samples_path = QtWidgets.QLineEdit(self.RF_get_samples_group)
        self.RF_output_samples_path.setGeometry(QtCore.QRect(90, 145, 235, 22))
        self.RF_output_samples_path.setObjectName("RF_output_samples_path")
        self.RF_output_samples_path.setReadOnly(True)
        self.select_RF_output_samples_path_button = QtWidgets.QPushButton(self.RF_get_samples_group)
        self.select_RF_output_samples_path_button.setGeometry(QtCore.QRect(350, 141, 100, 30))
        self.select_RF_output_samples_path_button.setObjectName("select_RF_output_samples_path_button")
        self.select_RF_output_samples_path_button.clicked.connect(
            self.RF_select_sample_path_pushButton_clicked
        )

        self.RF_get_samples_button = QtWidgets.QPushButton(self.RF_get_samples_group)
        self.RF_get_samples_button.setGeometry(QtCore.QRect(160, 180, 100, 30))
        self.RF_get_samples_button.setObjectName("get_RF_samples_button")
        self.RF_get_samples_button.clicked.connect(self.RF_get_sample_pushButton_clicked)

        # ======================================== RF_train_group ========================================

        self.RF_train_group = QtWidgets.QGroupBox(self.RF_tab_page)
        self.RF_train_group.setGeometry(QtCore.QRect(10, 240, 460, 320))
        self.RF_train_group.setObjectName("RF_train_group")

        # model save path
        self.RF_train_model_save_path_label = QtWidgets.QLabel(self.RF_train_group)
        self.RF_train_model_save_path_label.setGeometry(QtCore.QRect(10, 245, 90, 25))
        self.RF_train_model_save_path_label.setObjectName("RF_train_model_save_path_label")
        self.RF_train_model_save_path = QtWidgets.QLineEdit(self.RF_train_group)
        self.RF_train_model_save_path.setGeometry(QtCore.QRect(110, 246, 230, 22))
        self.RF_train_model_save_path.setObjectName("RF_train_model_save_path")
        self.RF_train_model_save_path.setReadOnly(True)
        self.select_RF_train_model_save_path_button = QtWidgets.QPushButton(self.RF_train_group)
        self.select_RF_train_model_save_path_button.setGeometry(QtCore.QRect(350, 243, 100, 30))
        self.select_RF_train_model_save_path_button.setObjectName("select_RF_train_model_save_path_button")
        self.select_RF_train_model_save_path_button.clicked.connect(
            self.select_RF_train_Save_model_path_pushButton_clicked
        )

        self.RF_train_button = QtWidgets.QPushButton(self.RF_train_group)
        self.RF_train_button.setGeometry(QtCore.QRect(160, 280, 100, 30))
        self.RF_train_button.setObjectName("RF_train_button")

        self.RF_train_button.clicked.connect(self.RF_train_pushButton_clicked)

        self.RF_val_datasets_size_label = QtWidgets.QLabel(self.RF_train_group)
        self.RF_val_datasets_size_label.setGeometry(QtCore.QRect(10, 20, 100, 20))
        self.RF_val_datasets_size_label.setObjectName("RF_val_datasets_size_label")
        self.RF_val_datasets_size = QtWidgets.QLineEdit(self.RF_train_group)
        self.RF_val_datasets_size.setGeometry(QtCore.QRect(130, 20, 100, 20))
        self.RF_val_datasets_size.setObjectName("val_datasets_size")

        self.RF_n_estimators_label = QtWidgets.QLabel(self.RF_train_group)
        self.RF_n_estimators_label.setGeometry(QtCore.QRect(10, 65, 130, 20))
        self.RF_n_estimators_label.setObjectName("RF_n_estimators_label")
        self.RF_n_estimators_min = QtWidgets.QLineEdit(self.RF_train_group)
        self.RF_n_estimators_min.setGeometry(QtCore.QRect(160, 65, 80, 20))
        self.RF_n_estimators_min.setObjectName("RF_n_estimators_min")
        self.RF_n_estimators_interval = QtWidgets.QLineEdit(self.RF_train_group)
        self.RF_n_estimators_interval.setGeometry(QtCore.QRect(340, 65, 80, 20))
        self.RF_n_estimators_interval.setObjectName("RF_n_estimators_interval")
        self.RF_n_estimators_max = QtWidgets.QLineEdit(self.RF_train_group)
        self.RF_n_estimators_max.setGeometry(QtCore.QRect(250, 65, 80, 20))
        self.RF_n_estimators_max.setObjectName("RF_n_estimators_max")

        self.RF_Min_label = QtWidgets.QLabel(self.RF_train_group)
        self.RF_Min_label.setGeometry(QtCore.QRect(188, 45, 80, 20))
        self.RF_Min_label.setObjectName("Min_label")
        self.RF_Max_label = QtWidgets.QLabel(self.RF_train_group)
        self.RF_Max_label.setGeometry(QtCore.QRect(278, 45, 80, 20))
        self.RF_Max_label.setObjectName("RF_Max_label")
        self.RF_Interval_label = QtWidgets.QLabel(self.RF_train_group)
        self.RF_Interval_label.setGeometry(QtCore.QRect(355, 45, 80, 20))
        self.RF_Interval_label.setObjectName("RF_Interval_label")

        self.RF_max_depth_min = QtWidgets.QLineEdit(self.RF_train_group)
        self.RF_max_depth_min.setGeometry(QtCore.QRect(160, 95, 80, 20))
        self.RF_max_depth_min.setObjectName("max_depth_min")
        self.RF_max_depth_max = QtWidgets.QLineEdit(self.RF_train_group)
        self.RF_max_depth_max.setGeometry(QtCore.QRect(250, 95, 80, 20))
        self.RF_max_depth_max.setObjectName("max_depth_max")
        self.RF_max_depth_interval = QtWidgets.QLineEdit(self.RF_train_group)
        self.RF_max_depth_interval.setGeometry(QtCore.QRect(340, 95, 80, 20))
        self.RF_max_depth_interval.setObjectName("max_depth_interval")
        self.RF_max_depth_label = QtWidgets.QLabel(self.RF_train_group)
        self.RF_max_depth_label.setGeometry(QtCore.QRect(10, 95, 140, 20))
        self.RF_max_depth_label.setObjectName("max_depth_label")

        self.RF_min_samples_leaf_min = QtWidgets.QLineEdit(self.RF_train_group)
        self.RF_min_samples_leaf_min.setGeometry(QtCore.QRect(160, 125, 80, 20))
        self.RF_min_samples_leaf_min.setObjectName("RF_min_samples_leaf_min")
        self.RF_min_samples_leaf_max = QtWidgets.QLineEdit(self.RF_train_group)
        self.RF_min_samples_leaf_max.setGeometry(QtCore.QRect(250, 125, 80, 20))
        self.RF_min_samples_leaf_max.setObjectName("RF_min_samples_leaf_max")
        self.RF_min_samples_leaf_interval = QtWidgets.QLineEdit(self.RF_train_group)
        self.RF_min_samples_leaf_interval.setGeometry(QtCore.QRect(340, 125, 80, 20))
        self.RF_min_samples_leaf_interval.setObjectName("RF_min_samples_leaf_interval")
        self.RF_min_samples_leaf_label = QtWidgets.QLabel(self.RF_train_group)
        self.RF_min_samples_leaf_label.setGeometry(QtCore.QRect(10, 125, 140, 20))
        self.RF_min_samples_leaf_label.setObjectName("RF_min_samples_leaf_label")

        self.RF_min_samples_split_min = QtWidgets.QLineEdit(self.RF_train_group)
        self.RF_min_samples_split_min.setGeometry(QtCore.QRect(160, 155, 80, 20))
        self.RF_min_samples_split_min.setObjectName("RF_min_samples_split_min")
        self.RF_min_samples_split_max = QtWidgets.QLineEdit(self.RF_train_group)
        self.RF_min_samples_split_max.setGeometry(QtCore.QRect(250, 155, 80, 20))
        self.RF_min_samples_split_max.setObjectName("RF_min_samples_split_max")
        self.RF_min_samples_split_interval = QtWidgets.QLineEdit(self.RF_train_group)
        self.RF_min_samples_split_interval.setGeometry(QtCore.QRect(340, 155, 80, 20))
        self.RF_min_samples_split_interval.setObjectName("RF_min_samples_split_interval")
        self.RF_min_samples_split_label = QtWidgets.QLabel(self.RF_train_group)
        self.RF_min_samples_split_label.setGeometry(QtCore.QRect(10, 155, 140, 20))
        self.RF_min_samples_split_label.setObjectName("min_samples_split_label")

        self.sqrt_cb = QtWidgets.QCheckBox(self.RF_train_group)
        self.sqrt_cb.setGeometry(QtCore.QRect(160, 185, 80, 20))
        self.sqrt_cb.setObjectName("sqrt_cb")
        # 选中
        self.sqrt_cb.setChecked(True)

        self.log2_cb = QtWidgets.QCheckBox(self.RF_train_group)
        self.log2_cb.setGeometry(QtCore.QRect(250, 185, 80, 20))
        self.log2_cb.setObjectName("log2_cb")
        # 选中
        self.log2_cb.setChecked(True)

        self.None_cb = QtWidgets.QCheckBox(self.RF_train_group)
        self.None_cb.setGeometry(QtCore.QRect(340, 185, 80, 20))
        self.None_cb.setObjectName("None_cb")
        # 选中
        self.None_cb.setChecked(True)

        self.RF_max_features_label = QtWidgets.QLabel(self.RF_train_group)
        self.RF_max_features_label.setGeometry(QtCore.QRect(10, 185, 140, 20))
        self.RF_max_features_label.setObjectName("max_features_label")

        self.False_cb = QtWidgets.QCheckBox(self.RF_train_group)
        self.False_cb.setGeometry(QtCore.QRect(250, 215, 90, 20))
        self.False_cb.setObjectName("False_cb")
        # 选中
        self.False_cb.setChecked(True)

        self.True_cb = QtWidgets.QCheckBox(self.RF_train_group)
        self.True_cb.setGeometry(QtCore.QRect(160, 215, 90, 20))
        self.True_cb.setObjectName("True_cb")
        # 选中
        self.True_cb.setChecked(True)

        self.bootstrap_label = QtWidgets.QLabel(self.RF_train_group)
        self.bootstrap_label.setGeometry(QtCore.QRect(10, 215, 130, 20))
        self.bootstrap_label.setObjectName("bootstrap_label")

        # ======================================== RF_predict_group ========================================
        self.RF_predict_group = QtWidgets.QGroupBox(self.RF_tab_page)
        self.RF_predict_group.setGeometry(QtCore.QRect(10, 580, 460, 260))
        self.RF_predict_group.setObjectName("RF_predict_group")

        self.RF_model_path_label = QtWidgets.QLabel(self.RF_predict_group)
        self.RF_model_path_label.setGeometry(QtCore.QRect(10, 40, 80, 30))
        self.RF_model_path_label.setObjectName("RF_model_path_label")
        self.RF_model_path = QtWidgets.QLineEdit(self.RF_predict_group)
        self.RF_model_path.setGeometry(QtCore.QRect(90, 44, 235, 22))
        self.RF_model_path.setObjectName("RF_model_path")
        self.RF_model_path.setReadOnly(True)
        self.select_RF_model_path_button = QtWidgets.QPushButton(self.RF_predict_group)
        self.select_RF_model_path_button.setGeometry(QtCore.QRect(350, 40, 100, 30))
        self.select_RF_model_path_button.setObjectName("select_RF_model_path_button")
        self.select_RF_model_path_button.clicked.connect(self.select_RF_model_path_pushButton_clicked)

        self.RF_img_predict_path_label = QtWidgets.QLabel(self.RF_predict_group)
        self.RF_img_predict_path_label.setGeometry(QtCore.QRect(10, 80, 80, 30))
        self.RF_img_predict_path_label.setObjectName("RF_img_predict_path_label")
        self.RF_img_predict_path = QtWidgets.QLineEdit(self.RF_predict_group)
        self.RF_img_predict_path.setGeometry(QtCore.QRect(90, 84, 235, 22))
        self.RF_img_predict_path.setObjectName("RF_img_predict_path")
        self.RF_img_predict_path.setReadOnly(True)
        self.select_RF_img_predict_path_button = QtWidgets.QPushButton(self.RF_predict_group)
        self.select_RF_img_predict_path_button.setGeometry(QtCore.QRect(350, 80, 100, 30))
        self.select_RF_img_predict_path_button.setObjectName("select_RF_img_predict_path_button")
        self.select_RF_img_predict_path_button.clicked.connect(
            self.select_RF_img_predict_path_pushButton_clicked
        )

        self.RF_class_num_label_2 = QtWidgets.QLabel(self.RF_predict_group)
        self.RF_class_num_label_2.setGeometry(QtCore.QRect(10, 120, 80, 30))
        self.RF_class_num_label_2.setObjectName("RF_class_num_label_2")
        self.RF_class_num_path_2 = QtWidgets.QLineEdit(self.RF_predict_group)
        self.RF_class_num_path_2.setGeometry(QtCore.QRect(90, 124, 235, 22))
        self.RF_class_num_path_2.setObjectName("RF_class_num_path_2")
        self.RF_class_num_path_2.setReadOnly(True)
        self.select_RF_class_num_button_2 = QtWidgets.QPushButton(self.RF_predict_group)
        self.select_RF_class_num_button_2.setGeometry(QtCore.QRect(350, 120, 100, 30))
        self.select_RF_class_num_button_2.setObjectName("select_RF_class_num_button_2")
        self.select_RF_class_num_button_2.clicked.connect(self.select_RF_class_num_pushButton_clicked)

        self.RF_predict_output_result_path_label = QtWidgets.QLabel(self.RF_predict_group)
        self.RF_predict_output_result_path_label.setGeometry(QtCore.QRect(10, 160, 80, 30))
        self.RF_predict_output_result_path_label.setObjectName("RF_predict_output_result_path_label")
        self.RF_output_predict_result_path = QtWidgets.QLineEdit(self.RF_predict_group)
        self.RF_output_predict_result_path.setGeometry(QtCore.QRect(90, 164, 235, 22))
        self.RF_output_predict_result_path.setObjectName("RF_output_predict_result_path")
        self.RF_output_predict_result_path.setReadOnly(True)
        self.RF_select_predict_result_output_path_button = QtWidgets.QPushButton(self.RF_predict_group)
        self.RF_select_predict_result_output_path_button.setGeometry(QtCore.QRect(350, 160, 100, 30))
        self.RF_select_predict_result_output_path_button.setObjectName(
            "RF_select_predict_result_output_path_button"
        )
        self.RF_select_predict_result_output_path_button.clicked.connect(
            self.select_RF_predict_result_output_path_pushButton_clicked
        )

        self.RF_predict_button = QtWidgets.QPushButton(self.RF_predict_group)
        self.RF_predict_button.setGeometry(QtCore.QRect(160, 200, 100, 30))
        self.RF_predict_button.setObjectName("RF_predict_button")
        self.RF_predict_button.clicked.connect(self.RF_predict_pushButton_clicked)

    def LightGBM_tab_page_init(self):
        # ======================================== LGB_get_samples_group ========================================
        self.LGB_get_samples_group = QtWidgets.QGroupBox(self.LGB_tab_page)
        self.LGB_get_samples_group.setGeometry(QtCore.QRect(10, 10, 460, 220))
        self.LGB_get_samples_group.setObjectName("LGB_get_samples_group")
        self.LGB_img_path_label = QtWidgets.QLabel(self.LGB_get_samples_group)
        self.LGB_img_path_label.setGeometry(QtCore.QRect(10, 20, 460, 30))
        self.LGB_img_path_label.setObjectName("LGB_img_path_label")
        self.LGB_img_path = QtWidgets.QLineEdit(self.LGB_get_samples_group)
        self.LGB_img_path.setGeometry(QtCore.QRect(90, 24, 235, 22))
        self.LGB_img_path.setObjectName("LGB_img_path")
        self.LGB_img_path.setReadOnly(True)
        self.select_LGB_img_button = QtWidgets.QPushButton(self.LGB_get_samples_group)
        self.select_LGB_img_button.setGeometry(QtCore.QRect(350, 20, 100, 30))
        self.select_LGB_img_button.setObjectName("select_LGB_img_button")
        self.select_LGB_img_button.clicked.connect(self.LGB_select_image_pushbutton_clicked)

        self.LGB_label_path_label = QtWidgets.QLabel(self.LGB_get_samples_group)
        self.LGB_label_path_label.setGeometry(QtCore.QRect(10, 60, 80, 30))
        self.LGB_label_path_label.setObjectName("LGB_label_path_label")
        self.LGB_label_path = QtWidgets.QLineEdit(self.LGB_get_samples_group)
        self.LGB_label_path.setGeometry(QtCore.QRect(90, 64, 235, 22))
        self.LGB_label_path.setObjectName("LGB_label_path")
        self.LGB_label_path.setReadOnly(True)
        self.select_LGB_lable_button = QtWidgets.QPushButton(self.LGB_get_samples_group)
        self.select_LGB_lable_button.setGeometry(QtCore.QRect(350, 60, 100, 30))
        self.select_LGB_lable_button.setObjectName("LGB_label_path_label")
        self.select_LGB_lable_button.clicked.connect(self.LGB_select_label_pushbutton_clicked)

        self.LGB_class_num_label = QtWidgets.QLabel(self.LGB_get_samples_group)
        self.LGB_class_num_label.setGeometry(QtCore.QRect(10, 100, 80, 30))
        self.LGB_class_num_label.setObjectName("LGB_class_num_label")
        self.LGB_class_num_path = QtWidgets.QLineEdit(self.LGB_get_samples_group)
        self.LGB_class_num_path.setGeometry(QtCore.QRect(90, 104, 235, 22))
        self.LGB_class_num_path.setObjectName("LGB_class_num_path")
        self.LGB_class_num_path.setReadOnly(True)
        self.select_LGB_class_num_button = QtWidgets.QPushButton(self.LGB_get_samples_group)
        self.select_LGB_class_num_button.setGeometry(QtCore.QRect(350, 100, 100, 30))
        self.select_LGB_class_num_button.setObjectName("select_LGB_class_num_button")
        self.select_LGB_class_num_button.clicked.connect(self.LGB_select_numclass_path_pushbutton_clicked)

        self.LGB_output_samples_path_label = QtWidgets.QLabel(self.LGB_get_samples_group)
        self.LGB_output_samples_path_label.setGeometry(QtCore.QRect(10, 140, 80, 30))
        self.LGB_output_samples_path_label.setObjectName("LGB_output_samples_path_label")
        self.LGB_output_samples_path = QtWidgets.QLineEdit(self.LGB_get_samples_group)
        self.LGB_output_samples_path.setGeometry(QtCore.QRect(90, 144, 235, 22))
        self.LGB_output_samples_path.setObjectName("LGB_output_samples_path")
        self.LGB_output_samples_path.setReadOnly(True)
        self.select_LGB_output_samples_path_button = QtWidgets.QPushButton(self.LGB_get_samples_group)
        self.select_LGB_output_samples_path_button.setGeometry(QtCore.QRect(350, 140, 100, 30))
        self.select_LGB_output_samples_path_button.setObjectName("select_LGB_output_samples_path_button")
        self.select_LGB_output_samples_path_button.clicked.connect(
            self.LGB_select_sample_path_pushbutton_clicked
        )

        self.LGB_get_samples_button = QtWidgets.QPushButton(self.LGB_get_samples_group)
        self.LGB_get_samples_button.setGeometry(QtCore.QRect(160, 180, 100, 30))
        self.LGB_get_samples_button.setObjectName("get_LGB_samples_button")
        self.LGB_get_samples_button.clicked.connect(self.LGB_get_sample_pushButton_clicked)

        # ======================================== LGB_train_group ========================================

        self.LGB_train_group = QtWidgets.QGroupBox(self.LGB_tab_page)
        self.LGB_train_group.setGeometry(QtCore.QRect(10, 235, 460, 400))
        self.LGB_train_group.setObjectName("LGB_train_group")

        # model save path
        self.LGB_train_model_save_path_label = QtWidgets.QLabel(self.LGB_train_group)
        self.LGB_train_model_save_path_label.setGeometry(QtCore.QRect(10, 326, 90, 25))
        self.LGB_train_model_save_path_label.setObjectName("LGB_train_model_save_path_label")
        self.LGB_train_model_save_path = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_train_model_save_path.setGeometry(QtCore.QRect(110, 327, 235, 22))
        self.LGB_train_model_save_path.setObjectName("LGB_train_model_save_path")
        self.LGB_train_model_save_path.setReadOnly(True)
        self.select_LGB_train_model_save_path_button = QtWidgets.QPushButton(self.LGB_train_group)
        self.select_LGB_train_model_save_path_button.setGeometry(QtCore.QRect(350, 323, 100, 30))
        self.select_LGB_train_model_save_path_button.setObjectName("select_LGB_train_model_save_path_button")
        self.select_LGB_train_model_save_path_button.clicked.connect(
            self.select_LGB_train_Save_model_path_pushButton_clicked
        )
        self.LGB_train_button = QtWidgets.QPushButton(self.LGB_train_group)
        self.LGB_train_button.setGeometry(QtCore.QRect(160, 360, 100, 30))
        self.LGB_train_button.setObjectName("LGB_train_button")
        self.LGB_train_button.clicked.connect(self.LGB_train_pushButton_clicked)

        # 训练的参数设置
        # 训练参数比例
        self.LGB_val_datasets_size_label = QtWidgets.QLabel(self.LGB_train_group)
        self.LGB_val_datasets_size_label.setGeometry(QtCore.QRect(10, 30, 100, 20))
        self.LGB_val_datasets_size_label.setObjectName("LGB_val_datasets_size_label")
        self.LGB_val_datasets_size = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_val_datasets_size.setGeometry(QtCore.QRect(130, 30, 100, 20))
        self.LGB_val_datasets_size.setObjectName("val_datasets_size")

        self.LGB_Min_label = QtWidgets.QLabel(self.LGB_train_group)
        self.LGB_Min_label.setGeometry(QtCore.QRect(188, 55, 80, 20))
        self.LGB_Min_label.setObjectName("LGB_Min_label")
        self.LGB_Max_label = QtWidgets.QLabel(self.LGB_train_group)
        self.LGB_Max_label.setGeometry(QtCore.QRect(278, 55, 80, 20))
        self.LGB_Max_label.setObjectName("LGB_Max_label")
        self.LGB_Interval_label = QtWidgets.QLabel(self.LGB_train_group)
        self.LGB_Interval_label.setGeometry(QtCore.QRect(355, 55, 80, 20))
        self.LGB_Interval_label.setObjectName("LGB_Interval_label")

        self.LGB_n_estimators_min = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_n_estimators_min.setGeometry(QtCore.QRect(160, 75, 80, 20))
        self.LGB_n_estimators_min.setObjectName("LGB_n_estimators_min")
        self.LGB_n_estimators_interval = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_n_estimators_interval.setGeometry(QtCore.QRect(340, 75, 80, 20))
        self.LGB_n_estimators_interval.setObjectName("LGB_n_estimators_interval")
        self.LGB_n_estimators_max = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_n_estimators_max.setGeometry(QtCore.QRect(250, 75, 80, 20))
        self.LGB_n_estimators_max.setObjectName("LGB_n_estimators_max")
        self.LGB_n_estimators_label = QtWidgets.QLabel(self.LGB_train_group)
        self.LGB_n_estimators_label.setGeometry(QtCore.QRect(10, 75, 130, 20))
        self.LGB_n_estimators_label.setObjectName("LGB_n_estimators_label")

        self.LGB_max_depth_min = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_max_depth_min.setGeometry(QtCore.QRect(160, 105, 80, 20))
        self.LGB_max_depth_min.setObjectName("max_depth_min")
        self.LGB_max_depth_max = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_max_depth_max.setGeometry(QtCore.QRect(250, 105, 80, 20))
        self.LGB_max_depth_max.setObjectName("max_depth_max")
        self.LGB_max_depth_interval = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_max_depth_interval.setGeometry(QtCore.QRect(340, 105, 80, 20))
        self.LGB_max_depth_interval.setObjectName("max_depth_interval")
        self.LGB_max_depth_label = QtWidgets.QLabel(self.LGB_train_group)
        self.LGB_max_depth_label.setGeometry(QtCore.QRect(10, 105, 140, 20))
        self.LGB_max_depth_label.setObjectName("max_depth_label")

        self.LGB_learning_rate_min = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_learning_rate_min.setGeometry(QtCore.QRect(160, 135, 80, 20))
        self.LGB_learning_rate_min.setObjectName("learning_rate_min")
        self.LGB_learning_rate_Max = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_learning_rate_Max.setGeometry(QtCore.QRect(250, 135, 80, 20))
        self.LGB_learning_rate_Max.setObjectName("learning_rate_Max")
        self.LGB_learning_rate_Interval = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_learning_rate_Interval.setGeometry(QtCore.QRect(340, 135, 80, 20))
        self.LGB_learning_rate_Interval.setObjectName("learning_rate_Interval")
        self.LGB_learning_rate_label = QtWidgets.QLabel(self.LGB_train_group)
        self.LGB_learning_rate_label.setGeometry(QtCore.QRect(10, 135, 140, 20))
        self.LGB_learning_rate_label.setObjectName("learning_rate_label")

        self.LGB_subsample_min = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_subsample_min.setGeometry(QtCore.QRect(160, 165, 80, 20))
        self.LGB_subsample_min.setObjectName("subsample_min")
        self.LGB_subsample_max = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_subsample_max.setGeometry(QtCore.QRect(250, 165, 80, 20))
        self.LGB_subsample_max.setObjectName("subsample_max")
        self.LGB_subsample_interval = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_subsample_interval.setGeometry(QtCore.QRect(340, 165, 80, 20))
        self.LGB_subsample_interval.setObjectName("subsample_interval")
        self.LGB_subsample_label = QtWidgets.QLabel(self.LGB_train_group)
        self.LGB_subsample_label.setGeometry(QtCore.QRect(10, 165, 140, 20))
        self.LGB_subsample_label.setObjectName("LGB_subsample_label")

        self.LGB_colsample_bytree_min = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_colsample_bytree_min.setGeometry(QtCore.QRect(160, 195, 80, 20))
        self.LGB_colsample_bytree_min.setObjectName("LGB_colsample_bytree_min")
        self.LGB_colsample_bytree_max = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_colsample_bytree_max.setGeometry(QtCore.QRect(250, 195, 80, 20))
        self.LGB_colsample_bytree_max.setObjectName("LGB_colsample_bytree_max")
        self.LGB_colsample_bytree_interval = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_colsample_bytree_interval.setGeometry(QtCore.QRect(340, 195, 80, 20))
        self.LGB_colsample_bytree_interval.setObjectName("LGB_colsample_bytree_interval")
        self.LGB_colsample_bytree_label = QtWidgets.QLabel(self.LGB_train_group)
        self.LGB_colsample_bytree_label.setGeometry(QtCore.QRect(10, 195, 140, 20))
        self.LGB_colsample_bytree_label.setObjectName("LGB_colsample_bytree_label")

        self.LGB_reg_alpha_min = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_reg_alpha_min.setGeometry(QtCore.QRect(160, 225, 80, 20))
        self.LGB_reg_alpha_min.setObjectName("LGB_reg_alpha_min")
        self.LGB_reg_alpha_max = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_reg_alpha_max.setGeometry(QtCore.QRect(250, 225, 80, 20))
        self.LGB_reg_alpha_max.setObjectName("LGB_reg_alpha_max")
        self.LGB_reg_alpha_interval = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_reg_alpha_interval.setGeometry(QtCore.QRect(340, 225, 80, 20))
        self.LGB_reg_alpha_interval.setObjectName("LGB_reg_alpha_interval")
        self.LGB_reg_alpha_label = QtWidgets.QLabel(self.LGB_train_group)
        self.LGB_reg_alpha_label.setGeometry(QtCore.QRect(10, 225, 140, 20))
        self.LGB_reg_alpha_label.setObjectName("LGB_reg_alpha_label")

        self.LGB_reg_lambda_min = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_reg_lambda_min.setGeometry(QtCore.QRect(160, 255, 80, 20))
        self.LGB_reg_lambda_min.setObjectName("LGB_reg_lambda_min")
        self.LGB_reg_lambda_max = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_reg_lambda_max.setGeometry(QtCore.QRect(250, 255, 80, 20))
        self.LGB_reg_lambda_max.setObjectName("LGB_reg_lambda_max")
        self.LGB_reg_lambda_interval = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_reg_lambda_interval.setGeometry(QtCore.QRect(340, 255, 80, 20))
        self.LGB_reg_lambda_interval.setObjectName("LGB_reg_lambda_interval")
        self.LGB_reg_lambda_label = QtWidgets.QLabel(self.LGB_train_group)
        self.LGB_reg_lambda_label.setGeometry(QtCore.QRect(10, 255, 140, 20))
        self.LGB_reg_lambda_label.setObjectName("LGB_reg_lambda_label")

        self.LGB_min_child_weight_min = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_min_child_weight_min.setGeometry(QtCore.QRect(160, 285, 80, 20))
        self.LGB_min_child_weight_min.setObjectName("LGB_min_child_weight_min")
        self.LGB_min_child_weight_max = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_min_child_weight_max.setGeometry(QtCore.QRect(250, 285, 80, 20))
        self.LGB_min_child_weight_max.setObjectName("LGB_min_child_weight_max")
        self.LGB_min_child_weight_interval = QtWidgets.QLineEdit(self.LGB_train_group)
        self.LGB_min_child_weight_interval.setGeometry(QtCore.QRect(340, 285, 80, 20))
        self.LGB_min_child_weight_interval.setObjectName("LGB_min_child_weight_interval")
        self.LGB_min_child_weight_label = QtWidgets.QLabel(self.LGB_train_group)
        self.LGB_min_child_weight_label.setGeometry(QtCore.QRect(10, 285, 140, 20))
        self.LGB_min_child_weight_label.setObjectName("LGB_min_child_weight_label")

        # ======================================== LGB_predict_group ========================================
        self.LGB_predict_group = QtWidgets.QGroupBox(self.LGB_tab_page)
        self.LGB_predict_group.setGeometry(QtCore.QRect(10, 640, 460, 220))
        self.LGB_predict_group.setObjectName("LGB_predict_group")
        # 模型路径
        self.LGB_model_path_label = QtWidgets.QLabel(self.LGB_predict_group)
        self.LGB_model_path_label.setGeometry(QtCore.QRect(10, 20, 80, 30))
        self.LGB_model_path_label.setObjectName("LGB_model_path_label")
        self.LGB_model_path = QtWidgets.QLineEdit(self.LGB_predict_group)
        self.LGB_model_path.setGeometry(QtCore.QRect(90, 24, 235, 22))
        self.LGB_model_path.setObjectName("LGB_model_path")
        self.LGB_model_path.setReadOnly(True)
        self.select_LGB_model_path_button = QtWidgets.QPushButton(self.LGB_predict_group)
        self.select_LGB_model_path_button.setGeometry(QtCore.QRect(350, 20, 100, 30))
        self.select_LGB_model_path_button.setObjectName("select_LGB_model_path_button")
        self.select_LGB_model_path_button.clicked.connect(self.select_LGB_model_path_pushbutton_clicked)
        # 输出路径
        self.LGB_predict_output_result_path_label = QtWidgets.QLabel(self.LGB_predict_group)
        self.LGB_predict_output_result_path_label.setGeometry(QtCore.QRect(10, 140, 80, 30))
        self.LGB_predict_output_result_path_label.setObjectName("LGB_predict_output_result_path_label")
        self.LGB_output_predict_result_path = QtWidgets.QLineEdit(self.LGB_predict_group)
        self.LGB_output_predict_result_path.setGeometry(QtCore.QRect(90, 144, 235, 22))
        self.LGB_output_predict_result_path.setObjectName("LGB_output_predict_result_path")
        self.LGB_output_predict_result_path.setReadOnly(True)
        self.LGB_select_predict_result_output_path_button = QtWidgets.QPushButton(self.LGB_predict_group)
        self.LGB_select_predict_result_output_path_button.setGeometry(QtCore.QRect(350, 140, 100, 30))
        self.LGB_select_predict_result_output_path_button.setObjectName(
            "LGB_select_predict_result_output_path_button"
        )
        self.LGB_select_predict_result_output_path_button.clicked.connect(
            self.LGB_select_predict_result_output_path_pushbutton_clicked
        )
        # 影像路径
        self.LGB_img_predict_path_label = QtWidgets.QLabel(self.LGB_predict_group)
        self.LGB_img_predict_path_label.setGeometry(QtCore.QRect(10, 60, 80, 30))
        self.LGB_img_predict_path_label.setObjectName("LGB_img_predict_path_label")
        self.LGB_img_predict_path = QtWidgets.QLineEdit(self.LGB_predict_group)
        self.LGB_img_predict_path.setGeometry(QtCore.QRect(90, 64, 235, 22))
        self.LGB_img_predict_path.setObjectName("LGB_img_predict_path")
        self.LGB_img_predict_path.setReadOnly(True)
        self.select_LGB_img_predict_path_button = QtWidgets.QPushButton(self.LGB_predict_group)
        self.select_LGB_img_predict_path_button.setGeometry(QtCore.QRect(350, 60, 100, 30))
        self.select_LGB_img_predict_path_button.setObjectName("select_LGB_img_predict_path_button")
        self.select_LGB_img_predict_path_button.clicked.connect(
            self.select_LGB_img_predict_path_button_clicked
        )
        # 类别路径
        self.LGB_class_num_label_2 = QtWidgets.QLabel(self.LGB_predict_group)
        self.LGB_class_num_label_2.setGeometry(QtCore.QRect(10, 100, 80, 30))
        self.LGB_class_num_label_2.setObjectName("LGB_class_num_label_2")
        self.LGB_class_num_path_2 = QtWidgets.QLineEdit(self.LGB_predict_group)
        self.LGB_class_num_path_2.setGeometry(QtCore.QRect(90, 104, 235, 22))
        self.LGB_class_num_path_2.setObjectName("LGB_class_num_path_2")
        self.LGB_class_num_path_2.setReadOnly(True)
        self.select_LGB_class_num_button_2 = QtWidgets.QPushButton(self.LGB_predict_group)
        self.select_LGB_class_num_button_2.setGeometry(QtCore.QRect(350, 100, 100, 30))
        self.select_LGB_class_num_button_2.setObjectName("select_LGB_class_num_button_2")
        self.select_LGB_class_num_button_2.clicked.connect(self.select_LGB_class_num_button_2_clicked)

        self.LGB_predict_button = QtWidgets.QPushButton(self.LGB_predict_group)
        self.LGB_predict_button.setGeometry(QtCore.QRect(160, 180, 100, 30))
        self.LGB_predict_button.setObjectName("LGB_predict_button")
        self.LGB_predict_button.clicked.connect(self.LGB_predict_pushButton_clicked)

    def XGB_tab_page_init(self):
        # ======================================== XGB_get_samples_group ========================================
        self.XGB_get_samples_group = QtWidgets.QGroupBox(self.XGB_tab_page)
        self.XGB_get_samples_group.setGeometry(QtCore.QRect(10, 10, 460, 215))
        self.XGB_get_samples_group.setObjectName("XGB_get_samples_group")
        self.XGB_img_path_label = QtWidgets.QLabel(self.XGB_get_samples_group)
        self.XGB_img_path_label.setGeometry(QtCore.QRect(10, 20, 460, 30))
        self.XGB_img_path_label.setObjectName("XGB_img_path_label")
        self.XGB_img_path = QtWidgets.QLineEdit(self.XGB_get_samples_group)
        self.XGB_img_path.setGeometry(QtCore.QRect(90, 24, 235, 22))
        self.XGB_img_path.setObjectName("XGB_img_path")
        self.XGB_img_path.setReadOnly(True)
        self.select_XGB_img_button = QtWidgets.QPushButton(self.XGB_get_samples_group)
        self.select_XGB_img_button.setGeometry(QtCore.QRect(350, 20, 100, 30))
        self.select_XGB_img_button.setObjectName("select_XGB_img_button")
        self.select_XGB_img_button.clicked.connect(self.XGB_select_image_pushbutton_clicked)

        self.XGB_label_path_label = QtWidgets.QLabel(self.XGB_get_samples_group)
        self.XGB_label_path_label.setGeometry(QtCore.QRect(10, 60, 80, 30))
        self.XGB_label_path_label.setObjectName("XGB_label_path_label")
        self.select_XGB_lable_button = QtWidgets.QPushButton(self.XGB_get_samples_group)
        self.select_XGB_lable_button.setGeometry(QtCore.QRect(350, 60, 100, 30))
        self.select_XGB_lable_button.setObjectName("XGB_label_path_label")
        self.select_XGB_lable_button.clicked.connect(self.XGB_select_label_pushbutton_clicked)
        self.XGB_label_path = QtWidgets.QLineEdit(self.XGB_get_samples_group)
        self.XGB_label_path.setGeometry(QtCore.QRect(90, 64, 235, 22))
        self.XGB_label_path.setObjectName("XGB_label_path")
        self.XGB_label_path.setReadOnly(True)

        self.XGB_class_num_label = QtWidgets.QLabel(self.XGB_get_samples_group)
        self.XGB_class_num_label.setGeometry(QtCore.QRect(10, 100, 80, 30))
        self.XGB_class_num_label.setObjectName("XGB_class_num_label")
        self.XGB_class_num_path = QtWidgets.QLineEdit(self.XGB_get_samples_group)
        self.XGB_class_num_path.setGeometry(QtCore.QRect(90, 104, 235, 22))
        self.XGB_class_num_path.setObjectName("XGB_class_num_path")
        self.XGB_class_num_path.setReadOnly(True)
        self.select_XGB_class_num_button = QtWidgets.QPushButton(self.XGB_get_samples_group)
        self.select_XGB_class_num_button.setGeometry(QtCore.QRect(350, 100, 100, 30))
        self.select_XGB_class_num_button.setObjectName("select_XGB_class_num_button")
        self.select_XGB_class_num_button.clicked.connect(self.XGB_select_numclass_path_pushbutton_clicked)
        self.XGB_output_samples_path_label = QtWidgets.QLabel(self.XGB_get_samples_group)
        self.XGB_output_samples_path_label.setGeometry(QtCore.QRect(10, 140, 80, 30))
        self.XGB_output_samples_path_label.setObjectName("XGB_output_samples_path_label")

        self.XGB_output_samples_path = QtWidgets.QLineEdit(self.XGB_get_samples_group)
        self.XGB_output_samples_path.setGeometry(QtCore.QRect(90, 144, 235, 22))
        self.XGB_output_samples_path.setObjectName("XGB_output_samples_path")
        self.XGB_output_samples_path.setReadOnly(True)
        self.select_XGB_output_samples_path_button = QtWidgets.QPushButton(self.XGB_get_samples_group)
        self.select_XGB_output_samples_path_button.setGeometry(QtCore.QRect(350, 140, 100, 30))
        self.select_XGB_output_samples_path_button.setObjectName("select_XGB_output_samples_path_button")
        self.select_XGB_output_samples_path_button.clicked.connect(
            self.XGB_select_sample_path_pushbutton_clicked
        )
        self.XGB_get_samples_button = QtWidgets.QPushButton(self.XGB_get_samples_group)
        self.XGB_get_samples_button.setGeometry(QtCore.QRect(160, 175, 100, 30))
        self.XGB_get_samples_button.setObjectName("get_XGB_samples_button")
        self.XGB_get_samples_button.clicked.connect(self.XGB_get_sample_pushButton_clicked)

        # ======================================== XGB_train_group ========================================

        self.XGB_train_group = QtWidgets.QGroupBox(self.XGB_tab_page)
        self.XGB_train_group.setGeometry(QtCore.QRect(10, 230, 460, 401))
        self.XGB_train_group.setObjectName("XGB_train_group")

        # model save path
        self.XGB_train_model_save_path_label = QtWidgets.QLabel(self.XGB_train_group)
        self.XGB_train_model_save_path_label.setGeometry(QtCore.QRect(10, 336, 90, 25))
        self.XGB_train_model_save_path_label.setObjectName("XGB_train_model_save_path_label")
        self.XGB_train_model_save_path = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_train_model_save_path.setGeometry(QtCore.QRect(110, 339, 230, 22))
        self.XGB_train_model_save_path.setObjectName("XGB_train_model_save_path")
        self.XGB_train_model_save_path.setReadOnly(True)
        self.select_XGB_train_model_save_path_button = QtWidgets.QPushButton(self.XGB_train_group)
        self.select_XGB_train_model_save_path_button.setGeometry(QtCore.QRect(350, 335, 100, 30))
        self.select_XGB_train_model_save_path_button.setObjectName("select_XGB_train_model_save_path_button")
        self.select_XGB_train_model_save_path_button.clicked.connect(
            self.select_XGB_train_Save_model_path_pushButton_clicked
        )
        self.XGB_train_button = QtWidgets.QPushButton(self.XGB_train_group)
        self.XGB_train_button.setGeometry(QtCore.QRect(160, 366, 100, 30))
        self.XGB_train_button.setObjectName("XGB_train_button")
        self.XGB_train_button.clicked.connect(self.XGB_train_pushButton_clicked)

        # 训练的参数设置
        # 训练参数比例
        self.XGB_val_datasets_size_label = QtWidgets.QLabel(self.XGB_train_group)
        self.XGB_val_datasets_size_label.setGeometry(QtCore.QRect(10, 20, 100, 20))
        self.XGB_val_datasets_size_label.setObjectName("XGB_val_datasets_size_label")
        self.XGB_val_datasets_size = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_val_datasets_size.setGeometry(QtCore.QRect(130, 20, 100, 20))
        self.XGB_val_datasets_size.setObjectName("val_datasets_size")

        self.XGB_Min_label = QtWidgets.QLabel(self.XGB_train_group)
        self.XGB_Min_label.setGeometry(QtCore.QRect(188, 45, 80, 20))
        self.XGB_Min_label.setObjectName("XGB_Min_label")
        self.XGB_Max_label = QtWidgets.QLabel(self.XGB_train_group)
        self.XGB_Max_label.setGeometry(QtCore.QRect(278, 45, 80, 20))
        self.XGB_Max_label.setObjectName("XGB_Max_label")
        self.XGB_Interval_label = QtWidgets.QLabel(self.XGB_train_group)
        self.XGB_Interval_label.setGeometry(QtCore.QRect(355, 45, 80, 20))
        self.XGB_Interval_label.setObjectName("XGB_Interval_label")

        self.XGB_n_estimators_min = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_n_estimators_min.setGeometry(QtCore.QRect(160, 65, 80, 20))
        self.XGB_n_estimators_min.setObjectName("XGB_n_estimators_min")
        self.XGB_n_estimators_interval = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_n_estimators_interval.setGeometry(QtCore.QRect(340, 65, 80, 20))
        self.XGB_n_estimators_interval.setObjectName("XGB_n_estimators_interval")
        self.XGB_n_estimators_max = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_n_estimators_max.setGeometry(QtCore.QRect(250, 65, 80, 20))
        self.XGB_n_estimators_max.setObjectName("XGB_n_estimators_max")
        self.XGB_n_estimators_label = QtWidgets.QLabel(self.XGB_train_group)
        self.XGB_n_estimators_label.setGeometry(QtCore.QRect(10, 65, 130, 20))
        self.XGB_n_estimators_label.setObjectName("XGB_n_estimators_label")

        self.XGB_max_depth_min = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_max_depth_min.setGeometry(QtCore.QRect(160, 95, 80, 20))
        self.XGB_max_depth_min.setObjectName("max_depth_min")
        self.XGB_max_depth_max = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_max_depth_max.setGeometry(QtCore.QRect(250, 95, 80, 20))
        self.XGB_max_depth_max.setObjectName("max_depth_max")
        self.XGB_max_depth_interval = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_max_depth_interval.setGeometry(QtCore.QRect(340, 95, 80, 20))
        self.XGB_max_depth_interval.setObjectName("max_depth_interval")
        self.XGB_max_depth_label = QtWidgets.QLabel(self.XGB_train_group)
        self.XGB_max_depth_label.setGeometry(QtCore.QRect(10, 95, 140, 20))
        self.XGB_max_depth_label.setObjectName("max_depth_label")

        self.XGB_learning_rate_min = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_learning_rate_min.setGeometry(QtCore.QRect(160, 125, 80, 20))
        self.XGB_learning_rate_min.setObjectName("learning_rate_min")
        self.XGB_learning_rate_Max = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_learning_rate_Max.setGeometry(QtCore.QRect(250, 125, 80, 20))
        self.XGB_learning_rate_Max.setObjectName("learning_rate_Max")
        self.XGB_learning_rate_Interval = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_learning_rate_Interval.setGeometry(QtCore.QRect(340, 125, 80, 20))
        self.XGB_learning_rate_Interval.setObjectName("learning_rate_Interval")
        self.XGB_learning_rate_label = QtWidgets.QLabel(self.XGB_train_group)
        self.XGB_learning_rate_label.setGeometry(QtCore.QRect(10, 125, 140, 20))
        self.XGB_learning_rate_label.setObjectName("learning_rate_label")

        self.XGB_subsample_min = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_subsample_min.setGeometry(QtCore.QRect(160, 155, 80, 20))
        self.XGB_subsample_min.setObjectName("subsample_min")
        self.XGB_subsample_max = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_subsample_max.setGeometry(QtCore.QRect(250, 155, 80, 20))
        self.XGB_subsample_max.setObjectName("subsample_max")
        self.XGB_subsample_interval = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_subsample_interval.setGeometry(QtCore.QRect(340, 155, 80, 20))
        self.XGB_subsample_interval.setObjectName("subsample_interval")
        self.XGB_subsample_label = QtWidgets.QLabel(self.XGB_train_group)
        self.XGB_subsample_label.setGeometry(QtCore.QRect(10, 155, 140, 20))
        self.XGB_subsample_label.setObjectName("XGB_subsample_label")

        self.XGB_colsample_bytree_min = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_colsample_bytree_min.setGeometry(QtCore.QRect(160, 185, 80, 20))
        self.XGB_colsample_bytree_min.setObjectName("XGB_colsample_bytree_min")
        self.XGB_colsample_bytree_max = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_colsample_bytree_max.setGeometry(QtCore.QRect(250, 185, 80, 20))
        self.XGB_colsample_bytree_max.setObjectName("XGB_colsample_bytree_max")
        self.XGB_colsample_bytree_interval = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_colsample_bytree_interval.setGeometry(QtCore.QRect(340, 185, 80, 20))
        self.XGB_colsample_bytree_interval.setObjectName("XGB_colsample_bytree_interval")
        self.XGB_colsample_bytree_label = QtWidgets.QLabel(self.XGB_train_group)
        self.XGB_colsample_bytree_label.setGeometry(QtCore.QRect(10, 185, 140, 20))
        self.XGB_colsample_bytree_label.setObjectName("XGB_colsample_bytree_label")

        self.XGB_gamma_min = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_gamma_min.setGeometry(QtCore.QRect(160, 215, 80, 20))
        self.XGB_gamma_min.setObjectName("XGB_gamma_min")
        self.XGB_gamma_max = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_gamma_max.setGeometry(QtCore.QRect(250, 215, 80, 20))
        self.XGB_gamma_max.setObjectName("XGB_gamma_max")
        self.XGB_gamma_interval = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_gamma_interval.setGeometry(QtCore.QRect(340, 215, 80, 20))
        self.XGB_gamma_interval.setObjectName("XGB_gamma_interval")
        self.XGB_gamma_label = QtWidgets.QLabel(self.XGB_train_group)
        self.XGB_gamma_label.setGeometry(QtCore.QRect(10, 215, 140, 20))
        self.XGB_gamma_label.setObjectName("XGB_gamma_label")

        self.XGB_reg_alpha_min = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_reg_alpha_min.setGeometry(QtCore.QRect(160, 245, 80, 20))
        self.XGB_reg_alpha_min.setObjectName("XGB_reg_alpha_min")
        self.XGB_reg_alpha_max = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_reg_alpha_max.setGeometry(QtCore.QRect(250, 245, 80, 20))
        self.XGB_reg_alpha_max.setObjectName("XGB_reg_alpha_max")
        self.XGB_reg_alpha_interval = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_reg_alpha_interval.setGeometry(QtCore.QRect(340, 245, 80, 20))
        self.XGB_reg_alpha_interval.setObjectName("XGB_reg_alpha_interval")
        self.XGB_reg_alpha_label = QtWidgets.QLabel(self.XGB_train_group)
        self.XGB_reg_alpha_label.setGeometry(QtCore.QRect(10, 245, 140, 20))
        self.XGB_reg_alpha_label.setObjectName("XGB_reg_alpha_label")

        self.XGB_reg_lambda_min = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_reg_lambda_min.setGeometry(QtCore.QRect(160, 275, 80, 20))
        self.XGB_reg_lambda_min.setObjectName("XGB_reg_lambda_min")
        self.XGB_reg_lambda_max = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_reg_lambda_max.setGeometry(QtCore.QRect(250, 275, 80, 20))
        self.XGB_reg_lambda_max.setObjectName("XGB_reg_lambda_max")
        self.XGB_reg_lambda_interval = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_reg_lambda_interval.setGeometry(QtCore.QRect(340, 275, 80, 20))
        self.XGB_reg_lambda_interval.setObjectName("XGB_reg_lambda_interval")
        self.XGB_reg_lambda_label = QtWidgets.QLabel(self.XGB_train_group)
        self.XGB_reg_lambda_label.setGeometry(QtCore.QRect(10, 275, 140, 20))
        self.XGB_reg_lambda_label.setObjectName("XGB_reg_lambda_label")

        self.XGB_min_child_weight_min = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_min_child_weight_min.setGeometry(QtCore.QRect(160, 305, 80, 20))
        self.XGB_min_child_weight_min.setObjectName("XGB_min_child_weight_min")
        self.XGB_min_child_weight_max = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_min_child_weight_max.setGeometry(QtCore.QRect(250, 305, 80, 20))
        self.XGB_min_child_weight_max.setObjectName("XGB_min_child_weight_max")
        self.XGB_min_child_weight_interval = QtWidgets.QLineEdit(self.XGB_train_group)
        self.XGB_min_child_weight_interval.setGeometry(QtCore.QRect(340, 305, 80, 20))
        self.XGB_min_child_weight_interval.setObjectName("XGB_min_child_weight_interval")
        self.XGB_min_child_weight_label = QtWidgets.QLabel(self.XGB_train_group)
        self.XGB_min_child_weight_label.setGeometry(QtCore.QRect(10, 305, 140, 20))
        self.XGB_min_child_weight_label.setObjectName("XGB_min_child_weight_label")

        # ======================================== XGB_predict_group ========================================
        self.XGB_predict_group = QtWidgets.QGroupBox(self.XGB_tab_page)
        self.XGB_predict_group.setGeometry(QtCore.QRect(10, 640, 460, 220))
        self.XGB_predict_group.setObjectName("XGB_predict_group")

        self.XGB_model_path_label = QtWidgets.QLabel(self.XGB_predict_group)
        self.XGB_model_path_label.setGeometry(QtCore.QRect(10, 20, 80, 30))
        self.XGB_model_path_label.setObjectName("XGB_model_path_label")
        self.XGB_model_path = QtWidgets.QLineEdit(self.XGB_predict_group)
        self.XGB_model_path.setGeometry(QtCore.QRect(90, 24, 235, 22))
        self.XGB_model_path.setObjectName("XGB_model_path")
        self.XGB_model_path.setReadOnly(True)
        self.select_XGB_model_path_button = QtWidgets.QPushButton(self.XGB_predict_group)
        self.select_XGB_model_path_button.setGeometry(QtCore.QRect(350, 20, 100, 30))
        self.select_XGB_model_path_button.setObjectName("select_XGB_model_path_button")
        self.select_XGB_model_path_button.clicked.connect(self.select_XGB_model_path_pushButton_clicked)

        self.XGB_predict_output_result_path_label = QtWidgets.QLabel(self.XGB_predict_group)
        self.XGB_predict_output_result_path_label.setGeometry(QtCore.QRect(10, 140, 80, 30))
        self.XGB_predict_output_result_path_label.setObjectName("XGB_predict_output_result_path_label")
        self.XGB_output_predict_result_path = QtWidgets.QLineEdit(self.XGB_predict_group)
        self.XGB_output_predict_result_path.setGeometry(QtCore.QRect(90, 144, 235, 22))
        self.XGB_output_predict_result_path.setObjectName("XGB_output_predict_result_path")
        self.XGB_output_predict_result_path.setReadOnly(True)
        self.XGB_select_predict_result_output_path_button = QtWidgets.QPushButton(self.XGB_predict_group)
        self.XGB_select_predict_result_output_path_button.setGeometry(QtCore.QRect(350, 140, 100, 30))
        self.XGB_select_predict_result_output_path_button.setObjectName(
            "XGB_select_predict_result_output_path_button"
        )
        self.XGB_select_predict_result_output_path_button.clicked.connect(
            self.XGB_select_predict_result_output_path_pushButton_clicked
        )

        self.XGB_img_predict_path_label = QtWidgets.QLabel(self.XGB_predict_group)
        self.XGB_img_predict_path_label.setGeometry(QtCore.QRect(10, 60, 80, 30))
        self.XGB_img_predict_path_label.setObjectName("XGB_img_predict_path_label")
        self.XGB_img_predict_path = QtWidgets.QLineEdit(self.XGB_predict_group)
        self.XGB_img_predict_path.setGeometry(QtCore.QRect(90, 64, 235, 22))
        self.XGB_img_predict_path.setObjectName("XGB_img_predict_path")
        self.XGB_img_predict_path.setReadOnly(True)
        self.select_XGB_img_predict_path_button = QtWidgets.QPushButton(self.XGB_predict_group)
        self.select_XGB_img_predict_path_button.setGeometry(QtCore.QRect(350, 60, 100, 30))
        self.select_XGB_img_predict_path_button.setObjectName("select_XGB_img_predict_path_button")
        self.select_XGB_img_predict_path_button.clicked.connect(
            self.select_XGB_img_predict_path_pushButton_clicked
        )

        self.XGB_class_num_label_2 = QtWidgets.QLabel(self.XGB_predict_group)
        self.XGB_class_num_label_2.setGeometry(QtCore.QRect(10, 100, 80, 30))
        self.XGB_class_num_label_2.setObjectName("XGB_class_num_label_2")
        self.XGB_class_num_path_2 = QtWidgets.QLineEdit(self.XGB_predict_group)
        self.XGB_class_num_path_2.setGeometry(QtCore.QRect(90, 104, 235, 22))
        self.XGB_class_num_path_2.setObjectName("XGB_class_num_path_2")
        self.XGB_class_num_path_2.setReadOnly(True)
        self.select_XGB_class_num_button_2 = QtWidgets.QPushButton(self.XGB_predict_group)
        self.select_XGB_class_num_button_2.setGeometry(QtCore.QRect(350, 100, 100, 30))
        self.select_XGB_class_num_button_2.setObjectName("select_XGB_class_num_button_2")
        self.select_XGB_class_num_button_2.clicked.connect(self.select_XGB_class_num_button_2_clicked)

        self.XGB_predict_button = QtWidgets.QPushButton(self.XGB_predict_group)
        self.XGB_predict_button.setGeometry(QtCore.QRect(160, 180, 100, 30))
        self.XGB_predict_button.setObjectName("XGB_predict_button")
        self.XGB_predict_button.clicked.connect(self.XGB_predict_pushButton_clicked)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "遥感影像机器学习地物分类工具箱"))

        # RF_tab_page
        self.tabWidgetPages.setTabText(
            self.tabWidgetPages.indexOf(self.RF_tab_page), _translate("MainWindow", "随机森林算法")
        )
        self.RF_get_samples_group.setTitle(_translate("MainWindow", "获取样本"))
        self.RF_img_path_label.setText(_translate("MainWindow", "影像路径:"))
        self.select_RF_img_button.setText(_translate("MainWindow", "选择路径"))
        self.RF_label_path_label.setText(_translate("MainWindow", "标签路径:"))
        self.select_RF_label_button.setText(_translate("MainWindow", "选择路径"))
        self.RF_class_num_label.setText(_translate("MainWindow", "类别数目:"))
        self.select_RF_class_num_button.setText(_translate("MainWindow", "选择路径"))
        self.RF_output_samples_path_label.setText(_translate("MainWindow", "样本路径:"))
        self.select_RF_output_samples_path_button.setText(_translate("MainWindow", "选择路径"))
        self.RF_get_samples_button.setText(_translate("MainWindow", "获取样本"))
        self.RF_train_group.setTitle(_translate("MainWindow", "训练寻优参数"))
        self.RF_train_button.setText(_translate("MainWindow", "训练"))
        self.RF_val_datasets_size_label.setText(_translate("MainWindow", "验证集比例:"))
        self.RF_val_datasets_size.setText(_translate("MainWindow", "0.3"))
        self.RF_n_estimators_label.setText(_translate("MainWindow", "n_estimators:"))
        self.RF_n_estimators_min.setText(_translate("MainWindow", "50"))
        self.RF_n_estimators_interval.setText(_translate("MainWindow", "1"))
        self.RF_n_estimators_max.setText(_translate("MainWindow", "150"))
        self.RF_Min_label.setText(_translate("MainWindow", "Min"))
        self.RF_Max_label.setText(_translate("MainWindow", "Max"))
        self.RF_Interval_label.setText(_translate("MainWindow", "Interval"))
        self.RF_max_depth_min.setText(_translate("MainWindow", "5"))
        self.RF_max_depth_max.setText(_translate("MainWindow", "20"))
        self.RF_max_depth_interval.setText(_translate("MainWindow", "1"))
        self.RF_max_depth_label.setText(_translate("MainWindow", "max_depth:"))
        self.RF_min_samples_leaf_min.setText(_translate("MainWindow", "1"))
        self.RF_min_samples_leaf_max.setText(_translate("MainWindow", "20"))
        self.RF_min_samples_leaf_interval.setText(_translate("MainWindow", "1"))
        self.RF_min_samples_leaf_label.setText(_translate("MainWindow", "min_samples_leaf:"))
        self.RF_min_samples_split_min.setText(_translate("MainWindow", "2"))
        self.RF_min_samples_split_max.setText(_translate("MainWindow", "10"))
        self.RF_min_samples_split_interval.setText(_translate("MainWindow", "1"))
        self.RF_min_samples_split_label.setText(_translate("MainWindow", "min_samples_split:"))
        self.RF_max_features_label.setText(_translate("MainWindow", "max_features:"))
        self.sqrt_cb.setText(_translate("MainWindow", "sqrt"))
        self.log2_cb.setText(_translate("MainWindow", "log2"))
        self.None_cb.setText(_translate("MainWindow", "None"))
        self.True_cb.setText(_translate("MainWindow", "True"))
        self.bootstrap_label.setText(_translate("MainWindow", "bootstrap:"))
        self.False_cb.setText(_translate("MainWindow", "False"))

        self.RF_train_model_save_path_label.setText(_translate("MainWindow", "模型保存路径:"))
        self.select_RF_train_model_save_path_button.setText(_translate("MainWindow", "选择路径"))

        self.RF_predict_group.setTitle(_translate("MainWindow", "预测参数"))
        self.select_RF_img_predict_path_button.setText(_translate("MainWindow", "选择路径"))
        self.RF_select_predict_result_output_path_button.setText(_translate("MainWindow", "选择路径"))
        self.RF_class_num_label_2.setText(_translate("MainWindow", "类别数目:"))
        self.RF_img_predict_path_label.setText(_translate("MainWindow", "影像路径:"))
        self.select_RF_class_num_button_2.setText(_translate("MainWindow", "选择路径"))
        self.RF_model_path_label.setText(_translate("MainWindow", "模型路径:"))
        self.select_RF_model_path_button.setText(_translate("MainWindow", "选择路径"))
        self.RF_predict_button.setText(_translate("MainWindow", "预测"))
        self.RF_predict_output_result_path_label.setText(_translate("MainWindow", "输出路径:"))

        # XGB_tab_page
        self.tabWidgetPages.setTabText(
            self.tabWidgetPages.indexOf(self.XGB_tab_page), _translate("MainWindow", "XGBoost算法")
        )
        self.XGB_get_samples_group.setTitle(_translate("MainWindow", "获取样本"))
        self.XGB_img_path_label.setText(_translate("MainWindow", "影像路径:"))
        self.select_XGB_img_button.setText(_translate("MainWindow", "选择路径"))
        self.XGB_label_path_label.setText(_translate("MainWindow", "标签路径:"))
        self.select_XGB_lable_button.setText(_translate("MainWindow", "选择路径"))
        self.XGB_class_num_label.setText(_translate("MainWindow", "类别数目:"))
        self.select_XGB_class_num_button.setText(_translate("MainWindow", "选择路径"))
        self.XGB_output_samples_path_label.setText(_translate("MainWindow", "样本路径:"))
        self.select_XGB_output_samples_path_button.setText(_translate("MainWindow", "选择路径"))
        self.XGB_get_samples_button.setText(_translate("MainWindow", "获取样本"))
        self.XGB_train_group.setTitle(_translate("MainWindow", "训练寻优参数"))
        self.XGB_train_button.setText(_translate("MainWindow", "训练"))
        self.XGB_val_datasets_size_label.setText(_translate("MainWindow", "验证集比例:"))
        self.XGB_val_datasets_size.setText(_translate("MainWindow", "0.3"))
        self.XGB_n_estimators_label.setText(_translate("MainWindow", "n_estimators:"))
        self.XGB_n_estimators_min.setText(_translate("MainWindow", "50"))
        self.XGB_n_estimators_max.setText(_translate("MainWindow", "150"))
        self.XGB_n_estimators_interval.setText(_translate("MainWindow", "1"))
        self.XGB_Min_label.setText(_translate("MainWindow", "Min"))
        self.XGB_Max_label.setText(_translate("MainWindow", "Max"))
        self.XGB_Interval_label.setText(_translate("MainWindow", "Interval"))
        self.XGB_max_depth_label.setText(_translate("MainWindow", "max_depth:"))
        self.XGB_max_depth_min.setText(_translate("MainWindow", "5"))
        self.XGB_max_depth_max.setText(_translate("MainWindow", "10"))
        self.XGB_max_depth_interval.setText(_translate("MainWindow", "1"))
        self.XGB_learning_rate_label.setText(_translate("MainWindow", "learning_rate:"))
        self.XGB_learning_rate_min.setText(_translate("MainWindow", "0.01"))
        self.XGB_learning_rate_Max.setText(_translate("MainWindow", "0.1"))
        self.XGB_learning_rate_Interval.setText(_translate("MainWindow", "0.01"))
        self.XGB_subsample_label.setText(_translate("MainWindow", "subsample:"))
        self.XGB_subsample_min.setText(_translate("MainWindow", "0.5"))
        self.XGB_subsample_max.setText(_translate("MainWindow", "1"))
        self.XGB_subsample_interval.setText(_translate("MainWindow", "0.05"))
        self.XGB_colsample_bytree_label.setText(_translate("MainWindow", "colsample_bytree:"))
        self.XGB_colsample_bytree_min.setText(_translate("MainWindow", "0.5"))
        self.XGB_colsample_bytree_max.setText(_translate("MainWindow", "1"))
        self.XGB_colsample_bytree_interval.setText(_translate("MainWindow", "0.05"))
        self.XGB_gamma_label.setText(_translate("MainWindow", "gamma:"))
        self.XGB_gamma_min.setText(_translate("MainWindow", "0"))
        self.XGB_gamma_max.setText(_translate("MainWindow", "10"))
        self.XGB_gamma_interval.setText(_translate("MainWindow", "1"))
        self.XGB_reg_alpha_label.setText(_translate("MainWindow", "reg_alpha:"))
        self.XGB_reg_alpha_min.setText(_translate("MainWindow", "0"))
        self.XGB_reg_alpha_max.setText(_translate("MainWindow", "10"))
        self.XGB_reg_alpha_interval.setText(_translate("MainWindow", "1"))
        self.XGB_reg_lambda_label.setText(_translate("MainWindow", "reg_lambda:"))
        self.XGB_reg_lambda_min.setText(_translate("MainWindow", "0"))
        self.XGB_reg_lambda_max.setText(_translate("MainWindow", "10"))
        self.XGB_reg_lambda_interval.setText(_translate("MainWindow", "1"))
        self.XGB_min_child_weight_label.setText(_translate("MainWindow", "min_child_weight:"))
        self.XGB_min_child_weight_interval.setText(_translate("MainWindow", "1"))
        self.XGB_min_child_weight_min.setText(_translate("MainWindow", "1"))
        self.XGB_min_child_weight_max.setText(_translate("MainWindow", "10"))
        self.XGB_train_model_save_path_label.setText(_translate("MainWindow", "模型保存路径:"))
        self.select_XGB_train_model_save_path_button.setText(_translate("MainWindow", "选择路径"))

        self.XGB_predict_group.setTitle(_translate("MainWindow", "预测参数"))
        self.select_XGB_img_predict_path_button.setText(_translate("MainWindow", "选择路径"))
        self.XGB_select_predict_result_output_path_button.setText(_translate("MainWindow", "选择路径"))
        self.XGB_class_num_label_2.setText(_translate("MainWindow", "类别数目:"))
        self.XGB_img_predict_path_label.setText(_translate("MainWindow", "影像路径:"))
        self.select_XGB_class_num_button_2.setText(_translate("MainWindow", "选择路径"))
        self.XGB_model_path_label.setText(_translate("MainWindow", "模型路径:"))
        self.select_XGB_model_path_button.setText(_translate("MainWindow", "选择路径"))
        self.XGB_predict_button.setText(_translate("MainWindow", "预测"))
        self.XGB_predict_output_result_path_label.setText(_translate("MainWindow", "输出路径:"))

        # LGB_tab_page
        self.tabWidgetPages.setTabText(
            self.tabWidgetPages.indexOf(self.LGB_tab_page), _translate("MainWindow", "LightGBM算法")
        )
        self.LGB_get_samples_group.setTitle(_translate("MainWindow", "获取样本"))
        self.LGB_img_path_label.setText(_translate("MainWindow", "影像路径:"))
        self.select_LGB_img_button.setText(_translate("MainWindow", "选择路径"))
        self.LGB_label_path_label.setText(_translate("MainWindow", "标签路径:"))
        self.select_LGB_lable_button.setText(_translate("MainWindow", "选择路径"))
        self.LGB_class_num_label.setText(_translate("MainWindow", "类别数目:"))
        self.select_LGB_class_num_button.setText(_translate("MainWindow", "选择路径"))
        self.LGB_output_samples_path_label.setText(_translate("MainWindow", "样本路径:"))
        self.select_LGB_output_samples_path_button.setText(_translate("MainWindow", "选择路径"))
        self.LGB_get_samples_button.setText(_translate("MainWindow", "获取样本"))
        self.LGB_train_group.setTitle(_translate("MainWindow", "训练寻优参数"))
        self.LGB_train_button.setText(_translate("MainWindow", "训练"))
        self.LGB_val_datasets_size_label.setText(_translate("MainWindow", "验证集比例:"))
        self.LGB_val_datasets_size.setText(_translate("MainWindow", "0.3"))
        self.LGB_n_estimators_label.setText(_translate("MainWindow", "n_estimators:"))
        self.LGB_n_estimators_min.setText(_translate("MainWindow", "50"))
        self.LGB_n_estimators_max.setText(_translate("MainWindow", "150"))
        self.LGB_n_estimators_interval.setText(_translate("MainWindow", "1"))
        self.LGB_Min_label.setText(_translate("MainWindow", "Min"))
        self.LGB_Max_label.setText(_translate("MainWindow", "Max"))
        self.LGB_Interval_label.setText(_translate("MainWindow", "Interval"))
        self.LGB_max_depth_label.setText(_translate("MainWindow", "max_depth:"))
        self.LGB_max_depth_min.setText(_translate("MainWindow", "5"))
        self.LGB_max_depth_max.setText(_translate("MainWindow", "10"))
        self.LGB_max_depth_interval.setText(_translate("MainWindow", "1"))
        self.LGB_learning_rate_label.setText(_translate("MainWindow", "learning_rate:"))
        self.LGB_learning_rate_min.setText(_translate("MainWindow", "0.01"))
        self.LGB_learning_rate_Max.setText(_translate("MainWindow", "0.1"))
        self.LGB_learning_rate_Interval.setText(_translate("MainWindow", "0.01"))
        self.LGB_subsample_label.setText(_translate("MainWindow", "subsample:"))
        self.LGB_subsample_min.setText(_translate("MainWindow", "0.5"))
        self.LGB_subsample_max.setText(_translate("MainWindow", "1"))
        self.LGB_subsample_interval.setText(_translate("MainWindow", "0.05"))
        self.LGB_colsample_bytree_label.setText(_translate("MainWindow", "colsample_bytree:"))
        self.LGB_colsample_bytree_min.setText(_translate("MainWindow", "0.5"))
        self.LGB_colsample_bytree_max.setText(_translate("MainWindow", "1"))
        self.LGB_colsample_bytree_interval.setText(_translate("MainWindow", "0.05"))
        self.LGB_reg_alpha_label.setText(_translate("MainWindow", "reg_alpha:"))
        self.LGB_reg_alpha_min.setText(_translate("MainWindow", "0"))
        self.LGB_reg_alpha_max.setText(_translate("MainWindow", "10"))
        self.LGB_reg_alpha_interval.setText(_translate("MainWindow", "1"))
        self.LGB_reg_lambda_label.setText(_translate("MainWindow", "reg_lambda:"))
        self.LGB_reg_lambda_min.setText(_translate("MainWindow", "0"))
        self.LGB_reg_lambda_max.setText(_translate("MainWindow", "10"))
        self.LGB_reg_lambda_interval.setText(_translate("MainWindow", "1"))
        self.LGB_min_child_weight_label.setText(_translate("MainWindow", "min_child_weight:"))
        self.LGB_min_child_weight_interval.setText(_translate("MainWindow", "1"))
        self.LGB_min_child_weight_min.setText(_translate("MainWindow", "1"))
        self.LGB_min_child_weight_max.setText(_translate("MainWindow", "10"))
        self.LGB_train_model_save_path_label.setText(_translate("MainWindow", "模型保存路径:"))
        self.select_LGB_train_model_save_path_button.setText(_translate("MainWindow", "选择路径"))

        self.LGB_predict_group.setTitle(_translate("MainWindow", "预测参数"))
        self.select_LGB_img_predict_path_button.setText(_translate("MainWindow", "选择路径"))
        self.LGB_select_predict_result_output_path_button.setText(_translate("MainWindow", "选择路径"))
        self.LGB_class_num_label_2.setText(_translate("MainWindow", "类别数目:"))
        self.LGB_img_predict_path_label.setText(_translate("MainWindow", "影像路径:"))
        self.select_LGB_class_num_button_2.setText(_translate("MainWindow", "选择路径"))
        self.LGB_model_path_label.setText(_translate("MainWindow", "模型路径:"))
        self.select_LGB_model_path_button.setText(_translate("MainWindow", "选择路径"))
        self.LGB_predict_button.setText(_translate("MainWindow", "预测"))
        self.LGB_predict_output_result_path_label.setText(_translate("MainWindow", "输出路径:"))

        self.log_group.setTitle(_translate("MainWindow", "日志"))
        self.SaveLogButton.setText(_translate("MainWindow", "保存日志"))
        # 初始化log_text_browser内容为
        self.log_text_browser.append("*" * 20 + " 欢迎使用遥感影像机器学习地物分类工具箱 " + "*" * 20)
        self.log_text_browser.append("*" * 20 + " 当前页面为:随机森林 " + "*" * 20)

    def tabWidgetPagesChanged(self, index):
        if index == 0:
            self.log_text_browser.append("*" * 20 + " 当前页面为:随机森林 " + "*" * 20)
        elif index == 1:
            self.log_text_browser.append("*" * 20 + " 当前页面为:LightGBM " + "*" * 20)
        elif index == 2:
            self.log_text_browser.append("*" * 20 + " 当前页面为:XGBoost " + "*" * 20)
