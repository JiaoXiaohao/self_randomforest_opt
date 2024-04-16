# -*- coding: utf-8 -*-
# File Name：main.py
# description: 主程序入口
import sys
import os
if __name__ == "__main__":
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        from __UI__ import ClassificationUI
    except ImportError:
        # 读取requirements.txt
        with open("requirements.txt", "r") as f:
            requirements = f.readlines()
        # 安装依赖
        for requirement in requirements:
            os.system(
                f"pip install {requirement} -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com"
            )
        # 重新导入
        from PyQt5 import QtCore, QtGui, QtWidgets
        from __UI__ import ClassificationUI
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = ClassificationUI()
    ui.setupUi(MainWindow)
    # 设置图标
    MainWindow.setWindowIcon(QtGui.QIcon("icon.png"))
    MainWindow.show()
    sys.exit(app.exec_())
