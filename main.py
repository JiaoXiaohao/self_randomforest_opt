# -*- coding: utf-8 -*-
# Author: jiaoxiaohao
# E-mail: jiaoxiaohao876@gmail.com
# Time: 2024-02-28 13:40:57
# File name: main.py
# Nothing is true, everything is permitted.
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from __UI__ import ClassificationUI
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QIcon


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = ClassificationUI()
    ui.setupUi(MainWindow)
    # 设置图标
    MainWindow.setWindowIcon(QIcon("icon.png"))
    MainWindow.show()
    sys.exit(app.exec_())
