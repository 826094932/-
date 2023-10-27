import sys
import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import os

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setEnabled(True)
        Dialog.resize(959, 539)
        Dialog.setMinimumSize(QtCore.QSize(959, 539))
        Dialog.setMaximumSize(QtCore.QSize(19200, 10800))
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(0, 0, 959, 539))
        self.label.setAcceptDrops(False)
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "blob"))
        self.label.setText(_translate("Dialog", "拖拽图片到此处"))


# 读取中文路径
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


class mwindow(QWidget, Ui_Dialog):
    def __init__(self):
        super(mwindow, self).__init__()
        self.setupUi(self)
        # 调用Drops方法
        self.setAcceptDrops(True)
        # 图片
        self.img = None

    # 鼠标拖入事件
    def dragEnterEvent(self, evn):
        # print('鼠标拖入窗口了')
        # 鼠标放开函数事件
        evn.accept()

    # 鼠标放开执行
    def dropEvent(self, evn):
        # 判断文件类型
        if evn.mimeData().hasUrls():
            # 获取文件路径
            file_path = evn.mimeData().urls()[0].toLocalFile()
            print(file_path)
            # 判断是否是图片
            if file_path.endswith('.jpg') or file_path.endswith('.png'):
                self.img = cv_imread(file_path)
            else:
                # 提示
                QMessageBox.information(self, '提示', '请拖入图片')
                return
            # opencv 转qimage
            qimg = QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * self.img.shape[2],
                          QImage.Format_RGB888)
            # 显示图片 自适应
            self.label.setPixmap(
                QPixmap.fromImage(qimg).scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio))
        os_command="python detect.py --weights best_match.pt --source "+file_path
        os.system(os_command)
        # print('鼠标放开了')

    def dragMoveEvent(self, evn):
        pass
        # print('鼠标移入')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 初始化窗口
    m = mwindow()
    m.show()
    sys.exit(app.exec_())
