import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6 import uic
from PyQt6 import QtWidgets


class FastDDMApp(QMainWindow):
    def __init__(self):
        super(FastDDMApp, self).__init__()

        # load the ui file
        uic.loadUi("main.ui", self)

        # show the ui
        self.show()


# initialize the app
app = QApplication(sys.argv)
MainWindow = QMainWindow()

# create instance of app
ui = FastDDMApp()

# show the window and start the app
app.exec()
