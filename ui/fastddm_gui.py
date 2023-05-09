import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from main import Ui_MainWindow


class FastDDMApp(Ui_MainWindow):
    def __init__(self, window):
        self.setupUi(window)


app = QApplication(sys.argv)
MainWindow = QMainWindow()

# create instance of app
ui = FastDDMApp(MainWindow)

# show the window and start the app
MainWindow.show()
app.exec()
