import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6 import uic
from PyQt6 import QtWidgets


class FastDDMApp(QMainWindow):
    def __init__(self):
        super(FastDDMApp, self).__init__()

        # load the ui file
        uic.loadUi("main.ui", self)

        self.show_message("FastDDM application started...")

        # show to status bar when tabs are changed
        self.tabWidget.currentChanged.connect(self.tab_changed)

        # add tabs
        self.load_tab_widget()
        self.load_tab_widget()

        # show the ui
        self.show()

    def show_message(self, text):
        self.statusBar().showMessage(text)

    def tab_changed(self):
        # get current tab index
        idx = self.tabWidget.currentIndex()
        self.show_message(f"Switched to tab {idx}")

    def load_tab_widget(self):
        # create dummy tab an title
        tab = QtWidgets.QWidget()
        title = "Dummy tab"

        # add grid layout to tab
        grid_layout = QtWidgets.QGridLayout(tab)
        # add vertical layout in grid layout
        vertical_layout = QtWidgets.QVBoxLayout()
        grid_layout.addLayout(vertical_layout, 0, 0, 1, 1)

        # add widget to tab
        widget = QtWidgets.QPushButton(tab)
        widget.setText(f"Push Button {self.tabWidget.count()}")
        vertical_layout.addWidget(widget)

        # add tab
        self.tabWidget.addTab(tab, title)


# initialize the app
app = QApplication(sys.argv)
MainWindow = QMainWindow()

# create instance of app
ui = FastDDMApp()

# show the window and start the app
app.exec()
