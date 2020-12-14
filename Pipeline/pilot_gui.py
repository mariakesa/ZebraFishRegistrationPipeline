import argparse
import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg
from master_script import masking
from PyQt5 import QtGui

def parse_config_file(config_file):
    config_file=open(config_file,"r")
    print(config_file)
    lines=config_file.readlines()
    print(lines)

class PilotGUI(QMainWindow):
    def __init__(self, config_file=''):
        super(PilotGUI, self).__init__()
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setGeometry(70,70,1100,900)
        self.setWindowTitle('Visualize data')
        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QtGui.QGridLayout()
        self.cwidget.setLayout(self.l0)

        self.menu_config_load()

        #Register button

        #Mask button
        self.mask_button=self.ens_selector=QtGui.QPushButton('Masking tool')
        self.mask_button.clicked.connect(lambda:
                        masking(self.config_dict['filepath'],0,
                        self.config_dict['save_folder_mask'],
                        self.config_dict['save_folder_masked']))
        self.l0.addWidget(self.mask_button,4,0)

    def menu_config_load(self):
        self.main_menu=self.menuBar()
        self.config = self.main_menu.addMenu('&Config file')
        self.load_config = QtGui.QAction("Load config file", self)
        self.load_config.triggered.connect(lambda: self.load_dialog())
        self.config.addAction(self.load_config)

    def load_dialog(self):
        options=QtGui.QFileDialog.Options()
        options |= QtGui.QFileDialog.DontUseNativeDialog
        name = QtGui.QFileDialog.getOpenFileName(
            self, "Load config", options=options
        )
        self.config_file = name[0]
        self.parse_config_file()

    def parse_config_file(self):
        config_file=open(self.config_file,"r")
        print(config_file)
        lines=config_file.readlines()
        print(lines)
        self.config_dict={}
        for line in lines:
            el=line.strip('\n').split(' ')
            if el[0]=='filepath':
                self.config_dict['filepath']=el[1]
            if el[0]=='save_folder_mask':
                self.config_dict['save_folder_mask']=el[1]
            if el[0]=='save_folder_masked':
                self.config_dict['save_folder_masked']=el[1]
        print(self.config_dict)


def main():
    app = QtGui.QApplication(sys.argv)
    main = PilotGUI()
    main.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run pilot gui according taking in as input the config file.')
    parser.add_argument('--config_path', help='Path to the config file')
    args = parser.parse_args()
    config_file=args.config_path
    main()
    #parse_config_file(os.path.normpath(str(config_file)))
