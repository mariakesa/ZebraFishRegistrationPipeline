import argparse
import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow,QLabel
import pyqtgraph as pg
from master_script import masking, detrending, segment_detrended, segment_raw, dff_raw, dff_detrended
from PyQt5 import QtGui
from matplotlib.colors import hsv_to_rgb
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
sys.path.insert(0, 'C:/Users/koester_lab/Documents/Maria/ZebraFishRegistrationPipeline/Visualization')
from detrending_viz import detrending_verify

def parse_config_file(config_file):
    config_file=open(config_file,"r")
    print(config_file)
    lines=config_file.readlines()
    print(lines)

class PressToSelectButton(QLabel):
    def __init__(self,text,type,mainw,color_ind):
        super(PressToSelectButton, self).__init__()
        self.setAutoFillBackground(True)
        self.type=type
        self.mainw=mainw
        self.color_ind=color_ind
        #self.V_plot=V_plot
        palette = self.palette()
        colormap = cm.get_cmap("cool")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        color=matplotlib.colors.rgb2hex(lut[color_ind])
        self.qcolor=QtGui.QColor(color)
        palette.setColor(QtGui.QPalette.Window, self.qcolor)

        self.setPalette(palette)

        self.setText(text)

        self.setFixedWidth(800)
        self.setFixedHeight(100)

        self.setFont(QtGui.QFont('SansSerif', 30))
        self.setStyleSheet("color: white;")


    def mousePressEvent(self, event):
        palette = self.palette()
        self.qcolor.setAlpha(100)
        palette.setColor(QtGui.QPalette.Window, self.qcolor)

        self.setPalette(palette)

        if self.type=='masking':
            masking(self.mainw.config_dict['filepath'],0,
                            self.mainw.config_dict['save_folder_mask'],
                            self.mainw.config_dict['save_folder_masked'])
        if self.type=='detrending':
            detrending(self.mainw.config_dict['filepath'],
            self.mainw.config_dict['save_folder_masked'],
            self.mainw.config_dict['save_folder_detrending'])

        if self.type=='detrend_verify_plot':
            detrending_verify(self.mainw.config_dict['filepath'],
            self.mainw.config_dict['save_folder_detrending'],
            self.mainw.config_dict['save_folder_masked'],
            plane_ind=10)

        if self.type=='segment_raw':
            segment_raw(self.mainw.config_dict['filepath'],
            self.mainw.config_dict['save_folder_masked'],
            self.mainw.config_dict['save_folder_segmentation_raw'])

        if self.type=='segment_detrended':
            segment_detrended(self.mainw.config_dict['filepath'],
            self.mainw.config_dict['save_folder_detrending'],
            self.mainw.config_dict['save_folder_segmentation_detrended'])

        if self.type=='dff_raw':
            dff_raw(self.mainw.config_dict['filepath'],
            self.mainw.config_dict['save_folder_segmentation_raw'],
            self.mainw.config_dict['save_folder_dff_raw'])

        if self.type=='dff_detrended':
            dff_detrended(self.mainw.config_dict['filepath'],
            self.mainw.config_dict['save_folder_segmentation_detrended'],
            self.mainw.config_dict['save_folder_dff_detrended'])


class PilotGUI(QMainWindow):
    def __init__(self, config_file=''):
        super(PilotGUI, self).__init__()
        self.setStyleSheet("QMainWindow {background: 'black';}")
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
        color_ind=1
        self.mask_button=PressToSelectButton('Masking tool','masking',self,color_ind)
        #self.mask_button.clicked.connect(lambda:
                        #masking(self.config_dict['filepath'],0,
                        #self.config_dict['save_folder_mask'],
                        #self.config_dict['save_folder_masked']))
        self.l0.addWidget(self.mask_button,4,0)

        #Detrending button
        color_ind=25
        self.detrend_button=PressToSelectButton('Detrend','detrending',self,color_ind)
        self.l0.addWidget(self.detrend_button,8,0)

        color_ind=100
        self.detrend_verif_button=PressToSelectButton('Verify detrending','detrend_verify_plot',self,color_ind)
        self.l0.addWidget(self.detrend_verif_button,8,6)

        color_ind=150
        self.segment_raw_button=PressToSelectButton('Segment raw','segment_raw',self,color_ind)
        self.l0.addWidget(self.segment_raw_button,20,0)

        color_ind=200
        self.segment_detrended_button=PressToSelectButton('Segment detrended','segment_detrended',self,color_ind)
        self.l0.addWidget(self.segment_detrended_button,24,0)

        color_ind=200
        self.dff_raw_button=PressToSelectButton('Calculate dff raw','dff_raw',self,color_ind)
        self.l0.addWidget(self.calculate_dff_button,28,0)

        color_ind=200
        self.dff_detrended_button=PressToSelectButton('Calculate dff detrended','dff_detrended',self,color_ind)
        self.l0.addWidget(self.calculate_dff_button,32,0)

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
            if el[0]=='save_folder_detrending':
                self.config_dict['save_folder_detrending']=el[1]
            if el[0]=='save_folder_segmentation_detrended':
                self.config_dict['save_folder_segmentation_detrended']=el[1]
            if el[0]=='save_folder_segmentation_raw':
                self.config_dict['save_folder_segmentation_raw']=el[1]
            if el[0]=='save_folder_dff_detrended':
                self.config_dict['save_folder_dff_detrended']=el[1]
            if el[0]=='save_folder_dff_raw':
                self.config_dict['save_folder_dff_raw']=el[1]

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
