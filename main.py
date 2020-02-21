import sys
import os
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QErrorMessage, QMessageBox, QFormLayout
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from gui import Ui_MainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from reconstruction import reconstruction_3d, disp_to_ply

f_name_format = '<html><head/><body><p><span style=" font-size:10pt;">{}</span></p></body></html>'
mtch_format = '<html><head/><body><p><span style=" font-size:10pt;">{:.2f}</span></p></body></html>'

class Reconstruction3dGUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Reconstruction3dGUI, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('icons/app_logo.png'))

        self.workInProgressLabel.setPixmap(QtGui.QPixmap(os.getcwd() + '/icons/green.png'))

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.imageView.addWidget(self.canvas)

        self.img_l_name = None
        self.img_r_name = None

        self.ply_out_path = None
        self.calib_file = None

        # images
        self.img_l = None
        self.img_r = None
        self.img_l_ep = None
        self.img_r_ep = None
        self.img_l_rect = None
        self.img_r_rect = None
        self.disparity = None

        # sift params
        self.n_features = None
        self.sigma = None
        self.edge = None
        self.contrast = None
        self.layers = None

        # features matching params
        self.matcher_threshold = 0.8

        # stereoSGBM
        self.win_size = 400
        self.sgbm_block_size = 7
        self.ratio = 12
        self.disp_max_diff = 1
        self.spakle_range = 5

        self.set_validators()
        self.bindActions()


    def set_validators(self):
        self.siftFeaturesEdit.setValidator(QIntValidator(0, 1000))
        self.sigmaSiftEdit.setValidator(QDoubleValidator(0, 100, 5))
        self.edgeThresholdEdit.setValidator(QDoubleValidator(0, 100, 5))
        self.contrastThresholdEdit.setValidator(QDoubleValidator(0, 100, 5))
        self.octaveEdit.setValidator(QIntValidator(0, 1000))

        self.winSizeEdit.setValidator(QIntValidator(0, 1000))
        self.sgbmBlockSizeEdit.setValidator(QIntValidator(5, 255))
        self.ratioEdit.setValidator(QIntValidator(0, 1000))
        self.dsipMaxEdit.setValidator(QIntValidator(0, 1000))
        self.spakleEdit.setValidator(QIntValidator(0, 1000))

    def bindActions(self):
        self.loadFileButton.clicked.connect(self.open_images)
        self.matcherThreshold.valueChanged.connect(self.set_matcher_threshold)
        self.okButton.clicked.connect(self.compute)
        self.viewTypeCombo.currentIndexChanged.connect(self.switch_view)
        self.defaultButton.clicked.connect(self.restore_default_values)
        self.generatePLYButton.clicked.connect(self.generatePLY)

    def generatePLY(self):
        if self.disparity is None or self.calib_file is None or self.ply_out_path is None:
            self.error_message('Error', 'missing one or more file', 'cannot find file disparity or calibration file')
            return
        self.workInProgressLabel.setPixmap(QtGui.QPixmap(os.getcwd() + '/icons/red.png'))
        # force gui to update
        QApplication.instance().processEvents()

        disp_to_ply(self.img_l_name, self.calib_file, self.disparity, self.ply_out_path)

        self.workInProgressLabel.setPixmap(QtGui.QPixmap(os.getcwd() + '/icons/green.png'))

    def restore_default_values(self):

        self.matcherThreshold.setValue(80)
        self.siftFeaturesEdit.setText('0')
        self.sigmaSiftEdit.setText('1.6')
        self.edgeThresholdEdit.setText('10')
        self.contrastThresholdEdit.setText('0.04')
        self.octaveEdit.setText('3')
        
        self.winSizeEdit.setText('400')
        self.sgbmBlockSizeEdit.setText('7')
        self.ratioEdit.setText('12')
        self.dsipMaxEdit.setText('1')
        self.spakleEdit.setText('5')

    def error_message(self, title, msg, info=''):
        error_dialog = QMessageBox()
        error_dialog.setWindowIcon(QtGui.QIcon('icons/app_logo.png'))
        error_dialog.setWindowTitle(title)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setText(msg)
        error_dialog.setInformativeText(info)
        error_dialog.exec_()

    def switch_view(self, i):

        images = (self.img_l, self.img_r, self.img_l_ep, self.img_r_ep,
                  self.img_l_rect, self.img_r_rect, self.disparity)

        if any(image is None for image in images):

            self.error_message('Error', 'Images not available',
                               'no image to display \nselect images and press Ok')
            return

        if   i == 0:
            self.set_view(self.img_l, self.img_r)
        elif i == 1:
            self.set_view(self.img_l_ep, self.img_r_ep)
        elif i == 2:
            self.set_view(self.img_l_rect, self.img_r_rect)
        elif i == 3:
            self.set_view(self.disparity)

    def set_view(self, img1, img2=None):
        # delete old image
        self.figure.clear()

        if img2 is None:
            im_1 = self.figure.add_subplot(111)
            im_1.imshow(img1, cmap='gray')
        else:
            im_1 = self.figure.add_subplot(121)
            im_2 = self.figure.add_subplot(122)
            im_1.imshow(img1, cmap='gray')
            im_2.imshow(img2, cmap='gray')

        # refresh canvas
        self.canvas.draw()

    def open_images(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Image Files (*.jpg *.png)", options=options)
        if file_path:
            self.img_l_name = file_path
            start = file_path.rfind('/')
            end = file_path.rfind('.')
            name = file_path[start+1:end]
            self.filenameLabel.setText(f_name_format.format(name))
            # path to right file
            self.img_r_name = file_path.replace('L.png', 'R.png')
            # path to calibration file
            self.calib_file = file_path.replace('L.png', '.txt')
            # path to ply file
            self.ply_out_path = file_path.replace('L.png', '.ply').replace('images', 'out')

            if os.path.isfile(self.img_r_name) is False:
                self.error_message('Error', 'Right image not found')
            if os.path.isfile(self.calib_file) is False:
                self.error_message('Error', 'calibration file not found')

    def set_matcher_threshold(self):
        value = self.matcherThreshold.value()
        self.matcher_threshold = value / 100
        matcher_string = mtch_format.format(self.matcher_threshold)
        self.matcherThresholdLabel.setText(matcher_string)

    def get_values(self):
        # SIFT params
        self.n_features = int(self.siftFeaturesEdit.text()) if self.siftFeaturesEdit.text() != '' else 0
        self.sigma = float(self.sigmaSiftEdit.text()) if self.sigmaSiftEdit.text() != '' else 1.6
        self.edge = float(self.edgeThresholdEdit.text()) if self.edgeThresholdEdit.text() != '' else 10
        self.contrast = float(self.contrastThresholdEdit.text()) if self.contrastThresholdEdit.text() != '' else 0.04
        self.layers = int(self.octaveEdit.text()) if self.octaveEdit.text() != '' else 3
        # stereoSGBM params
        self.win_size = int(self.winSizeEdit.text()) if self.winSizeEdit.text() != '' else 400
        self.sgbm_block_size = int(self.sgbmBlockSizeEdit.text()) if self.sgbmBlockSizeEdit.text() != '' else 7
        self.ratio = int(self.ratioEdit.text()) if self.ratioEdit.text() != '' else 12
        self.disp_max_diff = int(self.dsipMaxEdit.text()) if self.dsipMaxEdit.text() != '' else 1
        self.spakle_range = int(self.spakleEdit.text()) if self.spakleEdit.text() != '' else 5

    def compute(self):

        if self.img_l_name is None or self.img_r_name is None:
            self.error_message('Error', 'Images not available', 'image not selected')
            return

        self.get_values()

        self.workInProgressLabel.setPixmap(QtGui.QPixmap(os.getcwd() + '/icons/red.png'))
        # force gui to update
        QApplication.instance().processEvents()

        result = reconstruction_3d(self.img_l_name, self.img_r_name,
                                   self.n_features, self.sigma, self.edge, self.contrast, self.layers,
                                   self.matcher_threshold, self.win_size, self.sgbm_block_size, self.ratio,
                                   self.disp_max_diff, self.spakle_range)
        
        self.workInProgressLabel.setPixmap(QtGui.QPixmap(os.getcwd() + '/icons/green.png'))

        if type(result) == bool and not result:
            self.error_message('Error', 'Cannot calculate fundamental matrix', 'matched keypoint less than 8')
            return

        self.img_l = result[0]
        self.img_r = result[1]
        self.img_l_ep = result[2]
        self.img_r_ep = result[3]
        self.img_l_rect = result[4]
        self.img_r_rect = result[5]
        self.disparity = result[6]

        self.switch_view(self.viewTypeCombo.currentIndex())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Reconstruction3dGUI()
    window.show()
    app.exec_()
