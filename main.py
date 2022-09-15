import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QFileDialog, QAction,QMessageBox,QHBoxLayout
from PyQt5.QtGui import QPixmap,QFont
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_of_test_samples=1000

class Ui_MainWindow(object):    
    def setupUi(self, MainWindow):        
        MainWindow.resize(830, 616)
        MainWindow.setWindowTitle("MainWindow")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.dialogs = list()

        self.centralwidget.setObjectName("centralwidget")
        self.btnUpload = QtWidgets.QPushButton(self.centralwidget)
        self.btnUpload.setGeometry(QtCore.QRect(140, 420, 141, 121))
        self.btnUpload.setObjectName("btnUpload")
        self.btnSearch = QtWidgets.QPushButton(self.centralwidget)
        self.btnSearch.setGeometry(QtCore.QRect(340, 420, 141, 121))
        self.btnSearch.setObjectName("btnSearch")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(140, 260, 311, 41))
        self.textBrowser.setObjectName("textBrowser")
        self.lblResult = QtWidgets.QLabel(self.centralwidget)
        self.lblResult.setGeometry(QtCore.QRect(20, 265, 91, 31))
        self.lblResult.setObjectName("lblResult")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(650, 260, 171, 41))
        self.textBrowser_2.setObjectName("textBrowser_2")        
        self.lblAccuracy = QtWidgets.QLabel(self.centralwidget)
        self.lblAccuracy.setGeometry(QtCore.QRect(480, 265, 151, 31))
        self.lblAccuracy.setObjectName("lblAccuracy")

        self.btnClear = QtWidgets.QPushButton(self.centralwidget)
        self.btnClear.setGeometry(QtCore.QRect(560, 420, 141, 121))
        self.btnClear.setObjectName("btnClear")        

        self.cb_Model = QtWidgets.QComboBox(self.centralwidget)
        self.cb_Model.setGeometry(QtCore.QRect(140, 310, 311, 41))
        self.cb_Model.setObjectName("cb_Model")

        self.cb_SizeImages = QtWidgets.QComboBox(self.centralwidget)
        self.cb_SizeImages.setGeometry(QtCore.QRect(710, 310, 111, 41))
        self.cb_SizeImages.setObjectName("cb_SizeImages")
        self.lblModel = QtWidgets.QLabel(self.centralwidget)
        self.lblModel.setGeometry(QtCore.QRect(20, 320, 101, 31))
        self.lblModel.setObjectName("lblModel")
        self.lblSizeImages = QtWidgets.QLabel(self.centralwidget)
        self.lblSizeImages.setGeometry(QtCore.QRect(460, 320, 241, 31))
        self.lblSizeImages.setObjectName("lblSizeImages")

        self.lblImage = QtWidgets.QLabel(self.centralwidget)
        self.lblImage.setGeometry(QtCore.QRect(300, -10, 500, 221))
        self.lblImage.setObjectName("lblImage")        
       
        self.cb_Model.addItem("alexnet_224x224x3")    
        self.cb_Model.addItem("vgg16x224x224")


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 855, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.actionShowImageViewer = QtWidgets.QAction(MainWindow)
        self.actionShowImageViewer.setObjectName("actionShowImageViewer")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionShowImageViewer)
        self.menuFile.addAction(self.actionAbout)
        self.menuFile.addAction(self.actionExit)

        self.menubar.addAction(self.menuFile.menuAction())
        self.actionAbout.triggered.connect(self.about)
        self.actionShowImageViewer.triggered.connect(self.ShowImageViewer)
        self.actionExit.triggered.connect(QCoreApplication.instance().quit)     

        self.cb_SizeImages.addItem("224")

        font = QFont('Arial', 13)        
        self.cb_Model.setFont(font)
        self.cb_SizeImages.setFont(font)
        self.textBrowser.setFont(font)
        self.textBrowser_2.setFont(font)
        self.btnUpload.setFont(font)
        self.btnSearch.setFont(font)
        self.btnClear.setFont(font)
        self.lblText = QtWidgets.QLabel(self.centralwidget)
        self.lblText.setHidden(True)
        MainWindow.setCentralWidget(self.centralwidget)
        

        self.btnUpload.clicked.connect(self.openfile)
        self.btnSearch.clicked.connect(self.SearchResult)
        self.btnClear.clicked.connect(self.Clear)
        self.cb_Model.currentIndexChanged.connect(self.selectionchange)
        self.cb_SizeImages.currentIndexChanged.connect(self.selectionchange2)

        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow) 

    def selectionchange(self,i):
        self.cb_Model.currentText()
 
    def ShowImageViewer(self):           
        dialog = QImageViewer(self.centralwidget)
        self.dialogs.append(dialog)
        dialog.show()
        
    def selectionchange2(self,i):
        self.cb_SizeImages.currentText()    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate        
        MainWindow.setWindowTitle(_translate("MainWindow", "Kết quả thực nghiệm "))
        self.lblAccuracy.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:610;\">Xác suất</span></p></body></html>"))
        self.btnUpload.setText(_translate("MainWindow", "Tải ảnh"))
        self.lblResult.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:610;\">Kết Quả</span></p></body></html>"))
        self.btnClear.setText(_translate("MainWindow", "Làm Lại"))
        self.btnSearch.setText(_translate("MainWindow", "Tra cứu"))        
        self.lblModel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Mô hình</span></p><p><span style=\" font-size:14pt; font-weight:600;\"><br/></span></p></body></html>"))
        self.lblSizeImages.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Kích thước hinh ảnh</span></p><p><br/></p><p><br/></p></body></html>"))
        self.menuFile.setTitle(_translate("MainWindow", "Tập tin"))
        self.actionShowImageViewer.setText(_translate("MainWindow", "Xem Ảnh"))
        self.actionAbout.setText(_translate("MainWindow", "Hướng Dẫn"))
        self.actionExit.setText(_translate("MainWindow", "Thoát"))
    def Clear(self):        
        pixmap = QtGui.QPixmap()
        self.lblImage.setPixmap(pixmap)
        self.textBrowser.setText("")
        self.textBrowser_2.setText("")
        

    def about(self):
        QMessageBox.about(None, "Về giao diện nhận diện bệnh trên lá",
                          "<p>Về giao diện nhận diện bệnh trên lá, các bạn nhấn vào nút chọn ảnh và chọn ảnh trên thư mục testing mặc định."
                          "Sau đó bạn chọn mô hình nào mà bạn thích, tiếp theo bạn chọn định dạng hình ảnh để phù hợp với mô hình đó."
                          "Mô hình ở cuối tên có 2 chữ số cuối thường là định dạng hình ảnh của mô hình đó"
                          "Tiếp theo bạn nhấn vào nút tra cứu phần mềm sẽ chạy trong vài giây"
                          "và sẽ hiện ra kết quả là tên của bệnh và độ chính xác phần trăm"
                          "lên 2 thanh trống trên màn hình.</p>"
                          "<p>Tiếp theo bạn có thể tiếp tục nhấn nút chọn ảnh hoặc là nhấn nút làm lại"
                          "để tiếp tục phần tra cứu của mình."
                          "Ngoài ra giao diện còn có chức năng là hướng dẫn tức là chức năng này."
                          "</p>"
                          "<p>.</p>")
        
    def ModelAndSize(self):     
        model = pathlib.Path('D:/BaiTap/DoAnTotNghiep/model/')
        flower_model = keras.models.load_model(os.path.join(model, self.cb_Model.currentText()+'.h5'))
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join('D:/BaiTap/DoAnTotNghiep/dataset/testing'),
        image_size=(int(self.cb_SizeImages.currentText()),int(self.cb_SizeImages.currentText())),
        batch_size=num_of_test_samples)    

        class_names = test_ds.class_names
                        # specify test image filename
        test_img_fn = os.path.join(self.lblText.text())                

                        # read image
        img = keras.preprocessing.image.load_img(
        test_img_fn, target_size=(int(self.cb_SizeImages.currentText()),int(self.cb_SizeImages.currentText()))
        )

                        # convert image into array
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

                        # apply the model to predict flower name
        prob = flower_model.predict(img_array)
        predicted_class = class_names[np.argmax(prob)]
        b = 'Lá ớt chuông khỏe mạnh'
        a = 'Bệnh đốm lá ớt chuông'
        c = 'Bệnh cháy lá sớm khoai tây'
        d = 'Bệnh móc sương khoai tây'
        e = 'Bệnh đốm lá cà chua'
        if predicted_class  == 'Pepper__bell___healthy': 
            self.textBrowser.setText(b)
        if predicted_class  == 'Pepper__bell___Bacterial_spot': 
            self.textBrowser.setText(a)
        if predicted_class  == 'Potato___Early_blight': 
            self.textBrowser.setText(c) 
        if predicted_class  == 'Potato___Late_blight': 
            self.textBrowser.setText(d)
        if predicted_class  == 'Tomato__Target_Spot': 
            self.textBrowser.setText(e) 
        result = (np.max(prob))
        print('Predicted class: ', predicted_class, '(Probability = ', np.max(prob), ')')
        self.textBrowser_2.setText(str(result))     
    def SearchResult(self):        
        try:            
            c = self.cb_SizeImages.currentText() =='224'  
            a8 = self.cb_Model.currentText() == 'vgg16x224x224'  
            
            a10 = self.cb_Model.currentText() == "alexnet_224x224x3"
            
            if a8 and c or a10 and c:
                self.ModelAndSize()               
            else:            
                self.msg = QMessageBox()
                self.msg.setIcon(QMessageBox.Information)          
                # setting message for Message Box
                self.msg.setText("Mô hình không có định dạng hình ảnh này. Vui lòng chọn lại!!")              
                # setting Message box window title
                self.msg.setWindowTitle("Thông báo")              
                # declaring buttons on Message Box
                self.msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)                
                retval = self.msg.exec_()
        except:
                self.msg2 = QMessageBox()
                self.msg2.setIcon(QMessageBox.Information)          
                # setting message for Message Box
                self.msg2.setText("Vui lòng chọn ảnh ~!!!!!")              
                # setting Message box window title
                self.msg2.setWindowTitle("Thông báo")              
                # declaring buttons on Message Box
                self.msg2.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                # pixmap = QtGui.QPixmap()
                # self.lblImage.setPixmap(pixmap)                
                retval = self.msg2.exec_()

    def openfile(self):
        try:
            options = QFileDialog.Options()
        
            fileName ,_= QFileDialog.getOpenFileName(None, 'QFileDialog.getOpenFileName()', 'D:/BaiTap/DoAnTotNghiep/dataset/testing',
                                                  'Images (*.png *.jpeg *.jpg *.bmp *.gif)')        
            if fileName:
                self.pixmap = QtGui.QPixmap(fileName)
            if self.pixmap.isNull():
                QMessageBox.information(self, "Image Viewer", "Cannot load %s." % fileName)
                return            
            self.pixmap_resized = self.pixmap.scaled(356, 256, QtCore.Qt.KeepAspectRatio)        
            self.lblImage.setPixmap(self.pixmap_resized)            
            self.lblImage.adjustSize()            
            #print(fileName)
            self.lblText.setText(fileName)
            print(self.lblText.text()) 
        except :
            self.msg = QMessageBox()
            self.msg.setIcon(QMessageBox.Information)          
            # setting message for Message Box
            self.msg.setText("Vui lòng chọn lại ảnh !!")
            pixmap = QtGui.QPixmap()
            self.lblImage.setPixmap(pixmap)
            # setting Message box window title
            self.msg.setWindowTitle("Thông báo") 
            retval = self.msg.exec_()           
class QImageViewer(QMainWindow):
    def __init__(self, parent=None):
        super(QImageViewer, self).__init__(parent)

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setVisible(False)

        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Image Viewer")
        self.resize(800, 600)

    def open(self):
        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', 'D:/BaiTap/DoAnTotNghiep/test/testing',
                                                  'Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer", "Cannot load %s." % fileName)
                return

            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0

            self.scrollArea.setVisible(True)
            self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.updateActions()

            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()
    
    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O", triggered=self.open)
        self.printAct = QAction("&Print...", self, shortcut="Ctrl+P", enabled=False, triggered=self.print_)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F",
                                      triggered=self.fitToWindow)        
        self.aboutQtAct = QAction("About &Qt", self, triggered=qApp.aboutQt)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)        
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))
def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)    
    MainWindow.show()   
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()
    