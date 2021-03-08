#! /user/local/bin/python3.7

#######################################################
#   Author:     Tsung Lin Hsia
#   email:      thsia@purdue.edu
#   ID:         ee364g21
#   Date:       3/31/2019
#######################################################

import numpy as np
from scipy.spatial import Delaunay
import imageio
import os
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from Lab15.Morphing import *
from Lab15.MorphingGUI import *

class MorphingApp(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MorphingApp, self).__init__(parent)
        self.setupUi(self)
        #connect to respective member function
        self.startTriangles = []
        self.endTriangles = []
        self.startingImageArray = np.array([])
        self.endingImageArray = np.array([])
        self.startPoints = np.array([])
        self.endPoints = np.array([])
        self.tempStartPoints = []#get the user point
        self.tempEndPoints = []
        self.tempStartDot =  QGraphicsEllipseItem()
        self.tempEndDot =  QGraphicsEllipseItem()

        self.btnLoadStartingImage.clicked.connect(self.loadStartingImage)

        self.btnLoadEndingImage.clicked.connect(self.loadEndingImage)
        self.chkShowTriangles.stateChanged.connect(self.showTriangles)
        self.txtAlphaValue.setText('0.0')
        self.alpha = 0
        self.sliderAlpha.valueChanged.connect(self.adjustAlpha)

        self.txtAlphaValue.setReadOnly(True)
        self.sliderAlpha.setRange(0,20)
        self.sliderAlpha.setTickInterval(2)

        self.btnBlendImages.clicked.connect(self.blendImage)

        #user set points
        self.StartingImage.mousePressEvent = self.drawStartImagePoints
        self.EndingImage.mousePressEvent = self.drawEndImagePoints
        self.StartingImage.keyPressEvent = self.deleteStartImagePoint
        self.EndingImage.keyPressEvent = self.deleteEndImagePoint
        self.mousePressEvent = self.confirmedPoints

        #Initial State
        self.btnLoadStartingImage.setEnabled(True)
        self.btnLoadEndingImage.setEnabled(True)
        self.chkShowTriangles.setEnabled(False)
        self.sliderAlpha.setEnabled(False)
        self.txtAlphaValue.setEnabled(False)
        self.btnBlendImages.setEnabled(False)

        #flag
        self.f_startselect = False
        self.f_endselect = False
        self.f_pointsSet = False
        self.f_userPointSet = False

        self.f_readyforStartPoint = True
        self.f_readyforEndPoint = False
        self.f_plsConfirm = False
        self.f_deleteStartPointEnable = False
        self.f_deleteEndPointEnable = False

    def loadStartingImage(self):
        self.startfilePath,_ = QFileDialog.getOpenFileName(self, caption='Load Starting Image...',filter="Images(*.jpg *.png)")
        if not self.startfilePath:
            return
        else:
            leftgray= imageio.imread(self.startfilePath)
            self.startingImageArray = np.array(leftgray, dtype=np.uint8)
            tempScene = QGraphicsScene()
            temppixmap = QPixmap(self.startfilePath)
            tempScene.addPixmap(temppixmap)
            self.StartingImage.setScene(tempScene)
            self.StartingImage.fitInView(tempScene.sceneRect())

            if os.path.isfile(self.startfilePath+'.txt'):
                self.startPoints = np.loadtxt(self.startfilePath+'.txt',dtype=np.float64)

                brush = QBrush(Qt.red)
                pen = QPen(Qt.red)
                for x,y in self.startPoints:
                    dot = QGraphicsEllipseItem(0,0,15,15)#from QGraphicsItem rotate offset in the scene in stackOverflow
                    dot.setPos(QPointF(QPoint(x-6,y-6)))
                    dot.setBrush(brush)
                    dot.setPen(pen)
                    self.StartingImage.scene().addItem(dot)
                    self.StartingImage.fitInView(self.StartingImage.scene().sceneRect())
                    self.f_pointsSet = True

        self.f_startselect = True
        if self.f_endselect:
            self.chkShowTriangles.setEnabled(True)
            self.sliderAlpha.setEnabled(True)
            self.txtAlphaValue.setEnabled(True)
            self.btnBlendImages.setEnabled(True)

    def loadEndingImage(self):
        self.endfilePath,_ = QFileDialog.getOpenFileName(self, caption='Load Ending Image...', filter="Images(*.jpg *.png)")
        if not self.endfilePath:
            return
        else:
            rightgray = imageio.imread(self.endfilePath)
            self.endingImageArray = np.array(rightgray, dtype=np.uint8)
            tempScene = QGraphicsScene()
            temppixmap = QPixmap(self.endfilePath)
            tempScene.addPixmap(temppixmap)
            self.EndingImage.setScene(tempScene)
            self.EndingImage.fitInView(tempScene.sceneRect())

            if os.path.isfile(self.endfilePath+'.txt'):
                self.endPoints = np.loadtxt(self.endfilePath+'.txt',dtype=np.float64)
                brush = QBrush(Qt.red)
                pen = QPen(Qt.red)
                for x,y in self.endPoints:
                    dot = QGraphicsEllipseItem(0,0,15,15)#from QGraphicsItem rotate offset in the scene in stackOverflow
                    dot.setPos(QPointF(QPoint(x-6,y-6)))
                    dot.setBrush(brush)
                    dot.setPen(pen)
                    self.EndingImage.scene().addItem(dot)
                    self.EndingImage.fitInView(self.EndingImage.scene().sceneRect())
                    self.f_pointsSet = True

        self.f_endselect = True
        if self.f_startselect:
            self.chkShowTriangles.setEnabled(True)
            self.sliderAlpha.setEnabled(True)
            self.txtAlphaValue.setEnabled(True)
            self.btnBlendImages.setEnabled(True)

    def showTriangles(self):
        if self.chkShowTriangles.isChecked() and len(self.startPoints) != 0:
            if self.f_pointsSet and self.f_userPointSet: pen = QPen(Qt.green)
            elif self.f_pointsSet: pen = QPen(Qt.red)
            elif self.f_userPointSet: pen = QPen(Qt.blue)
            pen.setWidth(4)
            self.startTriangles = []
            self.endTriangles = []
            self.startDelaunay = Delaunay(self.startPoints)
            for delaunay in self.startDelaunay.simplices:
                newSP = np.array([[self.startPoints[delaunay[0], 0], self.startPoints[delaunay[0], 1]],
                                  [self.startPoints[delaunay[1], 0], self.startPoints[delaunay[1], 1]],
                                  [self.startPoints[delaunay[2], 0], self.startPoints[delaunay[2], 1]]],dtype = np.float64)
                newEP = np.array([[self.endPoints[delaunay[0], 0], self.endPoints[delaunay[0], 1]],
                                  [self.endPoints[delaunay[1], 0], self.endPoints[delaunay[1], 1]],
                                  [self.endPoints[delaunay[2], 0], self.endPoints[delaunay[2], 1]]],dtype = np.float64)
                self.startTriangles.append(Triangle(self.startPoints[delaunay]))
                self.endTriangles.append(Triangle(self.endPoints[delaunay]))
                s1 = QGraphicsLineItem(QLineF(QPointF(QPoint(newSP[0][0],newSP[0][1])),QPoint(newSP[1][0],newSP[1][1])))#.setPen(pen)
                s2 = QGraphicsLineItem(QLineF(QPointF(QPoint(newSP[1][0],newSP[1][1])),QPoint(newSP[2][0],newSP[2][1])))#.setPen(pen)
                s3 = QGraphicsLineItem(QLineF(QPointF(QPoint(newSP[2][0],newSP[2][1])),QPoint(newSP[0][0],newSP[0][1])))#.setPen(pen)
                e1 = QGraphicsLineItem(QLineF(QPointF(QPoint(newEP[0][0],newEP[0][1])),QPoint(newEP[1][0],newEP[1][1])))#.setPen(pen)
                e2 = QGraphicsLineItem(QLineF(QPointF(QPoint(newEP[1][0],newEP[1][1])),QPoint(newEP[2][0],newEP[2][1])))#.setPen(pen)
                e3 = QGraphicsLineItem(QLineF(QPointF(QPoint(newEP[2][0],newEP[2][1])),QPoint(newEP[0][0],newEP[0][1])))#.setPen(pen)
                s1.setPen(pen)
                s2.setPen(pen)
                s3.setPen(pen)
                e1.setPen(pen)
                e2.setPen(pen)
                e3.setPen(pen)
                self.StartingImage.scene().addItem(s1)
                self.StartingImage.scene().addItem(s2)
                self.StartingImage.scene().addItem(s3)
                self.EndingImage.scene().addItem(e1)
                self.EndingImage.scene().addItem(e2)
                self.EndingImage.scene().addItem(e3)
        else:
            for line in self.StartingImage.items():
                if type(line) is QGraphicsLineItem:
                    self.StartingImage.scene().removeItem(line)
            for line in self.EndingImage.items():
                if type(line) is QGraphicsLineItem:
                    self.EndingImage.scene().removeItem(line)

    def adjustAlpha(self):
        self.alpha = round(float(self.sliderAlpha.value()) / 20.0,2)
        self.txtAlphaValue.setText(str(self.sliderAlpha.value() / 20.0))#not sure the value
        #print(self.alpha)

    def blendImage(self):
        if len(self.startingImageArray) == 0 or len(self.startTriangles) == 0 or len(self.endingImageArray) == 0 or len(self.endTriangles) == 0:
            return
        image = Morpher(self.startingImageArray,self.startTriangles,self.endingImageArray,self.endTriangles).getImageAtAlpha(self.alpha)
        image = QtGui.QImage(image, image.shape[1],image.shape[0], image.shape[1], QtGui.QImage.Format_Indexed8)#onstructs an image with the given width, height and format, that uses an existing read-only memory buffer, data.
        pix = QtGui.QPixmap.fromImage(image)
        tempScene = QGraphicsScene()
        tempScene.addPixmap(pix)
        self.BlendingResult.setScene(tempScene)
        self.BlendingResult.fitInView(tempScene.sceneRect())

    def drawStartImagePoints(self,event):
        self.confirmedPoints(event)
        if(len(self.tempStartPoints) == len(self.tempEndPoints)) and self.f_readyforStartPoint and len(self.startingImageArray) != 0 :
            actualP = self.StartingImage.mapToScene(event.pos())
            self.tempStartDot = QGraphicsEllipseItem(0, 0, 10, 10)
            self.tempStartDot.setPos(actualP)
            self.tempStartDot.setBrush(QBrush(Qt.green))
            self.StartingImage.scene().addItem(self.tempStartDot)
            self.StartingImage.fitInView(self.StartingImage.scene().sceneRect())
            self.tempStartPoints.append([actualP.x(),actualP.y()])
            self.f_readyforStartPoint = False
            self.f_readyforEndPoint = True
            self.f_plsConfirm = False
            self.f_deleteStartPointEnable = True
            self.f_deleteEndPointEnable = False

    def drawEndImagePoints(self,event):

        if(len(self.tempStartPoints) != len(self.tempEndPoints)) and self.f_readyforEndPoint and self.f_plsConfirm is False and len(self.endingImageArray) != 0:
            actualP = self.StartingImage.mapToScene(event.pos())
            self.tempEndDot = QGraphicsEllipseItem(0, 0, 10, 10)
            self.tempEndDot.setPos(actualP)
            self.tempEndDot.setBrush(QBrush(Qt.green))
            self.EndingImage.scene().addItem(self.tempEndDot)
            self.EndingImage.fitInView(self.EndingImage.scene().sceneRect())
            self.tempEndPoints.append([actualP.x(),actualP.y()])
            self.f_readyforStartPoint = False
            self.f_readyforEndPoint = False
            self.f_plsConfirm = True
            self.f_deleteStartPointEnable = False
            self.f_deleteEndPointEnable = True

    def deleteStartImagePoint(self,event):
        if(len(self.tempStartPoints) != len(self.tempEndPoints)) and self.f_deleteStartPointEnable:
            self.StartingImage.scene().removeItem(self.tempStartDot)
            self.tempStartPoints.pop()
            self.f_readyforStartPoint = True
            self.f_readyforEndPoint = False
            self.f_plsConfirm = False
            self.f_deleteStartPointEnable = False
            self.f_deleteEndPointEnable = False

    def deleteEndImagePoint(self,event):
        if(len(self.tempStartPoints) == len(self.tempEndPoints)) and self.f_deleteEndPointEnable:
            self.EndingImage.scene().removeItem(self.tempEndDot)
            self.tempEndPoints.pop()
            self.f_readyforStartPoint = False
            self.f_readyforEndPoint = True
            self.f_plsConfirm = False
            self.f_deleteStartPointEnable = False
            self.f_deleteEndPointEnable = False

    def confirmedPoints(self,event):
        if self.f_plsConfirm:
            self.StartingImage.scene().removeItem(self.tempStartDot)
            self.EndingImage.scene().removeItem(self.tempEndDot)
            self.tempStartDot.setBrush(QBrush(Qt.blue))
            self.tempEndDot.setBrush(QBrush(Qt.blue))
            self.StartingImage.scene().addItem(self.tempStartDot)
            self.StartingImage.fitInView(self.StartingImage.scene().sceneRect())
            self.EndingImage.scene().addItem(self.tempEndDot)
            self.EndingImage.fitInView(self.EndingImage.scene().sceneRect())
            tempS = np.array([self.tempStartPoints[-1][0],self.tempStartPoints[-1][1]],dtype = np.float64)
            tempE = np.array([self.tempEndPoints[-1][0],self.tempEndPoints[-1][1]],dtype = np.float64)
            if len(self.startPoints):
                self.startPoints = np.vstack((self.startPoints,tempS))
                self.endPoints = np.vstack((self.endPoints,tempE))
                with open(self.startfilePath + '.txt', 'a') as file:
                    file.write('\n' + str(round(tempS[0], 1)) + ' ' + str(round(tempS[1], 1)))
                with open(self.endfilePath + '.txt', 'a') as file:
                    file.write('\n' + str(round(tempE[0], 1)) + ' ' + str(round(tempE[1], 1)))
            else:
                self.startPoints = tempS
                self.endPoints = tempE
                with open(self.startfilePath + '.txt', 'w') as file:
                    file.write(str(round(tempS[0], 1)) + ' ' + str(round(tempS[1], 1)))
                with open(self.endfilePath + '.txt', 'w') as file:
                    file.write(str(round(tempE[0], 1)) + ' ' + str(round(tempE[1], 1)))

            self.f_readyforStartPoint = True
            self.f_readyforEndPoint = False
            self.f_plsConfirm = False
            self.f_deleteStartPointEnable = False
            self.f_deleteEndPointEnable = False
            self.f_userPointSet = True





if __name__ == "__main__":
    currentApp = QApplication(sys.argv)
    currentForm = MorphingApp()
    currentForm.show()
    currentApp.exec_()