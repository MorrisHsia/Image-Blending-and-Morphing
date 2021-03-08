import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import Delaunay
import imageio
import os
from copy import deepcopy
import time
import math
from PIL import Image,ImageDraw

DataPath = os.path.expanduser("/home/ecegridfs/a/ee364g21/Documents/labs-MorrisHsia/Lab12/TestData")

def loadTriangles(leftPointFilePath,rightPointFilePath):
    with open(leftPointFilePath,'r') as file:
        leftlist = np.loadtxt(file, dtype=np.float64)  # .shape)
    with open(rightPointFilePath, 'r') as file:
        rightlist = np.loadtxt(file, dtype=np.float64)
    templeftTriangles = np.array(leftlist, dtype=np.float64)
    temprightTriangles = np.array(rightlist, dtype=np.float64)
    delaunay = Delaunay(templeftTriangles).simplices

    leftTriangles = []
    rightTriangles = []
    for i in delaunay:
        leftTriangles.append(Triangle(templeftTriangles[i]))
        rightTriangles.append(Triangle(temprightTriangles[i]))
    return (leftTriangles,rightTriangles)

class Triangle:
    vertices = np.zeros((3,2),dtype=np.float64)
    def __init__(self,vertice):
        for i in vertice:
            for j in i:
                if type(j) != np.float64:
                     raise ValueError("The type of input is not np.float64")
        if vertice.shape != (3,2):
            raise ValueError("The dimension is not 3 x 2")
        self.vertices = vertice


    def getPoints(self):
        x1,y1 = self.vertices[0]
        x2,y2 = self.vertices[1]
        x3,y3 = self.vertices[2]
        xcord = [x1,x2,x3]
        ycord = [y1,y2,y3]
        xmin = int(math.ceil(min(x1,x2,x3)))
        xmax = int(math.floor(max(x1,x2,x3)))
        ymin = int(math.ceil(min(y1,y2,y3)))
        ymax = int(math.floor(max(y1,y2,y3)))
        #xmin_index = np.argmin([x1,x2,x3])
        #xmax_index = np.argmax([x1,x2,x3])
        #ymin_index = np.argmin([y1, y2, y3])
        #ymax_index = np.argmax([y1, y2, y3])
        #xcord[xmin_index] = xmin
        #xcord[xmax_index] = xmax
        #ycord[ymin_index] = ymin
        #ycord[ymax_index] = ymax
        #xcord[3 - int(xmin_index) - int(xmax_index)] = int(math.ceil(xcord[3 - int(xmin_index) - int(xmax_index)]))
        #ycord[3 - int(ymin_index) - int(ymax_index)] = int(math.ceil(ycord[3 - int(ymin_index) - int(ymax_index)]))


        im=Image.new('L',(xmax+1,ymax+1),color=0)
        ImageDraw.Draw(im).polygon([(xcord[0],ycord[0]),(xcord[1],ycord[1]),(xcord[2],ycord[2])],outline = 255,fill = 255)
        nonempty = np.nonzero(np.array(im))
        result=np.transpose(nonempty)
        result = np.array(result,dtype = np.float)
        #temp = []
        #mask = np.array(im)
        #points_array = np.where(mask==255)
        #for i in range(len(points_array[0])):
        #    temp.append((points_array[1][i],points_array[0][i]))
        #result = np.array(temp,dtype=np.float64)

        return result


class Morpher:
    def __init__(self,leftImage,leftTriangles,rightImage,rightTriangles):
        if not((type(rightImage) == np.ndarray) and (type(leftImage) == np.ndarray) and isinstance(leftTriangles[0],Triangle) and isinstance(rightTriangles[0],Triangle) and type(leftImage[0][0]) == np.uint8 and type(rightImage[0][0]) == np.uint8):
            raise TypeError("Check arguments!")
        self.leftImage= leftImage
        self.rightImage = rightImage
        self.leftTriangles = leftTriangles
        self.rightTriangles = rightTriangles

    def affineTransform(self,index):
        A_left = np.array([[self.leftTriangles[index].vertices[0][0], self.leftTriangles[index].vertices[0][1], 1, 0, 0, 0],[0, 0, 0, self.leftTriangles[index].vertices[0][0], self.leftTriangles[index].vertices[0][1],1],
                          [self.leftTriangles[index].vertices[1][0], self.leftTriangles[index].vertices[1][1], 1, 0, 0, 0],[0, 0, 0, self.leftTriangles[index].vertices[1][0], self.leftTriangles[index].vertices[1][1],1],
                          [self.leftTriangles[index].vertices[2][0], self.leftTriangles[index].vertices[2][1], 1, 0, 0, 0],[0, 0, 0, self.leftTriangles[index].vertices[2][0], self.leftTriangles[index].vertices[2][1],1]], dtype = np.float64)
        A_right = np.array([[self.rightTriangles[index].vertices[0][0], self.rightTriangles[index].vertices[0][1], 1, 0, 0, 0],[0, 0, 0, self.rightTriangles[index].vertices[0][0], self.rightTriangles[index].vertices[0][1],1],
                          [self.rightTriangles[index].vertices[1][0], self.rightTriangles[index].vertices[1][1], 1, 0, 0, 0],[0, 0, 0, self.rightTriangles[index].vertices[1][0], self.rightTriangles[index].vertices[1][1],1],
                          [self.rightTriangles[index].vertices[2][0], self.rightTriangles[index].vertices[2][1], 1, 0, 0, 0],[0, 0, 0, self.rightTriangles[index].vertices[2][0], self.rightTriangles[index].vertices[2][1],1]], dtype = np.float64)


        #print(self.middleTriangles[7].vertices[2][0])
        #print(self.leftTriangles[7].vertices[2][0])

        b_middle = np.array([[self.middleTriangles[index].vertices[0][0]],[self.middleTriangles[index].vertices[0][1]],[self.middleTriangles[index].vertices[1][0]],
                             [self.middleTriangles[index].vertices[1][1]],[self.middleTriangles[index].vertices[2][0]],[self.middleTriangles[index].vertices[2][1]]],dtype = np.float64)
        h_left = np.linalg.solve(A_left,b_middle)
        h_right = np.linalg.solve(A_right,b_middle)


        H_left = np.array([[h_left[0],h_left[1],h_left[2]], [h_left[3],h_left[4],h_left[5]], [0,0,1]], dtype = np.float64)
        H_right = np.array([[h_right[0],h_right[1],h_right[2]], [h_right[3],h_right[4],h_right[5]], [0,0,1]], dtype = np.float64)
        invertH_left = np.linalg.inv(H_left)
        invertH_right = np.linalg.inv(H_right)

        return invertH_left,invertH_right


    def getImageAtAlpha(self,alpha):
        inter_left = RectBivariateSpline(np.arange(self.leftImage.shape[0]),np.arange(self.leftImage.shape[1]), z=self.leftImage,kx=1,ky=1)
        inter_right = RectBivariateSpline(np.arange(self.rightImage.shape[0]),np.arange(self.rightImage.shape[1]), z=self.rightImage,kx=1,ky=1)

        empty_left = np.zeros(shape = self.leftImage.shape)
        empty_right = np.zeros(shape = self.rightImage.shape)

        self.middleTriangles = deepcopy(self.leftTriangles)
        for i in range(len(self.leftTriangles)):
            for j in range(len(self.leftTriangles[0].vertices)):
                for k in range(len(self.leftTriangles[0].vertices[0])):
                    self.middleTriangles[i].vertices[j][k] = ((1-alpha)*self.leftTriangles[i].vertices[j][k] + alpha*self.rightTriangles[i].vertices[j][k])
        #print(self.middleTriangles[183].vertices)
        #print(self.leftTriangles[183].vertices)
        #self.affineTransform(0)
        #for i in range()
        for index in range(len(self.middleTriangles)):
            invleftH, invrightH = self.affineTransform(index)
            points = self.middleTriangles[index].getPoints()
            for temp in points:
                middlexy = np.array([[temp[1]],[temp[0]],[1]],dtype = np.float64)
                leftpoints = np.matmul(invleftH,middlexy)
                rightpoints = np.matmul(invrightH,middlexy)
                x1 = int(temp[1])
                y1 = int(temp[0])
                empty_left[y1,x1] = inter_left.ev(leftpoints[1],leftpoints[0])
                empty_right[y1,x1] = inter_right.ev(rightpoints[1],rightpoints[0])
                #empty_left[x1,y1] = inter_left.ev(leftpoints[0],leftpoints[1])
                #empty_right[x1,y1] = inter_right.ev(rightpoints[0],rightpoints[1])

        blend = np.asarray(empty_left*(1-alpha)+empty_right*alpha,dtype = np.uint8)
        #print(blend)
        imageio.imwrite("result.png",blend)
        #print(leftpoints.shape)
        return blend
    def saveVideo(self,targetFilePath,frameCount,framerate,includeReversed=True):
        pass





class ColorMorpher:
    def __init__(self,leftImage,leftTriangles,rightImage,rightTriangles):
        if not((type(rightImage) == np.ndarray) and (type(leftImage) == np.ndarray) and isinstance(leftTriangles[0],Triangle) and isinstance(rightTriangles[0],Triangle) and type(leftImage[0][0][0]) == np.uint8 and type(rightImage[0][0][0]) == np.uint8):
            raise TypeError("Check arguments!")
        self.leftImage= leftImage
        self.rightImage = rightImage
        self.leftTriangles = leftTriangles
        self.rightTriangles = rightTriangles

    def affineTransform(self,index):
        A_left = np.array([[self.leftTriangles[index].vertices[0][0], self.leftTriangles[index].vertices[0][1], 1, 0, 0, 0],[0, 0, 0, self.leftTriangles[index].vertices[0][0], self.leftTriangles[index].vertices[0][1],1],
                          [self.leftTriangles[index].vertices[1][0], self.leftTriangles[index].vertices[1][1], 1, 0, 0, 0],[0, 0, 0, self.leftTriangles[index].vertices[1][0], self.leftTriangles[index].vertices[1][1],1],
                          [self.leftTriangles[index].vertices[2][0], self.leftTriangles[index].vertices[2][1], 1, 0, 0, 0],[0, 0, 0, self.leftTriangles[index].vertices[2][0], self.leftTriangles[index].vertices[2][1],1]], dtype = np.float64)
        A_right = np.array([[self.rightTriangles[index].vertices[0][0], self.rightTriangles[index].vertices[0][1], 1, 0, 0, 0],[0, 0, 0, self.rightTriangles[index].vertices[0][0], self.rightTriangles[index].vertices[0][1],1],
                          [self.rightTriangles[index].vertices[1][0], self.rightTriangles[index].vertices[1][1], 1, 0, 0, 0],[0, 0, 0, self.rightTriangles[index].vertices[1][0], self.rightTriangles[index].vertices[1][1],1],
                          [self.rightTriangles[index].vertices[2][0], self.rightTriangles[index].vertices[2][1], 1, 0, 0, 0],[0, 0, 0, self.rightTriangles[index].vertices[2][0], self.rightTriangles[index].vertices[2][1],1]], dtype = np.float64)


        #print(self.middleTriangles[7].vertices[2][0])
        #print(self.leftTriangles[7].vertices[2][0])

        b_middle = np.array([[self.middleTriangles[index].vertices[0][0]],[self.middleTriangles[index].vertices[0][1]],[self.middleTriangles[index].vertices[1][0]],
                             [self.middleTriangles[index].vertices[1][1]],[self.middleTriangles[index].vertices[2][0]],[self.middleTriangles[index].vertices[2][1]]],dtype = np.float64)
        h_left = np.linalg.solve(A_left,b_middle)
        h_right = np.linalg.solve(A_right,b_middle)


        H_left = np.array([[h_left[0],h_left[1],h_left[2]], [h_left[3],h_left[4],h_left[5]], [0,0,1]], dtype = np.float64)
        H_right = np.array([[h_right[0],h_right[1],h_right[2]], [h_right[3],h_right[4],h_right[5]], [0,0,1]], dtype = np.float64)
        invertH_left = np.linalg.inv(H_left)
        invertH_right = np.linalg.inv(H_right)

        return invertH_left,invertH_right


    def getImageAtAlpha(self,alpha):
        inter_left_r = RectBivariateSpline(np.arange(self.leftImage.shape[0]),np.arange(self.leftImage.shape[1]), z=self.leftImage[:,:,0],kx=1,ky=1)
        inter_left_g = RectBivariateSpline(np.arange(self.leftImage.shape[0]),np.arange(self.leftImage.shape[1]), z=self.leftImage[:,:,1],kx=1,ky=1)
        inter_left_b = RectBivariateSpline(np.arange(self.leftImage.shape[0]),np.arange(self.leftImage.shape[1]), z=self.leftImage[:,:,2],kx=1,ky=1)

        inter_right_r = RectBivariateSpline(np.arange(self.rightImage.shape[0]),np.arange(self.rightImage.shape[1]), z=self.rightImage[:,:,0],kx=1,ky=1)
        inter_right_g = RectBivariateSpline(np.arange(self.rightImage.shape[0]),np.arange(self.rightImage.shape[1]), z=self.rightImage[:,:,1],kx=1,ky=1)
        inter_right_b = RectBivariateSpline(np.arange(self.rightImage.shape[0]),np.arange(self.rightImage.shape[1]), z=self.rightImage[:,:,2],kx=1,ky=1)

        empty_left = np.zeros(shape = self.leftImage.shape)
        empty_right = np.zeros(shape = self.rightImage.shape)

        self.middleTriangles = deepcopy(self.leftTriangles)
        for i in range(len(self.leftTriangles)):
            for j in range(len(self.leftTriangles[0].vertices)):
                for k in range(len(self.leftTriangles[0].vertices[0])):
                    self.middleTriangles[i].vertices[j][k] = ((1-alpha)*self.leftTriangles[i].vertices[j][k] + alpha*self.rightTriangles[i].vertices[j][k])
        #print(self.middleTriangles[183].vertices)
        #print(self.leftTriangles[183].vertices)
        #self.affineTransform(0)
        #for i in range()
        for index in range(len(self.middleTriangles)):
            invleftH, invrightH = self.affineTransform(index)
            points = self.middleTriangles[index].getPoints()
            for temp in points:
                middlexy = np.array([[temp[1]],[temp[0]],[1]],dtype = np.float64)
                leftpoints = np.matmul(invleftH,middlexy)
                rightpoints = np.matmul(invrightH,middlexy)
                x1 = int(temp[1])
                y1 = int(temp[0])
                empty_left[y1,x1,0] = inter_left_r.ev(leftpoints[1],leftpoints[0])
                empty_left[y1,x1,1] = inter_left_g.ev(leftpoints[1],leftpoints[0])
                empty_left[y1,x1,2] = inter_left_b.ev(leftpoints[1],leftpoints[0])

                empty_right[y1,x1,0] = inter_right_r.ev(rightpoints[1],rightpoints[0])
                empty_right[y1,x1,1] = inter_right_g.ev(rightpoints[1],rightpoints[0])
                empty_right[y1,x1,2] = inter_right_b.ev(rightpoints[1],rightpoints[0])

                #empty_left[x1,y1] = inter_left.ev(leftpoints[0],leftpoints[1])
                #empty_right[x1,y1] = inter_right.ev(rightpoints[0],rightpoints[1])

        blend = np.asarray(empty_left*(1-alpha)+empty_right*alpha,dtype = np.uint8)
        #print(blend)
        #imageio.imwrite("result.png",blend)
        #print(leftpoints.shape)
        return blend





if __name__ == "__main__":

    start_time = time.time()
    (leftTriangles,rightTriangles)=loadTriangles(DataPath+"/points.left.txt",DataPath+"/points.right.txt")
    #print(leftTriangles[4])
    leftgray = imageio.imread('LeftGray.png')
    rightgray = imageio.imread('RightGray.png')
    leftImage = np.array(leftgray,dtype=np.uint8)
    rightImage = np.array(rightgray,dtype=np.uint8)

    #leftcolor = imageio.imread('LeftColor.png')
    #rightcolor = imageio.imread('RightColor.png')
    #colorleftImage = np.array(leftcolor,dtype=np.uint8)
    #colorrightImage = np.array(rightcolor,dtype=np.uint8)

    #ColorMorpher(colorleftImage,leftTriangles,colorrightImage,rightTriangles).getImageAtAlpha(0.75)


    #print(leftTriangles[3].getPoints())
    Morpher(leftImage,leftTriangles,rightImage,rightTriangles).getImageAtAlpha(0.25)
    #Morpher(leftImage,leftTriangles,rightImage,rightTriangles).saveVideo('/home/ecegridfs/a/ee364g21/Desktop/targetFilePath',10,5,True)

    #print("Took", time.time() - start_time, "sec to run")
    #http://andrew.gibiansky.com/blog/image-processing/image-morphing/
