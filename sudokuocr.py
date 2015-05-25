#!/usr/bin/env python2
import cv2
import numpy as np
import sys

matStr = ''

def getImageFromCamera():
	camera = cv2.VideoCapture(1)
	while True:
		img = camera.read()
		thresh = cv2.cvtColor(img[1],cv2.COLOR_BGR2GRAY)
		thresh = cv2.adaptiveThreshold(thresh,255,1,1,11,2)
		thresh = getBiggestContour(thresh)                    
		
		try:
			cv2.drawContours(img[1],[thresh.astype('int')],0,(0,255,0),3)
		except Exception:
			pass

		cv2.imshow("SudokuOCR",img[1])
		if (cv2.waitKey(5) != -1):           
			return [thresh,img[1]]

def getBiggestContour(image):
	biggest = None
	maxArea = 0
	contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		contour = contour.astype('int')
		area = cv2.contourArea(contour)
		if area > 100 and area > maxArea:
			peri = cv2.arcLength(contour,True)
			approx = cv2.approxPolyDP(contour, 0.02*peri, True)
			biggest = approx
			maxArea = area
	return biggest

def transformImage(contour, image):
	contour = contour.reshape((4,2))
	hnew = np.zeros((4,2),dtype = np.float32)
	add = contour.sum(1)
	hnew[0] = contour[np.argmin(add)]
	hnew[2] = contour[np.argmax(add)]
	diff = np.diff(contour,axis = 1)
	hnew[1] = contour[np.argmin(diff)]
	hnew[3] = contour[np.argmax(diff)]
	h = np.array([[0,0], [449,0], [449,449], [0,449] ], np.float32)
	retval = cv2.getPerspectiveTransform(hnew, h)
	image = cv2.warpPerspective(image, retval, (450,450))
	cv2.imshow("SudokuOCR", image)
	return image

def ocrRead():
	matrix = np.zeros((9,9), np.int8)
	imgRGB = cv2.imread('trans.jpg')
	imgGrayscale = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
	imgGrayscale = cv2.GaussianBlur(imgGrayscale,(5,5),1)
	for i in range(1,10):
		template = cv2.imread('digits/'+str(i)+'.png',0)
		w, h = template.shape[::-1]
		res = cv2.matchTemplate(imgGrayscale, template, cv2.TM_CCOEFF_NORMED)
		loc = np.where( res >= 0.8)
		for pt in zip(*loc[::-1]):
			matrix[pt[1]/50][pt[0]/50] = i
			cv2.rectangle(imgRGB, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
	cv2.putText(imgRGB, "Processing...", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
	cv2.imshow('SudokuOCR', imgRGB)
	#cv2.waitKey()
	return matrix

def sameRow(i,j): return (i/9 == j/9)
def sameCol(i,j): return (i-j) % 9 == 0
def sameBlock(i,j): return (i/27 == j/27 and i%9/3 == j%9/3)

def findSolution(a):
	i = a.find('0')
	if i == -1:
		showAnswer(a)

	exclNumbers = set()
	for j in range(81):
		if sameRow(i,j) or sameCol(i,j) or sameBlock(i,j):
			exclNumbers.add(a[j])

	for m in '123456789':
		if m not in exclNumbers:
			findSolution(a[:i]+m+a[i+1:])

def rectify(h):
	h = h.reshape((4,2))
	hnew = np.zeros((4,2),dtype = np.float32)
	add = h.sum(1)
	hnew[0] = h[np.argmin(add)]
	hnew[2] = h[np.argmax(add)]
	diff = np.diff(h,axis = 1)
	hnew[1] = h[np.argmin(diff)]
	hnew[3] = h[np.argmax(diff)]
	return hnew

def showAnswer(stt):
	wOverlay = np.zeros((480,640,3), np.uint8)
	overlay = np.zeros((480,640,3), np.uint8)
	for i in range(81):
		if matStr[i] == '0':
			cv2.putText(overlay, stt[i], ((i%9)*71 + 20 ,(i/9)*53 + 40), cv2.FONT_HERSHEY_PLAIN, 3.0, (0,255,0), 3)
	
	camera = cv2.VideoCapture(1)
	while True:
		img = camera.read()
		thresh = cv2.cvtColor(img[1], cv2.COLOR_BGR2GRAY)
		thresh = cv2.adaptiveThreshold(thresh, 255, 1, 1, 11, 2)    
		thresh = getBiggestContour(thresh)
		if(len(thresh) == 4):
			thresh = rectify(thresh)
			h = np.array([[0,0], [640,0], [640,480], [0,480]], np.float32)
			retval = cv2.getPerspectiveTransform(h, thresh)
			wOverlay = cv2.warpPerspective(overlay, retval, (640,480)) 
	   
		try:
			cv2.drawContours(img[1], [thresh.astype('int')], 0, (0,0,255), 3)
		except Exception:
			pass 

		cv2.imshow("SudokuOCR",cv2.addWeighted(img[1], 1, wOverlay, 1, 1))
		if (cv2.waitKey(5) != -1):            
			sys.exit()                   

def main():
	global matStr
	img = getImageFromCamera();
	cv2.imshow("SudokuOCR", img[1])
	img = transformImage(*img)

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img, (15,15), 1)
	cv2.imwrite("trans.jpg", cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2))
	cv2.waitKey()

	mat = ocrRead()

	matStr = ''.join(str(j) for item in mat for j in item)
	findSolution(matStr)

if __name__ == "__main__":
	main()