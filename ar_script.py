import math
import cv2
import numpy as np
import argparse
import os
import sys

#--function to draw the cuboid-----
def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2) #reshaping
    img = cv2.drawContours(img, [imgpts[:4]],-1,(255,0,0),3) #creating the base rectangle
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3) #creating the pillars
    img = cv2.drawContours(img, [imgpts[4:]],-1,(255,0,0),3)
    return img

if __name__=="__main__":
    parser=argparse.ArgumentParser()  #to accept data from the user
    parser.add_argument("-tar","--target",help="target location",default=os.getcwd()+"\\sample\\sample_image.jpg")
    parser.add_argument("-vid","--video",help="video location",default=os.getcwd()+"\\sample\\sample_video.avi")
    parser.add_argument("-cal","--calib_data",help="Calibration data",default=os.getcwd()+"\\sample\\sample_calib.npy")
    args=parser.parse_args()
    cap=cv2.VideoCapture(args.video) #declaring video variable
    im_tar=cv2.imread(args.target) #declaring target variable
    calib=np.load(args.calib_data,allow_pickle=True) #declaring calibration data (intrinsic and distortion)
    mtx,dist=calib[0],calib[1]
    objp=np.array([(0,0,0),(0,7,0),(4,7,0),(4,0,0)]) #Object point declaration
    axisbox=np.float32([[0,0,0],[0,7,0],[4,7,0],[4,0,0],[0,0,-3],[0,7,-3],[4,7,-3],[4,0,-3]]) #Box size location
    orb=cv2.ORB_create(nfeatures=1000) #initiation ORB detector 
    kp1,des1=orb.detectAndCompute(im_tar,None) # finding kepoints and descriptor from target image
    hT,wT,cT=im_tar.shape #getting a shape of the target image
    try:
        while True:
            _,imgw=cap.read() #reading the video frame by frame
            kp2,des2=orb.detectAndCompute(imgw,None) # finding kepoints and descriptor from each frame
            bg=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # initiating Brute Force Matcher
            match=bg.match(des1,des2) #checking the match by passing descriptor of two images
            match = sorted(match, key = lambda x:x.distance) # rearranging with respect to the distance
            g=match
            if len(g)>15:
                spts=np.float32([kp1[m.queryIdx].pt for m in g]).reshape(-1,1,2) #getting keypoints of video frame
                dispts=np.float32([kp2[m.trainIdx].pt for m in g]).reshape(-1,1,2) #getting keypoints of target image
                mat,mask=cv2.findHomography(spts,dispts,cv2.RANSAC,5) #finding the homography matrix
                pts=np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1,1,2) #declaring target image size for prepective
                ds= cv2.perspectiveTransform(pts,mat) #getting the bounding box corners
                imgpts=np.int32(ds).reshape(-1,2)
                ds=ds.reshape(-1,2)
                objp=np.array(objp,dtype="double")
                mtx=np.array(mtx,dtype="double")
                ret,rvecs,tvecs=cv2.solvePnP(objp,ds,mtx,dist,flags=cv2.SOLVEPNP_ITERATIVE) #finding the location of the target in video frame
                imgptsf,jac=cv2.projectPoints(axisbox,rvecs,tvecs,mtx,dist) #finding the cuboid points in the camera space
                imgptsf=imgptsf.reshape(-1,2)
                print("x:  "+str(tvecs[0][0])+" "+"y:  "+str(tvecs[1][0])+" "+"z:  "+str(tvecs[2][0]))
                print("rx:  "+str(rvecs[0][0])+" "+"ry:  "+str(rvecs[1][0])+" "+"rz:  "+str(rvecs[2][0]))
                imgw=draw(imgw,ds,imgptsf) #passing to a function to draw the cuboid
            cv2.imshow("result",imgw) #visualizing the result
            if cv2.waitKey(1) & 0xFF == ord('q'):
                     break
    except:
        pass
cv2.destroyAllWindows() #destroying window after completion


