# http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# http://python.jobbole.com/81593/
# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
#import cv2.cv as cv
import numpy as np

# 获取参数
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=10000, help="minimum area size") #容忍度
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)

# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])


# 定义第一帧
# initialize the first frame in the video stream
firstFrame = None

# Define the codec
#cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是YUV编码类型，文件名后缀为.avi

#cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'),该参数是MPEG-1编码类型，文件名后缀为.avi

#cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),该参数是MPEG-4编码类型，文件名后缀为.avi

#cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'),该参数是Ogg Vorbis,文件名后缀为.ogv

#cv2.VideoWriter_fourcc('F', 'L', 'V', '1'),该参数是Flash视频，文件名后缀为.flv
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
framecount = 0
frame = np.zeros((640,480)) # 【640，480】：0
out = cv2.VideoWriter('./videos/'+'calm_down_video_'+datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.avi',fourcc, 5.0, np.shape(frame))

# to begin with, the light is not stable, calm it down
tc = 40
while tc:
    ret, frame = camera.read()
    out.write(frame)
    #cv2.imshow("vw",frame)
    cv2.waitKey(10) # wait 10MS
    tc -= 1
totalc = 2000
tc = totalc
out.release()

# loop over the frames of the video
while True:

    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()
    text = "Unoccupied"

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        time.sleep(0.25)
        continue

    # resize the frame, convert it to grayscale, and blur it
    # 将大小缩放至想要的大小，减轻系统压力
    frame = imutils.resize(frame, width=500)

    # 将图像转换为灰度，因为颜色与我们的运动检测算法无关
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #对21 x 21区域的平均像素强度应用高斯平滑 。这有助于消除可能使我们的运动检测算法无法正常运行的高频噪声
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # 这里是每totalc帧对fristFrame进行更新
    # update firstFrame for every while
    if tc%totalc == 0:
        firstFrame = gray
        tc = (tc+1) % totalc
        continue
    else:
        tc = (tc+1) % totalc

    #print tc

    # compute the absolute difference between the current frame and
    # first frame

    #计算两个帧之间的差异是一个简单的减法，其中我们取其对应像素强度差异的绝对值：
    # delta = | background_model – current_frame |
    frameDelta = cv2.absdiff(firstFrame, gray)
    # 如果增量小于25，我们将丢弃像素并将其设置为黑色（即背景）。如果增量大于25，则将其设置为白色（即前景）
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    _, cnts,_= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 使用会报错误：Contours tuple must have length 2 or 3, otherwise OpenCV changed their cv2.findContours return signature yet again. Refer to OpenCV's documentation in that case
    # cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:

    # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow("222",frame[y:y+h,x:x+w])
        text = "Occupied"

    # 这里主要是在视频的角写文字
    # draw the text and timestamp on the frame
    cv2.putText(frame, "Monitoring Area Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)


    # save the detection result
    print(text)
    if text == "Occupied":
        if framecount == 0:
            # create VideoWriter object
            out = cv2.VideoWriter('./videos/'+datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.avi',fourcc, 10.0, np.shape(gray)[::-1])
            cv2.imwrite('./images/'+datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.jpg',frame)
            print('cv2.imwrite1')
            # write the flipped frame
            out.write(frame)
            framecount += 1
        else:
            # write the flipped frame
            out.write(frame)
            if framecount%10 == 0:
                print('cv2.imwrite2')
                cv2.imwrite('./images/'+datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.jpg',frame)
                print(type(frame))
            framecount += 1
    elif framecount < 30 and framecount>1:
        # write the flipped frame
        out.write(frame)
        #if framecount%10 == 0:
                #cv2.imwrite('./images/'+datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.jpg',frame)
        framecount += 1
    else:
        out.release()
        framecount = 0        

    key = cv2.waitKey(1) & 0xFF

    # if the `ESC` key is pressed, break from the lop
    if key == 27:
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
