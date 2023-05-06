from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.urls import reverse
from django.views.generic import View, TemplateView
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.models import User, auth
from sample.models import Contact
from django.views.decorators.cache import cache_control
from django.views.decorators.csrf import csrf_protect
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from scipy.spatial import distance as dist
import pyttsx3 
import numpy as np
import argparse
import imutils
import cv2
import os


# Create your views here.
@cache_control(no_cache=True,must_revalidate=True)
@csrf_protect
def home(request):
    if request.method=='POST':
        contact=Contact()
        name=request.POST.get('name')
        email=request.POST.get('email')
        subject=request.POST.get('subject')
        contact.name=name
        contact.email=email
        contact.subject=subject
        contact.save()
        return HttpResponse("<h1> Thanks for Contacting us </h1>")
        return render(reverse('sample:home')) 
    return render(request, 'home.html',{'name':'home'}) 
@cache_control(no_cache=True,must_revalidate=True)
def login(request):
    if request.method=='POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user=auth.authenticate(username=username,password=password)
        if user is not None:
            auth.login(request,user)
            return redirect('sample:header')
        else:
            messages.info(request,'invalid credentials')
            return redirect(reverse('sample:login'))
    else:
        return render(request,'login.html')
@cache_control(no_cache=True,must_revalidate=True)       
def register(request):
    if request.method=='POST':
        username=request.POST['username']
        email=request.POST['email']
        password1=request.POST['password1']
        password2=request.POST['password2']
        if password1==password2:
            if User.objects.filter(username=username).exists():
                messages.info(request,'Username Taken')
                return redirect(reverse('sample:register'))
            elif User.objects.filter(email=email).exists():
                messages.info(request,'Email Taken')
                return redirect(reverse('sample:register'))
            else:
                user=User.objects.create_user(username=username, password=password1, email=email)
                user.save();
                print('user created')
                return redirect('sample:login')
        else:
            messages.info(request,'password not matching...')
            return redirect(reverse('register.html'))
        return redirect('sample:home')
              
    else:
        return render(request, 'register.html') 
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url='/login/')
def logout(request):
    auth.logout(request)
    return redirect('/')
@cache_control(no_cache=True,must_revalidate=True)
def header(request):
    context={'a':'hello'}
    return render(request,'header.html',context)
@cache_control(no_cache=True,must_revalidate=True)
def counter(request):
    print(request)
    print(request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage( )
    filePathName=fs.save(fileObj.name,fileObj)
    
    filePathName=fs.url(filePathName)
    
    #web cam
    p='.'+filePathName
    # l=''
    # print(filePathName)
    x=(p.split('.')[-1])
    if x!='mp4':
        b='chosen file is not video type!!Choose a video file!!'
    else:
        b='video has choosen!!'
    MODLE_PATH="sample/yolo-coco"

    MIN_DISTANCE=50
    MIN_CONF=0.3
    NMS_THRESH=0.3

    USE_GPU=False
   
    def detect_people(frame, net, ln, personIdx=0):
            (H, W) = frame.shape[:2]
            results = []

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            boxes = []
            centroids = []
            confidences = []

            for output in layerOutputs:
        
                for detection in output:
            
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if classID == personIdx and confidence > MIN_CONF:
                
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
 
                
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
    
                        boxes.append([x, y, int(width), int(height)])
                        centroids.append((centerX, centerY))
                        confidences.append(float(confidence))
    
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
    
            if len(idxs) > 0:
        
                for i in idxs.flatten():
            
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                    results.append(r)

            return results
    
    labelsPath = os.path.sep.join([MODLE_PATH, "coco.names.txt"])
    #labelsPath = 'calc/yolo-coco/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")
    print(LABELS)
    print(len(LABELS))
    weightsPath = os.path.sep.join([MODLE_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([MODLE_PATH, "yolov3.cfg"])

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    if USE_GPU:
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture(p)
    videoStream = vs
    video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps =vs.get(cv2.CAP_PROP_FPS)
    print("FPS of the current video: ",fps)
    num_frames=vs.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Number of frames in the video:",num_frames )
    def initializeVideoWriter(video_width, video_height, videoStream):
    	# Getting the fps of the source video
        sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        outputVideoPath='E:/Output.mp4'
        return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
            (video_width, video_height), True)

    writer =  initializeVideoWriter(video_width, video_height, videoStream)

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,
            personIdx=LABELS.index("person"))

        violate = set()
        if len(results) >= 2:
    
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

        
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
            
                    if D[i, j] < MIN_DISTANCE:
                    
                        violate.add(i)
                        violate.add(j)
     
        for (i, (prob, bbox, centroid)) in enumerate(results):
        
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)
  
            if i in violate:
                color = (0, 0, 255)
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)
        
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        writer.write(frame) 
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        if key == ord("q"):
            break
        if violate:
            engine = pyttsx3.init()  
            text = "please maintain social distance"  
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
    
    writer.release()
    videoStream.release()

    cv2.destroyAllWindows()
    context={'filePathName':filePathName,'violations':len(violate)}
    return render(request,'header.html',context)
@cache_control(no_cache=True,must_revalidate=True)
def detection(request):
    print(request)
    print(request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage( )
    filePathName=fs.save(fileObj.name,fileObj)
    
    filePathName=fs.url(filePathName)
    
    #web cam
    p='.'+filePathName
    # l=''
    # print(filePathName)
    x=(p.split('.')[-1])
    if x!='mp4':
        b='chosen file is not video type!!Choose a video file!!'
    else:
        b='video has choosen!!'
    MODLE_PATH="sample/yolo-coco"

    MIN_DISTANCE=50
    MIN_CONF=0.3
    NMS_THRESH=0.3

    USE_GPU=False
   
    def detect_people(frame, net, ln, personIdx=0):
            (H, W) = frame.shape[:2]
            results = []

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            boxes = []
            centroids = []
            confidences = []

            for output in layerOutputs:
        
                for detection in output:
            
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if classID == personIdx and confidence > MIN_CONF:
                
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
 
                
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
    
                        boxes.append([x, y, int(width), int(height)])
                        centroids.append((centerX, centerY))
                        confidences.append(float(confidence))
    
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
    
            if len(idxs) > 0:
        
                for i in idxs.flatten():
            
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                    results.append(r)

            return results
    
    labelsPath = os.path.sep.join([MODLE_PATH, "coco.names.txt"])
    #labelsPath = 'calc/yolo-coco/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")
    print(LABELS)
    print(len(LABELS))
    weightsPath = os.path.sep.join([MODLE_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([MODLE_PATH, "yolov3.cfg"])

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    if USE_GPU:
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture(p)
    videoStream = vs
    video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps =vs.get(cv2.CAP_PROP_FPS)
    print("FPS of the current video: ",fps)
    num_frames=vs.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Number of frames in the video:",num_frames )
    def initializeVideoWriter(video_width, video_height, videoStream):
    	# Getting the fps of the source video
        sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        outputVideoPath='E:/Output.mp4'
        return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
            (video_width, video_height), True)

    writer =  initializeVideoWriter(video_width, video_height, videoStream)

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,
            personIdx=LABELS.index("person"))

        violate = set()
        if len(results) >= 2:
    
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

        
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
            
                    if D[i, j] < MIN_DISTANCE:
                    
                        violate.add(i)
                        violate.add(j)
     
        for (i, (prob, bbox, centroid)) in enumerate(results):
        
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)
        
        writer.write(frame) 
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        if key == ord("q"):
            break
    
    writer.release()
    videoStream.release()

    cv2.destroyAllWindows()
    context={'filePathName':filePathName,'violations':len(violate)}
    return render(request,'header.html',context)       
@cache_control(no_cache=True,must_revalidate=True)
def viewdatabase(request):
     import os
     listofVideos=os.listdir('C:/Users/LIKITHA KARI/OneDrive/Desktop/personal code/social_distancing/social_distancing/media/')
     listofVideos=['C:/Users/LIKITHA KARI/OneDrive/Desktop/personal code/social_distancing/social_distancing/media/'+i for i in listofVideos]
     context={'listofVideos':listofVideos}
     return render(request,'viewdatabase.html',context)
@cache_control(no_cache=True,must_revalidate=True)
def updatedatabase(request):
    return None
@cache_control(no_cache=True,must_revalidate=True)
def help(request):
     return render(request,'help.html')
@cache_control(no_cache=True,must_revalidate=True)
def aboutus(request):
    return render(request,'aboutus.html')