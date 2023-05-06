import social_distancing_config as config
from detection import detect_people
from scipy.spatial import distance as dist
import pyttsx3 
import numpy as np
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
    help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
    help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
    help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([config.MODLE_PATH, "coco.names.txt"])
LABELS = open(labelsPath).read().strip().split("\n")
print(LABELS)
print(len(LABELS))
weightsPath = os.path.sep.join([config.MODLE_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODLE_PATH, "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if config.USE_GPU:
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
fps =vs.get(cv2.CAP_PROP_FPS)
print("FPS of the current video: ",fps)
num_frames=vs.get(cv2.CAP_PROP_FRAME_COUNT)
print("Number of frames in the video:",num_frames )
writer = None

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
            
                if D[i, j] < config.MIN_DISTANCE:
                    
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

    if args["display"] > 0:
    
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
    
    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25,
            (frame.shape[1], frame.shape[0]), True)
           
    if writer is not None:
        writer.write(frame)   
    if violate:
        engine = pyttsx3.init()  
           
        text = "please maintain social distance"  
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
        

# USAGE
# python social_distancing.py --input pedestrians.mp4
# python social_distancing.py --input pedestrians.mp4 --output output.avi