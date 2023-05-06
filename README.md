# Social_Distancing_Detection_YOLOV3


One way of limiting the spread of an infectious disease, for instance, Covid-19, is to practice social distancing.
Social distancing aims to decrease or interrupt transmission of COVID-19 in a population by minimizing contact between potentially infected individuals and healthy individuals, or between population groups with high rates of transmission and population groups with no or low levels of transmission.

# How It Works

By using pre-trained video as input and the open-source object detection pretrained model by YOLOv3 algorithm which is pre-trained COCO dataset, we will detect the people in the frame. The distance between people can be estimated and any noncompliant pair of people in the display will be indicated with a red bounded box. We can tell if people are following social distancing or not and based on that we are creating red or green bounding boxes over it. The result shows that the proposed method is able to determine the social distancing measures between multiple people in the video . And it counts the number of persons who violates the social distancing measures and mark with red bounded box and produces a voice alert message as "PLEASE MAINTAIN SOCIAL DETECTION..!"

