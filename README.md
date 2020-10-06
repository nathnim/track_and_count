# Project objective

The aim of this project is to count the number of unique people voting in a room during the day.
The project is devided into several steps:

1. Custom object (an urn) detection
2. People tracking
3. Counting unique people

Each of these tasks is a separate topic itself and could be generalized further to be used for other purposes.
For instance, tasks 2 and 3 could be employed to count the number of unique customers in a shop etc.  

## Custom object detection 

The implementation of custom object detection could be found in a folder urn_detection_yolov5. 
First, the dataset of urn pictures was collected (see urn_detection_yolov5/collecting_urn_dataset.doc
for details). Note that the dataset was already augmented with different brightness levels to simulate the 
effect of illumination in a room or bad camera settings. The dataset is downloaded with curl.
Then, YOLOv5 detector is applied with 2 classes of objects specified: an urn (custom object) 
and a person (coco object). The neural network is then fine tuned to learn about the custom 
object class. Finaly, the inference is done on a subset of data and the result is visualized.     

NB Since an urn is a stationary object (i.e. it position is not supposed to change in time),
its dectection can be done on a single (initial) video frame. Then, the urn coordinares could
be passed further to other frames without performing the detection task over and over again. 

## Track and count

In the second part we track people in a room using the tracking-by-detection paradigm.
As it has been done in the custom object detection section above, YOLOv5 performs a person
detection on each video frame. Then, the detections on different frames must be associated 
between each other to reidentify the same person. We employ the SORT tracker which combines
the Kalman filter to predict the state of the object and the Hungarian algorithm to associate
the objects. 

![Gif example](https://github.com/maxmarkov/track_and_count/blob/master/example/tracker_example.gif)

**Content**

- detect_and_track_yolov5_sort.py is the implementation of detect+track task
- run_tracker_on_colab.ipynb shows how to run the tracker on google colab. 
- folder 'theory' contains the slides with summary of theoretical approaches  

## TO DO:

- Deep SORT+YOLOv5
- Create a set of bbox snapshots for each ID
- **Counter**: NN to identify the number of unique people from the dataset

## Literature

- [Simple Online and Realtime Tracking (SORT)](https://arxiv.org/abs/1602.00763)
- [Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT)](https://arxiv.org/pdf/1703.07402.pdf)
- [Real-Time Multiple Object Tracking: A Study on the Importance of Speed by S.Murray](https://arxiv.org/pdf/1709.03572.pdf)
- [Real time multiple camera person detection and tracking](https://repositorio.iscte-iul.pt/handle/10071/17743)
- [Kalman and Bayesian Filters in Python (pdf)](https://elec3004.uqcloud.net/2015/tutes/Kalman_and_Bayesian_Filters_in_Python.pdf)
- [Kalman and Bayesian Filters in Python (codes)](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
- [Deep Cosine Metric Learning for Person Re-Identification](https://elib.dlr.de/116408/1/WACV2018.pdf)

## Codes

- [SORT](https://github.com/abewley/sort)
- [Deep SORT (TF)](https://github.com/nwojke/deep_sort)
- [Deep SORT (PyTorch)](https://github.com/ZQPei/deep_sort_pytorch)
- [YOLOv5+DeepSORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
- [YOLOv4](https://github.com/AlexeyAB/darknet)
- [YOLOv4 PyTorch](https://github.com/Tianxiaomo/pytorch-YOLOv4)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [FilterPy library (the Kalman filter)](https://filterpy.readthedocs.io/en/latest/)

## Habr

- [Как работает Object Tracking на YOLO и DeepSort](https://habr.com/en/post/514450/)
- [Самая сложная задача в Computer Vision](https://habr.com/en/company/recognitor/blog/505694/) 
