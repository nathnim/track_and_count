# Project objective and description

The aim of this project is to count the number of unique people voting in a room during the election day.
The project is devided into several steps:

1. Custom object (an urn) detection
2. People tracking
3. Counting unique voters
4. Reidentification 

Each of these tasks is a separate topic itself and could be generalized further to be used for other purposes. For instance, **tasks 2, 3 and 4** could be employed to count the number of unique customers in a shop *etc*.  

![Gif example](https://github.com/maxmarkov/track_and_count/blob/master/example/example_count.gif)

Table of contents
=================
- [Custom object detection](#custom-detection)
- [Tracking](#tracking)
- [Count](#count)
- [Reidentification](#reid)
- [How to run the trackers](#run-tracker)
- [How to detect urns](#detect-urn)
- [Literature](#lit)
- [Codes](#codes)

<a name="custom-detection"></a>
## Custom object detection 

The implementation of custom object detection could be found in a folder *urn_detection_yolov5*. 
First, the dataset of urn pictures was collected (see *urn_detection_yolov5/collecting_urn_dataset.doc*
for details). Note that the dataset has already been augmented with different brightness levels to simulate the 
effect of illumination in a room and/or bad camera settings. The dataset can be downloaded with curl.
Then, the **YOLOv5 detector** is applied with 2 classes of objects specified: an urn (a custom object) 
and a person (a coco object). The neural network is then fine tuned to learn about the custom 
object class. Finaly, the inference is done on a subset of data and the result is visualized. 

**Example of urn detection with YOLOv5**

<img src="example/urn_detection_inference.jpeg" width="400" class="centerImage">

*NB*: Since an urn is a stationary object (i.e. its position is not supposed to change in time),
the dectection can be performed on a single (initial) video frame. Then, the urn's coordinares could
be easily passed further to other frames without performing the detection task over and over again. 


<a name="tracking"></a>
## Tracking

In the second part of the project we track people in a room using the tracking-by-detection paradigm.
As it has been done earlier in the custom object detection section, **YOLOv5** performs a person
detection on each single video frame. Then, the detections on different frames must be associated 
between each other to re-identify the same person. **The SORT tracker** combines the linear Kalman filter
to predict the state of the object (*the motion model*) and the Hungarian algorithm to associate objects 
from the previous frames with objects in the current frame. The tracker does not take into account any details
of the object's appearence. My implementation of the SORT tracker inside the YOLOv5 inference script could be found in 
*track_yolov5_sort.py*. The Jupyter notebook *colabs/run_sort_tracker_on_colab.ipynb* shows how to run the
tracker on **Google Colab**.

**Example of tracking in a room using SORT and YOLOv5**

![Gif example](https://github.com/maxmarkov/track_and_count/blob/master/example/tracker_example.gif)

A nice alternative to the SORT tracker is a [Deep SORT](https://arxiv.org/pdf/1703.07402.pdf).
**The Deep SORT** extends the SORT tracker adding a deep association metric to build an appearance
model in addition to the motion model. According to the authors, this extension enables to track objects
through longer periods of occlusions, effectively reducing the number of identity switches. My implemention
of the tracker inside the YOLOv5 inference script could be found in *track_yolov5_deepsort.py*. The Jupyter
notebook *colabs/run_deepsort_tracker_on_colab.ipynb* shows how to run the tracker on **Google Colab**.

<a name="count"></a>
## Count

Since our primary task is to count the number of unique voters but not the total number of people in a room (some people like kids
often just accomany their parents who vote), it is important to define the voting act in a more precise way. Both an urn and voters are
identified using the YOLOv5 detector which puts a bounding box around each of them. To vote, a person must come close to an urn and
spend a certain amount of time around (i.e. the distance between the object centroids must be within a certain critical radius). This
"certain amount of time" is necessary to distinguish the people who pass by and the ones who actually vote. This approach requires two
predefined **parameters**:

- Critical radius
- Minimum interaction time

The person whose motion satisfies the conditions defined above can be then tracked until he/she dissapears from the camera view. The
tracking is necessary in case the person stays in a room hanging around for a while. To further insure that we count the unique people only,
one can save an image of each tracked person inside the bound box building a database of voters in a video. When the dateset of images with
voters is built, one can run a neural network to find the unique voters based on their appearance similarity.

<a name="reid"></a>
## Reidentification

Both trackers listed above possess only a short-term memory. The object's track is erased from memory after max_age number of frames
without associated detections. Typically max_age is around 10-100 frames. If a person leaves a room and comes back in a while, the
tracker will not re-identify the person assigning a new ID instead. To solve this issue, one needs a long-term memory. Here we implement
long-term memory by means of appearance features from the Deep Sort algorithm. Appearance feature vector is a 1D array with 512 components. 
For each track ID we create a separate folder into which we write feature vectors. Feature vectors files are labeled in their names with 
frame number index where the object has been detected. When a new track is identified, one can compute the cosine distance between this 
track and all saved tracks in appearance space. If the distance is smaller than some threshold value, an old ID could be reassigned to 
a new object. Long-term memory enables us to exclude the security guards or the election board members who approach an urn frequently.

Feature extractor script is *deepsort_features.py*. Besides the standard output video file, it also writes features and corresponding croped
images of tracked objects being saved into inference/features and inference/image_crops folders respectively. The log file with dictionary 
storing the history of object detections is in inference/features/log_detection.txt. Keys of this dictionary are track IDs
and values are lists with frame numbers where the corresponding track has been registered. Moreover, we save frames per second rate which enables
us to restore the time (instead of frame number) when the track is detected.

**Content:**

- track_yolov5_sort.py implements the SORT tracker in YOLOv5
- track_yolov5_deepsort.py implements the Deep SORT tracker in YOLOv5
- colabs/run_sort_tracker_on_colab.ipynb and colabs/run_deepsort_tracker_on_colab.ipynb shows how to run the trackers on google colab. 
- track_yolov5_counter.py runs a counter
- deepsort_features.py implements the feature extractor
- folder 'theory' contains the slides with summary of theoretical approaches  

<a name="detect-urn"></a>
## How to detect urns.

1. Extract some snapshot frames into snapshot_frames folder

     python3 utils/extract_frames.py --source video_examples/election_2018_sample_1.mp4 --destination snapshot_frames --start 1 --end 10000 --step 1000

2. Run the detector which saves the coordinates into .txt file in urn_coordinates folder

     python3 yolov5/detect.py --weights urn_detection_yolov5/weights_best_urn.pt --img 416 --conf 0.2 --source snapshot_frames --output urn_coordinates --save-txt

<a name="run-tracker"></a>
## How to run the trackers

1. Follow the installation steps described in INSTALL.md

2. Run tracker: YOLOv5 + (SORT or Deep SORT)

     python3 track_yolov5_sort.py --source example/running.mp4 --weights yolov5/weights/yolov5s.pt --conf 0.4 --max_age 50 --min_hits 10 --iou_threshold 0.3

     python3 track_yolov5_deepsort.py --source example/running.mp4 --weights yolov5/weights/yolov5s.pt

3. Run tracker with pose estimator

     python3 track_yolov5_pose.py --source example/running.mp4 --weights yolov5/weights/yolov5s.pt

4. Run the counter

     python3 track_yolov5_counter.py --source video_examples/election_2018_sample_1.mp4 --weights yolov5/weights/yolov5s.pt

5. Run the feature extractor

     python3 deepsort_features.py --source example/running.mp4 --weights yolov5/weights/yolov5s.pt


<a name="lit"></a>
## Literature

- [Simple Online and Realtime Tracking (SORT)](https://arxiv.org/abs/1602.00763)
- [Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT)](https://arxiv.org/pdf/1703.07402.pdf)
- [Real-Time Multiple Object Tracking: A Study on the Importance of Speed by S.Murray](https://arxiv.org/pdf/1709.03572.pdf)
- [Real time multiple camera person detection and tracking by D.Baikova](https://repositorio.iscte-iul.pt/handle/10071/17743)
- [Detection-based Multi-Object Trackingin Presence of Unreliable Appearance Features by A.Kumar(UCL)](https://sites.uclouvain.be/ispgroup/uploads//Main/PHDAKC_thesis.pdf)
- [Slides on "Re-identification for multi-person tracking" by V. Sommers (UCL)](https://sites.uclouvain.be/ispgroup/uploads//ISPS/ABS220720_slides.pdf)
- [Kalman and Bayesian Filters in Python (pdf)](https://elec3004.uqcloud.net/2015/tutes/Kalman_and_Bayesian_Filters_in_Python.pdf)
- [Kalman and Bayesian Filters in Python (codes)](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
- [Deep Cosine Metric Learning for Person Re-Identification](https://elib.dlr.de/116408/1/WACV2018.pdf)

<a name="codes"></a>
## Codes

- [SORT](https://github.com/abewley/sort)
- [Deep SORT (TF)](https://github.com/nwojke/deep_sort), [Deep SORT (PyTorch)](https://github.com/ZQPei/deep_sort_pytorch)
- [YOLOv5+DeepSORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
- [Deep person reid (UCL)](https://github.com/VlSomers/deep-person-reid)
- [YOLOv4](https://github.com/AlexeyAB/darknet), [YOLOv4 PyTorch](https://github.com/Tianxiaomo/pytorch-YOLOv4)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [FilterPy library (the Kalman filter)](https://filterpy.readthedocs.io/en/latest/)

## Habr

- [Как работает Object Tracking на YOLO и DeepSort](https://habr.com/en/post/514450/)
- [Самая сложная задача в Computer Vision](https://habr.com/en/company/recognitor/blog/505694/) 
