# Track and count

Collection of scripts to solve the detection-tracking-counting problem. 
The algorithm makes use of YOLOv5 for detection, of SORT for tracking and
of the DNN to count the number of unique people from the set of snapshots
(the latter is not currently implemented).

## Content

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
