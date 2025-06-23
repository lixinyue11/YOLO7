# YOLO
## 物体检测train运行命令 
--workers 1 --device 0 --batch-size 4 --data data/neu.yaml --cfg cfg/training/yolov7.yaml --weights yolov7.pt --name yolov7 --hyp data/hyp.scratch.p5.yaml --epochs 20 
## detect运行命令：
--weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
