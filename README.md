
 
# Video human Counter  
 
This repository provides a solution for counting people in videos or live streams. It uses a hexagonal region-based counter: each time a person crosses a defined line, the system counts them and tracks how long they remain within the beyond 1 multiple regions. The result output then comapres result.


## Demo  
See `demo.gif`
[demo](demo.gif)
![demo](demo.gif)



## Setup 
1. Download `yolo12n.pt`
2.Search online how setup conda. 
Once you setup, create a conda environment with Python 3.11:
```
conda create --name ultralytics-env python=3.11 -y
conda activate ultralytics-env
conda install -c pytorch pytorch torchvision torchaudio

conda deactivate ultralytics-env
```

You may adjust the parameter in the code accordingly. Do read the ultralytics doc.
Run this way.
 


```
(ultralytics-env) python main_yolo_region.py --source video/human.mp4 --weights yolo12n.pt --view-img --save-img
```
 

## Video credit 
https://www.youtube.com/watch?v=Mp6klx9oeZs&pp=ygUlY29weXJpZ2h0IGZyZWUgZHJvbmUgdmlldyBjYXIgdHJhZmZpYw%3D%3D 


## Detail Doc   
1. Object counting - https://docs.ultralytics.com/guides/object-counting/#objectcounter-arguments
2. Object counting - https://docs.ultralytics.com/tasks/obb/#visual-samples
3. Region counting - https://docs.ultralytics.com/guides/region-counting/