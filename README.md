# ISUPT_Pose


## Install 

Step1: Install ISUPT_Pose. Downloads zip from "Code" buttom.
Step2: Open Comment line(Terminal), and change Direct to ISUPT_Pose folder.Install requestion module using comment line like below.
```
pip install -r requestion.txt
```

## homework 
Gait-analysis use camera.


Step1: Record video about walking a line.
Step2: Upload the video to [openpose_colab](https://colab.research.google.com/drive/1D28I5-3-AGUIIvFcs7JsGMV0u2MQcWDT?usp=share_link) and run all.
Step3: Download keypoints json file from output_json folder. And download ISUPT_Pose from this github.
Step4: Copy keypoints json file to ISUPT_Pose\keypoints_json. And then open Comment line(Terminal) change Direct to ISUPT_Pose folder.
Step5: Run 01.theta.py using Comment line.
```
python 01.theta.py
```
Step6: Run 02.theta_plt.py using Comment line.
```
python 02.theta_plt.py
```
