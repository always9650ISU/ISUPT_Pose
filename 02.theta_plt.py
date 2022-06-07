from queue import Full
import matplotlib.pyplot as plt
import json 
import os 
import pandas as pd 
import numpy as np 

jsonDir = './theta_json'
pltDir = './plt'

RollingWindowSize = 6


for FullName in os.listdir(jsonDir):
    print(FullName) 
    filename, _ = FullName.split('.')
    filepath = os.path.join(jsonDir, FullName)
    output = os.path.join(pltDir, filename + '.jpg')
    with open(filepath, 'r') as f: 
        data = json.load(f)

    if extension == 'gitignore':
        continue
    
    data_length = len(data)
    data_time = [i for i in range(data_length)]
    RHip_theta = []
    LHip_theta = []
    RKnee_theta = []
    LKnee_theta = []
    RAnkle_theta = []
    LAnkle_theta = []

    for i in data_time:
        RHip_theta.append(data[str(i)]["RHip_theta"])
        LHip_theta.append(data[str(i)]["LHip_theta"])
        RKnee_theta.append(data[str(i)]["RKnee_theta"])
        LKnee_theta.append(data[str(i)]["LKnee_theta"])
        RAnkle_theta.append(data[str(i)]["RAnkle_theta"])
        LAnkle_theta.append(data[str(i)]["LAnkle_theta"])

    df = pd.DataFrame(data= [RHip_theta, LHip_theta, RKnee_theta, LKnee_theta, RAnkle_theta, LAnkle_theta],
                index=['RHip_theta','LHip_theta','RKnee_theta','LKnee_theta','RAnkle_theta','LAnkle_theta']
                ).T
    df['RHip_theta'] = pd.to_numeric(df['RHip_theta'], errors='coerce')
    df['LHip_theta'] = pd.to_numeric(df['LHip_theta'], errors='coerce')
    df['RKnee_theta'] = pd.to_numeric(df['RKnee_theta'], errors='coerce')
    df['LKnee_theta'] = pd.to_numeric(df['LKnee_theta'], errors='coerce')
    df['RAnkle_theta'] = pd.to_numeric(df['RAnkle_theta'], errors='coerce')
    df['LAnkle_theta'] = pd.to_numeric(df['LAnkle_theta'], errors='coerce')
    df = df.interpolate()
    df = df.astype(np.int16)

    fig, ((ax1, ax2), (ax3,  ax4), (ax5,  ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))

    ax1.plot(data_time, df["LHip_theta"]) 
    ax1.set_title("LHip_theta")
    ax2.plot(data_time, df["LHip_theta"].rolling(RollingWindowSize).mean()) 
    ax2.set_title("LHip_theta")
    ax3.plot(data_time, df["LKnee_theta"])
    ax3.set_title("LKnee_theta")
    ax4.plot(data_time, df["LKnee_theta"].rolling(RollingWindowSize).mean())
    ax4.set_title("LKnee_theta")
    ax5.plot(data_time, df["LAnkle_theta"])
    ax5.set_title("LAnkle_theta")
    ax6.plot(data_time, df["LAnkle_theta"].rolling(RollingWindowSize).mean())
    ax6.set_title("LAnkle_theta")


    fig.tight_layout()

    plt.suptitle(f'Various Straight Lines')  
    plt.savefig(output)


'''
    RHip_theta
    LHip_theta
    RKnee_theta
    LKnee_theta
    RAnkle_theta
    LAnkle_theta
'''