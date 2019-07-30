from glob import glob
import SimpleITK as sitk
import numpy as np
import csv, time
from datetime import timedelta
from datetime import datetime
import pandas as pd
import skimage
from skimage.transform import rotate
import matplotlib.pyplot as plt

# where you put all .mhd files
Data_Dist = 'E:/Luna16_dataset/'
# where you put all out put files
prep_data_output_path = 'E:/Luna16/prepared_data/'
# where the annotations.csv located
Annotation_Dist = 'E:/Luna16/annonation_files/candidates_V2.csv'
new_annotation_path = 'E:/Luna16/annonation_files/new_annotation.csv'
Annot_output_path = 'E:/Luna16/annonation_files/'

width = 32 # patch size
dataset = []


def prep_new_annot_file(org_csv_file):
    csv_data = pd.read_csv(org_csv_file)

    pos = csv_data[csv_data['class'] == 1].index
    neg = csv_data[csv_data['class'] == 0].index
    pos = list(pos)
    pos = pos[:len(pos) - 7]

    negIdx = np.random.choice(neg, len(pos) * 6, replace=False)
    print(negIdx)
    negIdx = list(negIdx)

    csv_data = csv_data.iloc[pos + negIdx]
    print(csv_data)
    csv_data = csv_data.values.tolist()

    with open(Annot_output_path + 'new_annotation1.csv', "w", newline='') as csv_f:
        writer = csv.writer(csv_f, delimiter=',')
        writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'class'])
        for line in range(len(csv_data)):
            writer.writerows([csv_data[line]])
    return csv_data

def get_filename(file_list, case):      
    for f in file_list:
        if case in f:
            return(f)

def world_to_voxel(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return tuple(voxelCoord)
            
def extract_cubir_from_mhd(mhd_path,annatation_file):
    data = []
    labels = []
    patient_id = []
    counter = 0
    counter1 = 0
    rot_angle0 = 45
    rot_angle1 = 60

    #prep_new_annot_file(Annotation_Dist)
    file_list=glob(mhd_path+"*.mhd")
    # The locations of the nodes
    df_node = pd.read_csv(annatation_file)
    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    df_node = df_node.dropna()
    num_input = len(df_node)
    for img_file in file_list:
        mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
        file_name = str(img_file).split("/")[-1]
        if mini_df.shape[0]>0: # some files may not have a nodule--skipping those 
            # load the data once
            itk_img = sitk.ReadImage(img_file)
            img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
            #x,y,z
            origin = np.array(itk_img.GetOrigin())
            spacing = np.array(itk_img.GetSpacing())
            #print("origin: ", origin)
            #print("spacing: ", spacing)

            # go through all nodes 
            print("begin to process nodules...")
            for node_idx, row in mini_df.iterrows():
                P_ID = row["seriesuid"]
                node_x = row["coordX"]
                node_y = row["coordY"]
                node_z = row["coordZ"]
                label = row["class"]
                chk_label = label

                center = np.array([node_x, node_y, node_z])
                x,y,z = world_to_voxel(center, origin, spacing)

                x_start = int(x-(width/2))
                x_end = int(x+(width/2))
                y_start = int(y-(width/2))
                y_end = int(y+(width/2))
                z_dim = int(z)

                d,h,w = img_array.shape
                #print("img_array.shape: ", img_array.shape)
                
                if ((y_start and y_end < 0) or (y_start < 0) or (y_end < 0)) or ((x_start and x_end < 0) or (x_start < 0) or (x_end < 0)) or (y_start and y_end > h) or (x_start and x_end > w):
                    num_input -=1
                    counter1 +=1
                    print("not taking nodules: ", counter1)
                    continue
                else:
                    print("nodule is taking")
                
                temp_img = img_array[z_dim, y_start:y_end, x_start:x_end]

                n,p=temp_img.shape
                b1,b2,c1,c2=0,0,0,0
       
                if (n,p) != (width,width):
                    if n != width:
                        b=int(width-n)
                        if b%2!=0:
                            aa,bb = divmod(b,2)
                            b1=int(aa)
                            b2=int(aa+bb)
                        else:
                            b1=int(b/2)
                            b2=int(b-b1)

                    if p != width:
                        c=int(width-p)
                        if c%2!=0:
                            aa,bb = divmod(c,2)
                            c1=int(aa)
                            c2=int(aa+bb) 
                        else:
                            c1=int(c/2)
                            c2=int(c-c1)

                transform=((b1,b2),(c1,c2))
                temp_img = np.pad(temp_img,transform,'linear_ramp')

                # converting the label number into a one-hot-encoding
                if label == 1:
                    label = np.array([0, 1])
                elif label == 0:
                    label = np.array([1, 0])

                counter +=1
                print(num_input)
                print(counter)
                data=np.append(data,temp_img)
                labels=np.append(labels,label)
                patient_id = np.append(patient_id, P_ID)
   
                print("nodules %s from image %s extracted finished!..."%(node_idx,str(file_name)))

                if chk_label == 1:

                    num_input += 1
                    rot45 = skimage.transform.rotate(temp_img, angle=rot_angle0, mode='reflect')

                    data = np.append(data, rot45)
                    labels = np.append(labels, label)
                    patient_id = np.append(patient_id, P_ID)

                    #print('rotate it, 600 degree, and save again')
                    num_input += 1
                    rot60 = skimage.transform.rotate(temp_img, angle=rot_angle1, mode='reflect')

                    
                    data = np.append(data, rot60)
                    labels = np.append(labels, label)
                    patient_id = np.append(patient_id, P_ID)

    data = np.reshape(data,(num_input,width,width))
    labels = np.reshape(labels,(num_input, 2))
    print("not taking nodules: ", counter1)
    return data, labels, patient_id, num_input

start_time = time.time()
data, labels, patient_id, num_input = extract_cubir_from_mhd(Data_Dist, new_annotation_path)
print("number of input: ", num_input)
for i,j,k in zip(data, labels, patient_id):
    j = j.astype(np.int)
    dataset.append([i, j, k])
np.save(prep_data_output_path + 'patch_ds_2d_PID_{}_{}_{}_rotate_{}_{}_{}_{}_new.npy'.format(num_input, width, width, 45, 60, 90, 180), dataset)
print("%s time takes in seconds" % (time.time() - start_time))
time_dif = time.time() - start_time
print("Time used for 32x32: " + str(timedelta(seconds=int(round(time_dif)))))
res_file = open(prep_data_output_path+'Time_32x32.txt', 'w+')
res_file.write("Time used for 32x32: " + str(timedelta(seconds=int(round(time_dif)))) + "\n")
res_file.close()
