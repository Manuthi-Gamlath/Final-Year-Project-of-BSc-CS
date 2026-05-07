from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm
from src.ssi_stream_utils import Stream
import torch
import csv

class train_and_valDataset(Dataset):
    def __init__(self, annotation_path ,data_path,gt_path, id_list, modalities):
        self.modalities = modalities 
        self.anno_dir = annotation_path 
        self.data_GT=gt_path
        self.data_dir = data_path 
        self.data_label = []  # list of (list_of_modalities, label)
        self.process_data()

    def lable_path(self, file_dir):
        anno = {}
        for path, _, files in os.walk(file_dir):
            for f in files:
                session_id = os.path.basename(path)
                role = f.split('.')[0]
                if int(session_id) == 31:
                    continue
                if role == 'expert':
                    key = role + ";" + session_id
                    if ".engagement.annotation" in f:
                        anno[key] = os.path.join(path, f)
        return anno
    def load_stream_files(self,stream_folder):
        stream_data = {}
        for root, dirs, files in os.walk(stream_folder):
            for file in tqdm(files, desc=f"📥 Loading streams from {stream_folder}", leave=False):
                if file.endswith('.clip.stream'):
                    stream_path = os.path.join(root, file)
                    try:
                        data = torch.load(stream_path, map_location='cpu')
                        video_id = data.get('video_id', file)
                        clip_features = data.get('clip_features', None)
                        #print(file)
                        if clip_features is not None:
                            stream_data[file] = clip_features
                        else:
                            print(f"⚠️ No clip features found in {stream_path}")
    
                    except Exception as e:
                        print(f"❌ Failed to load {stream_path}: {e}")
    
        print(f"✅ Loaded {len(stream_data)} stream files from: {stream_folder}")
        #print(stream_data.keys())
        return stream_data

    def process_data(self):
        anno_dict = self.lable_path(self.anno_dir)
        #print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr',anno_dict)
        #lengths = []
        #label_file = "../ground_truth_outputs_hand.csv"
        #with open(label_file, "r") as f:
        #    anno_file = np.genfromtxt(f, delimiter="\n", dtype=str)
        #lengths.append(len(anno_file))
        #print(anno_file)
        output_csv_path="./ground_truth_outputs_final_reasontype2.csv"
        loaded_dic = {}
        with open(output_csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                loaded_dic[row[0]] = row[1]
        previous_video=''
        for aa, entry in enumerate(tqdm(anno_dict, desc="Loading data")):
            print(f"➡️ Processing session: {entry}")
            if aa == 1:  # Limit to 5 entries for debugging
               break
            aa+=1
            features_each_role = {}
            values_each_role = []
            

            base_path = os.path.dirname(anno_dict[entry])
            base_path_video=self.data_dir+base_path.split("/")[-1]+'/'
            base_path_video_GT=self.data_GT+base_path.split("/")[-1]+'/'
         
            
           # print("tttttttttttttt",temp)
            
            #if(previous_video != base_path_video):
            video_path_list =self.load_stream_files(base_path_video)
            #previous_video = base_path_video
            lenth_of_video = int((len(video_path_list.keys()))/2)
            stream_dic ={'novice':[],'expert':[]}
            for i in range(lenth_of_video):
                temp = os.path.join(base_path_video_GT.split("/")[5],base_path_video_GT.split("/")[6])
                expert_id = f"expert_video_{i}.clip.stream"
                novice_id = f"novice_video_{i}.clip.stream"
                expert_video_features = video_path_list[expert_id]
                novice_video_features = video_path_list[novice_id]
                expert_gt_id = f"expert_video_{i}.mp4"
                temp = os.path.join(temp,expert_gt_id)
                label_value= loaded_dic[temp]
                #print(expert_video_features.shape)
                #print(novice_video_features.shape)
                #print(label_value)
                self.data_label.append((expert_video_features,novice_video_features, label_value))
            #for video_file_path in video_path_list.keys():
            #    print(video_file_path)
            #    base_path_video_GT=base_path_video_GT+str(video_file_path.split(".")[0])+".mp4"
            #    role = ((video_file_path.split("_")[0]))
            #    video_segment_id=((video_file_path.split("_")[-1]).split(".")[0])
            #    temp = os.path.join(temp,str(video_file_path.split(".")[0])+".mp4")
            #    label_value= loaded_dic[temp]
            #    break
            #print(anno_file)
                #print(video_file_path,video_path_list[video_file_path].shape)
              #print(video_path_list.keys())
              #self.data_label.append(video_file_path)
            #role = entry.split(";")[0]
            
            ###print('ssssssssssss',base_path_video+ role)
            ### Load modalities
            #for modality in self.modalities:
            #    print(base_path, role + modality)
            #    modality_file = os.path.join(base_path, role + modality)
            #    if modality.endswith('npy'):
            #        data = np.load(modality_file)
            #    elif modality.endswith('stream'):
            #        data = Stream().load(modality_file).data
            #    else:
            #        continue
            #    features_each_role[modality] = np.nan_to_num(data)
            #    lengths.append(data.shape[0])
##
            # Load label file
            
#
            #num_samples = min(lengths)
#
            ## Clean label values
            #for label in anno_file:
            #    if label == '-nan(ind)':
            #        values_each_role.append(0.0)
            #    else:
            #        values_each_role.append(float(label))
            #values_each_role = np.nan_to_num(values_each_role)
#
            ## Store each sample
            #for i in range(num_samples):
            #    modality_list = [
            #        np.nan_to_num(features_each_role[modality][i])
            #        for modality in self.modalities
            #    ]
            #    label_value = values_each_role[i]
                

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        data = self.data_label[idx]
        #modality_tensors = [torch.tensor(m, dtype=torch.float32) for m in modality_list]
        #label = torch.tensor(label, dtype=torch.float32)
        return data


class testDataset(Dataset):
    def __init__(self, data_path, id, name, flag=True):
        self.data_list = []
        self.labels_list = []
        self.partner_data_list = []
        self.flag = flag

        cur_data_path = os.path.join(data_path, id)
        cur_data_path_list = os.listdir(cur_data_path)
        data_list_cur = [
            f for f in cur_data_path_list if f.split('_')[2] == name
        ]

        for i in range(len(data_list_cur)):
            self.data_list.append(os.path.join(cur_data_path, f'frame_feature_{name}_{i}.npy'))
            partner_name = f'frame_feature_{"novice" if name == "expert" else "expert"}_{i}.npy'
            self.partner_data_list.append(os.path.join(cur_data_path, partner_name))
            if self.flag:
                self.labels_list.append(os.path.join(cur_data_path, f'label_{name}_{i}.npy'))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        partner_data = np.load(self.partner_data_list[idx])
        if self.flag:
            label = np.load(self.labels_list[idx])
            return data, partner_data, label
        else:
            return data, partner_data
