from torch.utils.data import Dataset
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.ssi_stream_utils import Stream
import torch 
class train_and_valDataset(Dataset):
    def __init__(self, data_path, id_list,modalities):
        self.data_list = []
        self.modalities = modalities 
        self.data_dir = data_path 
        self.partner_data_list = []
        self.labels_list = []
        self.data_label = [] 
        self.labels = [] # 标签
        self.data = []  #数据  
        #self.split_data_Label()
        self.process_data() 
        
        
#
#        for id in id_list:
#            cur_data_path = os.path.join(data_path,id)
#            cur_data_path_list = os.listdir(cur_data_path)
#
#            data_list_cur = []
#            for i in range(len(cur_data_path_list)):
#                if cur_data_path_list[i].split('.')[0] != 'label':
#                    data_list_cur.append(cur_data_path_list[i])
#            for i in range(len(data_list_cur)):
#                name = data_list_cur[i]
#                if 'expert' in name:
#                    partner_name = name.replace('expert','novice')
#                else:
#                    partner_name = name.replace('novice','expert')
#                if os.path.exists(os.path.join(cur_data_path,partner_name)):
#                    self.data_list.append(os.path.join(cur_data_path,name))
#                    #print(name.split('.'))
#                    self.labels_list.append(os.path.join(cur_data_path,f"label_{name.split('.')[-2]}_{name.split('.')[-1].split('.')[0]}.npy"))
#                    print(self.labels_list)
#                    self.partner_data_list.append(os.path.join(cur_data_path,partner_name))
#                else:
#                    continue 


    def lable_path(self,file_dir):
        anno = {}
        # 遍历训练目录以构建注释字典
        for path, sub_dirs, files in os.walk(file_dir):
            for f in files:
                session_id = os.path.basename(path)  #path=r'D:\Data\MPIIGI\008和其他文件夹'
                role = f.split('.')[0] #role就是每个人的特征的名字，如subjectPos1.audio.egemapsv2
                key = role + ";" + session_id #就是每个特征的关键字，用来简化路径，如：subjectPos1.audio.egemapsv2；008
                file_keywords = ".engagement.annotation" #文件的关键字，如：subjectPos1.engagement.annotation
                if file_keywords in f: #提取.stream~后缀的文件，然后加入train_anno中
                    anno[key] = os.path.join(path, f)  # 简化路径连接
        return anno
    def process_data(self):
        # 保存所有标签路径
        anno_dict = self.lable_path(self.data_dir)  #标签路径
        #print(anno_dict)
        # 使用 tqdm 包装迭代器，显示进度条
        aa=0
        for entry in tqdm(anno_dict, desc="Loading data"):
            # 创建一个特征字典
            if(aa==5):
                break
            aa+=1
            features_each_role = {}
            # values 用来存储 label 的值
            values_each_role = []
            # lengths 总长度不用改，只是用来记录每个特征的长度
            lengths = []
            # base_path 是特征文件的路径,不用改
            base_path = os.path.dirname(anno_dict[entry])

            # 对于每个 label 文件，找到其对应的特征文件，并且读取其特征：
            # 并且读取其标签
            role = entry.split(";")[0]
            for modality in self.modalities:
                # 为了支持读取 text 编码后的npy文件
                if modality.split('.')[-1] == 'npy':
                    modality_file = os.path.join(base_path, role + modality)
                    stream_f_data = np.load(modality_file)
                    features_each_role[modality] = stream_f_data
                    lengths.append(stream_f_data.shape[0])
                elif modality.split('.')[-1] == 'stream':
                    modality_file = os.path.join(base_path, role + modality)
                    stream_f = Stream().load(modality_file)
                    features_each_role[modality] = stream_f.data
                    lengths.append(stream_f.data.shape[0])
                
            # 读取 label 文件中的内容，并做相应处理(去 NaN)转化为 float:
            label_file = os.path.join(base_path, role + ".engagement.annotation.csv")
            with open(label_file, "r") as f:
                anno_file = np.genfromtxt(f, delimiter="\n", dtype=str)
            lengths.append(len(anno_file))
            # 这里考虑对齐了吗？
            num_samples = min(lengths)
            # values 用来存储 label 的值
            # 下一段有必要吗？ 数据中有nan？
            for label in anno_file:
                if label == '-nan(ind)':
                    values_each_role.append(float(0))
                else:
                    values_each_role.append(float(label))
            values_each_role = np.nan_to_num(values_each_role)

            # 将特征合并，并且与 label 做对齐：
            for i in range(num_samples):
                sample_each_role = np.concatenate([np.nan_to_num(features_each_role[modality][i]) for modality in self.modalities])
                label_each_role = np.array([values_each_role[i]])
                temp_data_label = np.concatenate((sample_each_role, label_each_role))
                self.data_label.append(temp_data_label)
            #print(np.array(self.data_label).shape)
            #feature_data = self.data_label[:] 
            #label_data = np.array(self.data_label[:, -1]).squeeze()
            # 保存
            #self.data.append(feature_data)
            #self.labels.append(label_data)
    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
       data_label_item = self.data_label[idx]
       data = torch.tensor(data_label_item[:-1], dtype=torch.float32)
       label = torch.tensor(data_label_item[-1:], dtype=torch.float32)  # or `label.item()` if scalar
       return data, label

        #return data,partner_data, label

class testDataset(Dataset):
    def __init__(self,data_path, id, name,flag=True):
        self.data_list = []
        self.labels_list = []
        self.partner_data_list = []
        self.flag = flag

        cur_data_path = os.path.join(data_path,id)
        cur_data_path_list = os.listdir(cur_data_path)
        data_list_cur = []
        
        for i in range(len(cur_data_path_list)):
            if cur_data_path_list[i].split('_')[2] == name:
                data_list_cur.append(cur_data_path_list[i])

        for i in range(int(len(data_list_cur))):
            self.data_list.append(os.path.join(cur_data_path,f'frame_feature_{name}_{i}.npy'))
            if name == 'expert':
                partner_name = f'frame_feature_novice_{i}.npy'
            else:
                partner_name = f'frame_feature_expert_{i}.npy'
            self.partner_data_list.append(os.path.join(cur_data_path,partner_name))
            if self.flag:
                self.labels_list.append(os.path.join(cur_data_path,f'label_{name}_{i}.npy'))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        partner_data = np.load(self.partner_data_list[idx])
        if self.flag:
            label = np.load(self.labels_list[idx])
            
            return data, partner_data,label
        else:
            return data, partner_data
    
