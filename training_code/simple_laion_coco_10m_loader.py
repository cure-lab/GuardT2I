from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import os
import copy
# from datasets import Dataset

# Note that this file creates GuardT2I dataset from intermediate data files. Not necessary for training. Users can refer this file to create their own dataset. 



def get_loader(train, clip_backbone="CLIP-vit-large",sample_start_index=0, batch_size=52, sample_slot=4000000, dataset_path = "../datasets/"):
  if train:
    split = "train"
  else:
    split = "val"
    
  if train:
    processed_data_path = dataset_path + "full_laion_coco_clip_features_train.npy"
  else:
    processed_data_path = dataset_path + "full_laion_coco_clip_features_val.npy"
  sample_end_index = (sample_start_index+1)*sample_slot
  if os.path.isfile(processed_data_path):
    with open(processed_data_path, 'rb') as e:
      clip_features = np.load(e, allow_pickle=True)[sample_start_index:sample_end_index]
  
  if train:
        tokenized = torch.load("bert_tokenized_train.pt",weights_only=False)
  else:
        tokenized = torch.load("bert_tokenized_val.pt",weights_only=False)
    
  if train:
      label_ids = copy.deepcopy(tokenized['input_ids'])[sample_start_index:sample_end_index]
      label_ids[label_ids == 0] = -100
      input_ids = tokenized['input_ids'][sample_start_index:sample_end_index]
      attention_mask = tokenized['attention_mask'][sample_start_index:sample_end_index]
  else:
      label_ids = copy.deepcopy(tokenized['input_ids'])
      label_ids[label_ids == 0] = -100
      input_ids = tokenized['input_ids']
      attention_mask = tokenized['attention_mask']
  input_ids = torch.tensor(input_ids, dtype=torch.long)
  attention_mask = torch.tensor(attention_mask, dtype=torch.long)
  label_ids = torch.tensor(label_ids, dtype=torch.long)
  clip_features = torch.tensor(clip_features, dtype=torch.long)
  print(input_ids.size(), attention_mask.size(), label_ids.size(), clip_features.size())
  hidden_size = clip_features.size(1)
  print(clip_features.repeat(1,1).view(-1, hidden_size).size())
  dataset = TensorDataset(input_ids, attention_mask, label_ids, clip_features.repeat(1,1).view(-1, hidden_size))
  loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, shuffle=True)
  # import ipdb; ipdb.set_trace()
  torch.save({
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'label_ids': label_ids,
    'clip_features': clip_features,
  }, 'GuardT2I_dataset_train_'+str(sample_end_index-1)+'.pt')
  # data_dict = {
  #   "input_ids": input_ids.numpy(),
  #   "attention_mask": attention_mask.numpy(),
  #   "label_ids": label_ids.numpy(),
  #   "clip_features": clip_features.repeat(1,1).view(-1, hidden_size).numpy(),
  # }
  # # Create a Hugging Face Dataset
  # hf_dataset = Dataset.from_dict(data_dict)
  # hf_dataset.save_to_disk('GuardT2I_train_dataset_hf_1')
  return loader   


if __name__=='__main__':

      # if i == 0: continue
    # dataset = torch.load("/bian_data/NeurIPS2024_release/datasets/GuardT2I_dataset_train_part1_5M.pt")
    # dataloader = DataLoader(dataset=dataset, batch_size=52, num_workers=4, shuffle=True)
    # get_loader(train=True, sample_start_index=0)
    # import ipdb; ipdb.set_trace()
    
    get_loader(train=True, sample_start_index=0)
    
    # import ipdb; ipdb.set_trace()
