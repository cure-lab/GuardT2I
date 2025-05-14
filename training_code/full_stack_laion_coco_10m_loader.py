from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm
from transformers import BertGenerationTokenizer
import copy


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionInpaintPipeline
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
import warnings
from diffusers import StableDiffusionInpaintPipeline
from torch.utils.data import Dataset

warnings.simplefilter("ignore")

# GuardT2I Dataset Builder

# This script creates the GuardT2I training dataset from raw LAION-COCO files, walking you through each step of our data-processing pipeline. It is provided purely as a reference to help you understand and customize dataset creation; it is **not** required for model training if you already have preprocessed data available.

# **Prerequisite:** Download and extract the LAION-COCO dataset before running this script.



class My_laion_coco_dataset(Dataset):
    def __init__(self, train=True):
        if train:
            data_path = "/u01/yangyijun/data/datasets/laion-coco-caption-first20m.pt"
        else:
            data_path = "/u01/yangyijun/data/datasets/laion-coco-caption-testset-last100000.pt"
        
        super(My_laion_coco_dataset, self).__init__()
        self.data = torch.load(data_path)
    
    def __len__(self):
        return(self.data.__len__())
    
    def __getitem__(self, idx):
        return self.data[idx]
    

def get_clip_text_features(laion_coco_dataset, split, clip_backbone="CLIP-vit-large", device="cuda"):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "/bian_data/safe_diffusion/dataloaders/stable-diffusion-inpainting",
  ).to(device)
  clip_model = pipe_inpaint.text_encoder 
  clip_tokenizer = pipe_inpaint.tokenizer 
  if split == "train":
    processed_data_path = "/bian_data/NeurIPS2024_release/datasets/full_laion_coco_clip_features_train.npy"
  else:
    processed_data_path = "/bian_data/NeurIPS2024_release/datasets/full_laion_coco_clip_features_val.npy"
  if os.path.isfile(processed_data_path):
    with open(processed_data_path, 'rb') as e:
      clip_out_all = np.load(e, allow_pickle=True)
  else:
    print('calculating all clip text encoder embeddings')
    loader = DataLoader(dataset=laion_coco_dataset, batch_size=1000, shuffle=False)
    clip_out_all = []
    
    with torch.no_grad():
      for i, captions in enumerate(tqdm(loader)):

          caption_tokenlized = clip_tokenizer(captions, padding="max_length", max_length=77, return_tensors="pt", truncation=True)
          clip_input = caption_tokenlized["input_ids"].cuda()
          
          clip_text_embedding = clip_model(clip_input)["pooler_output"] # torch.Size([1, 768])
          # Calculate the required padding size
          padding_size = 1024 - 768

          # Pad the embedding
          padded_embedding = F.pad(clip_text_embedding, (0, padding_size))

          clip_out_all.append(padded_embedding.cpu().numpy())

      clip_out_all = np.concatenate(clip_out_all)
      try:
        torch.save(clip_out_all, "full_laion_coco_clip_features_{}.pt".format(split))
        with open("full_laion_coco_clip_features_{}.npy".format(split), 'wb') as e:
          np.save(e, clip_out_all, allow_pickle=True)
      except:
        print("torch save error")

  return clip_out_all
        


def get_bos_sentence_eos(laion_coco_dataset, berttokenizer, split, clip_backbone):
    print(clip_backbone)
    # import ipdb; ipdb.set_trace()
    plain_text_dataset_path = '/bian_data/safe_diffusion/dataloaders/processed_laion_coco/{}/5xCaptions/full_laion_coco_processed_annot_{}.npy'.format(clip_backbone, split)
    if os.path.isfile(plain_text_dataset_path):
        with open(plain_text_dataset_path, 'rb') as e:
            bos_sentence_eos = np.load(e, allow_pickle=True)
            bos_sentence_eos = bos_sentence_eos.tolist()
    else:
        print('preprocessing all sentences...')
        bos_sentence_eos = []
        for i, caption in enumerate(tqdm(laion_coco_dataset)):
                bos_sentence_eos.append(berttokenizer.bos_token + ' ' + caption + ' ' + berttokenizer.eos_token)
        with open('full_laion_coco_processed_annot_{}.npy'.format(clip_backbone, split), 'wb') as e:
            np.save(e, bos_sentence_eos, allow_pickle=True)
    return bos_sentence_eos


def get_bert_training_features(coco_dataset, train, clip_backbone):
    berttokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
    # sentences = get_bos_sentence_eos(coco_dataset, berttokenizer, train, clip_backbone)
    # print('tokenizing all processed sentences...')
    # if train == "train":
    #     tokenized = berttokenizer(sentences, padding=True,
    #                             truncation=True, max_length=77,
    #                             return_token_type_ids=False, return_tensors='pt')
    #     torch.save(tokenized,"bert_tokenized_train_20m.pt")
    # else:
    #     tokenized = berttokenizer(sentences, padding=True,
    #                             truncation=True, max_length=77,
    #                             return_token_type_ids=False, return_tensors='pt')
    #     torch.save(tokenized,"bert_tokenized_val_20m.pt")        
    if train=="train":
        tokenized = torch.load("../bert_tokenized_train.pt",weights_only=False)
    else:
        tokenized = torch.load("../bert_tokenized_val.pt",weights_only=False)
    
    if train=="train":
        label_ids = copy.deepcopy(tokenized['input_ids'])[8000000:]
        label_ids[label_ids == 0] = -100
        input_ids = tokenized['input_ids'][8000000:]
        attention_mask = tokenized['attention_mask'][8000000:]
        # import ipdb; ipdb.set_trace()
    else:
        label_ids = copy.deepcopy(tokenized['input_ids'])
        label_ids[label_ids == 0] = -100
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
    return input_ids, attention_mask, label_ids



def get_loader(train, clip_backbone="CLIP-vit-large", sap):
    if train:
        split='train'
    else:
        split='val'

    laion_coco_dataset = None #*My_laion_coco_dataset(train)
    if train:
        clip_features = get_clip_text_features(laion_coco_dataset, split, clip_backbone, device='cuda')[8000000:]
    else:
        clip_features = get_clip_text_features(laion_coco_dataset, split, clip_backbone, device='cuda')
    input_ids, attention_mask, label_ids = get_bert_training_features(laion_coco_dataset, split, clip_backbone)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    clip_features = torch.tensor(clip_features, dtype=torch.long)
    print(input_ids.size(), attention_mask.size(), label_ids.size(), clip_features.size())
    hidden_size = clip_features.size(1)
    print(clip_features.repeat(1,1).view(-1, hidden_size).size())
    dataset = TensorDataset(input_ids, attention_mask, label_ids, clip_features.repeat(1,1).view(-1, hidden_size))
    loader = DataLoader(dataset=dataset, batch_size=52, num_workers=4, shuffle=True)
    return loader
    # import ipdb; ipdb.set_trace()


if __name__=='__main__':
    #with open('./processed_coco/{}/coco_clip_features_{}.npy'.format('ViT-B32', 'train'),'rb') as e:
    #    clip_out_all = np.load(e, allow_pickle=True)
    #print(np.shape(clip_out_all))
    #len(dset) 118287
    # dset = My_laion_coco_dataset(train=True)
    max_length=0
    get_loader(train=True)
    import ipdb; ipdb.set_trace()
    
    exit()
    # get_loader(train=False)
    # exit()
    for i, (image, captions) in enumerate(tqdm(dset)):
        import ipdb; ipdb.set_trace()
        pass
