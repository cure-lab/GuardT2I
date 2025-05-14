import argparse
import torch
from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
import os
from transformers import AdamW
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, DataLoader


class GuardT2IDataset(Dataset):
    def __init__(self, data_dict):
        self.ids   = data_dict["input_ids"]
        self.mask  = data_dict["attention_mask"]
        self.labels= data_dict["label_ids"]
        self.clip  = data_dict["clip_features"]

    def __len__(self):
        return self.ids.size(0)

    def __getitem__(self, idx):
        return {
          "input_ids":      self.ids[idx],
          "attention_mask": self.mask[idx],
          "labels":         self.labels[idx],
          "clip_feat":      self.clip[idx],
        }

class GuardT2IDataset_eval(Dataset):
    def __init__(self, data_dict):
        self.ids   = data_dict["input_ids"][:4000]
        self.mask  = data_dict["attention_mask"][:4000]
        self.labels= data_dict["label_ids"][:4000]
        self.clip  = data_dict["clip_features"][:4000]

    def __len__(self):
        return self.ids.size(0)

    def __getitem__(self, idx):
        return {
          "input_ids":      self.ids[idx],
          "attention_mask": self.mask[idx],
          "labels":         self.labels[idx],
          "clip_feat":      self.clip[idx],
        }



def train_decoder(cllm, train_loader, eval_loader, optimizer, accelerator):

    for epoch in range(args.num_epochs):
        acc_loss = 0
        print('Training : epoch {}'.format(epoch))
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
                            
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label_ids = batch['labels']
            clip_embeds = batch['clip_feat']
            clip_extended_embed = clip_embeds.repeat(1, 1).type(torch.FloatTensor)
            N, seq_length = input_ids.shape
            position_ids = torch.arange(0, seq_length).expand(N, seq_length)
            cllm.train()
            out = cllm(input_ids=input_ids.to(device),
                             position_ids=position_ids.to(device),
                             attention_mask=attention_mask.to(device),
                             encoder_hidden_states=clip_extended_embed.unsqueeze(1).to(device),
                             labels=label_ids.to(device))
            accelerator.backward(out.loss)
            optimizer.step()
            optimizer.zero_grad()
            acc_loss += out.loss.detach().item()
            avg_loss = out.loss.detach().item() 
            progress_bar.set_description(f"Epoch {epoch}/{args.num_epochs}, Loss: {avg_loss:.4f}")

        validation_loss = eval_decoder(cllm, eval_loader)
        print('validation loss in this epoch: ', validation_loss)
        state = {'net': cllm.state_dict(),
                 'epoch': epoch,
                 'validation loss': validation_loss}

        if epoch == 0:
            best_val_loss = validation_loss
            torch.save(state, args.saved_model_path+str(best_val_loss)[:7]+'.pt')
            # accelerator.save_model(cllm, args.saved_model_path+'.pt')
        else:
            if validation_loss < best_val_loss :
                best_val_loss = validation_loss
                torch.save(state, args.saved_model_path+str(best_val_loss)[:7]+'.pt')
                # accelerator.save_model(cllm, args.saved_model_path+str(best_val_loss)[:7]+".pt")

        # print('textdecoder Average loss on {} training batches in this epoch:{}\n'.format(num_batch, acc_loss/num_batch))
        
    return acc_loss


def eval_decoder(cllm, eval_loader):
    # num_batch = len(iter(eval_loader))
    print('cllm evaluating loss on validation data ...')
    acc_loss = 0
    cllm.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_loader)):
          
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label_ids = batch['labels']
            clip_embeds = batch['clip_feat']
            
            clip_extended_embed = clip_embeds.repeat(1, 1).type(torch.FloatTensor)

            N, seq_length = input_ids.shape
            position_ids = torch.arange(0, seq_length).expand(N, seq_length)
            out = cllm(input_ids=input_ids.to(device),
                                 position_ids=position_ids.to(device),
                                 attention_mask=attention_mask.to(device),
                                 encoder_hidden_states=clip_extended_embed.unsqueeze(1).to(device),
                                 labels=label_ids.to(device))
            acc_loss += out.loss.detach().item()
    # print('textdecoder Average loss on {} validation batches={}\n'.format(num_batch, acc_loss/num_batch))
    return acc_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50, help="End epoch")  # trained with 25 epochs
    parser.add_argument('--trained_path', type=str, default='./trained_models/')
    parser.add_argument('--dataset_path', type=str,default="/bian_data/NeurIPS2024_release/datasets/",help="Path to datasets.")
    args = parser.parse_args()
    
    accelerator = Accelerator()
    device = accelerator.device 
    args.saved_model_path = args.trained_path

    if not os.path.exists(args.saved_model_path):
        os.makedirs(args.saved_model_path)
    berttokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
    # berttokenizer = BertGenerationTokenizer.from_pretrained("/bian_data/safe_diffusion/bert_for_seq_generation_L-24_bbc_encoder")

    
    # Uncomment the data to enrich the training set
    data_train = torch.load(args.dataset_path+"GuardT2I_dataset_train_part2_4M.pt")
    # data_train = torch.load(args.dataset_path+"datasets/GuardT2I_dataset_train_part3_4M.pt")
    data_eval = torch.load(args.dataset_path+"GuardT2I_dataset_train_part1_2M.pt")
    
    dataset_train = GuardT2IDataset(data_train)
    dataset_eval = GuardT2IDataset_eval(data_eval)
    
    # batch_size 52 suits well for 24GB GPU.
    train_loader  = DataLoader(dataset_train, batch_size=52, shuffle=True, num_workers=4)
    eval_loader = DataLoader(dataset_eval, batch_size=52, shuffle=True, num_workers=4)
    device = accelerator.device
    
    bert_config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
    # bert_config = BertGenerationConfig.from_pretrained("/bian_data/safe_diffusion/bert_for_seq_generation_L-24_bbc_encoder")
    bert_config.is_decoder=True
    bert_config.add_cross_attention=True
    cllm = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder',
                                                       config=bert_config).to(device).train()
    # cllm = BertGenerationDecoder.from_pretrained('/bian_data/safe_diffusion/bert_for_seq_generation_L-24_bbc_encoder',
    #                                                    config=bert_config).to(device).train()
    #* uncomment the following code to fineturn from GuardT2I
    # cllm = torch.load("/bian_data/NeurIPS2024_release/guardt2i.pt", weights_only=False).to(device).train()
    optimizer = AdamW(cllm.parameters(), lr=args.lr)

    # cllm_checkpoint = torch.nn.DataParallel(cllm)
    #* cllm.load_state_dict(torch.load("path_to_your_checkpoint")["net"])


    cllm, optimizer, train_loader, eval_loader =  accelerator.prepare(
        cllm, optimizer, train_loader, eval_loader
    )
    loss = train_decoder(cllm, train_loader, eval_loader, optimizer, accelerator)
    print('final training loss={}'.format(loss))
