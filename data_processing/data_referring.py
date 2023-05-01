import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random

from bert.tokenization_bert import BertTokenizer

import h5py
from refer.refer import REFER

from args import get_parser

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()

class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)

        self.max_tokens = 32

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids
        
        self.new_ref_ids = dict()
        self.input_ids = dict()
        self.attention_masks = dict()
        self.input_ids_val = []
        self.attention_masks_val = []
        self.raw_sentence = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        self.new_items = [[10505, 10508, 10507],[10506, 10510, 10513, 10511],[10509, 10515, 10514, 10512],[10509, 10510, 10507], [10506, 10508, 10514, 10511],[10505, 10515, 10513, 10512], [238,236],[238,237],[16906, 16904],[16906, 16905],[7132, 7131],[7129, 7130],[10155, 10153],[10152,10154],[10029, 10027],[10028,10030],[12388,12386],[115706,12389],[19913,19914],[19915,19916],[22330,22331],[22329,22328],[27015, 27014],[27013,27016],[28871,28872],[28873,28870],[28895,28896],[28897,28898],
                          [29849, 29848],[29847,29850],[34395,34393],[34394,34392],[35535, 35533],[35536,35534],[39245,39244],[39242,39243],[40200,40201],[40202,40199],[40183, 40185],[40184,40182],[41415,41412],[41413,41414],[43036,43035],[43034,43037],[44676,44678],[44677,44679],[48139,48138],[48140,48141],[48845,48846],[48847,48844],[48894,48897],[48896,48895],[9325,9327,9326],[9328,9329,9326],[16149,16150,16148],[16148,16151,16152],[17347,17344,17346],[17343,17345],[27818,27819,27822],[27820,27821],
                          [30020,30019,30021],[30023,30022],[37167,37168],[37169,37171,37170],[42140,42141,42138],[42142,42139],[49497,49493,49494],[49495,49496],[21708,21709,21712],[21710,21713,21711],[25047,25048,25049],[25050,25051,25052],[29768,29770,29771],[29772,29773,29769],[39372,39371,39370],[39374,39375,39373],[5394,5396,5395],[5397,5399,5398,5393],[5397,5399,5398,5393],[46039,46042,46040,46041],[46037,46038,46036],[46039,46042,46040,46041],[46044,46043],
                          [9381,9380,9378,9377],[9379,9382,9383,9384],[9379,9380,9378,9377],[9381,9382,9383,9384], [15590,15591,15592,15596],[15595,15594,15597,15593],[15597,15591,15592,15595],[15594,15590,15593,15596]]
        new_sent = {142207:'on leftmost', 142172:'first from the left', 142142: 'baby elephant attacked by an elephant in the back', 142130:'woman wearing sunglasses and holding a wine glass', 142094:'pink cream donut on the left',142058:"standing next to the puple", 142059:"middle", 142056:"a bald looking down"}
        self.eval_mode = eval_mode
        remove_ids = {460986:0, 46065:0, 474256:0, 402235:0, 519607:0, 123336:0, 238502:0, 294837:0, 330652:0, 6406:0, 90328:0, 149078:0, 235651:0, 260953:0, 382102:0, 395684:0, 474963:0, 581136:0, 13720:0, 14138:0, 22014:0, 60874:0, 79822:0, 98304:0, 114142:0, 113985:0, 124893:0, 169495:0, 183445:0, 237617:0, 248564:0, 248833:0, 270186:0, 324381:0, 349947:0, 439988:0, 466097:0, 464854:0, 500136:0, 387264:0, 579255:0}
        
        for r in ref_ids:
            ref = self.refer.Refs[r]
            
            sentences_for_ref = []
            attentions_for_ref = []
            
            if not self.eval_mode:
                if ref['image_id'] not in remove_ids:
                    key_name = str(ref['image_id'])#+'_'+str(ref['category_id'])
                    if key_name not in self.new_ref_ids:
                        self.new_ref_ids[key_name] = [ref['ref_id']]
                    else:
                        item_id = self.new_ref_ids.get(key_name)
                        item_id.append(ref['ref_id'])
                        self.new_ref_ids[key_name] = item_id
                sent_diff = dict()
                for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                    if sent_id in new_sent:
                        sentence_raw = new_sent[sent_id]
                    else:
                        sentence_raw = el['sent']
                    
                    if sentence_raw not in sent_diff:
                        sent_diff[sentence_raw] = 1
                    else:
                        sent_diff_list.append(ref)
                        
                    #attention_mask = [0] * self.max_tokens
                    #padded_input_ids = [0] * self.max_tokens

                    inputs = self.tokenizer(text=sentence_raw, add_special_tokens=True, padding="max_length", truncation=True, max_length=self.max_tokens)
                    padded_input_ids = inputs['input_ids']
                    attention_mask = inputs['attention_mask']
                    # truncation of tokens
                    """if (self.max_tokens-2) < len(el['tokens']):
                        input_ids = input_ids[:self.max_tokens-1] + [input_ids[-1]]                   
                        
                    else:
                        input_ids = input_ids[:self.max_tokens]"""

                    #padded_input_ids[:len(input_ids)] = input_ids
                    #attention_mask[:len(input_ids)] = [1]*len(input_ids)

                    sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                    attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))


            
                if i == 0:

                    sent2words = el['tokens']
                    
                    n = round(len(sent2words) * 0.4)
                    idx = np.random.choice(len(sent2words),n, replace=False)
                    words = []
                    for x in idx:
                        temp = sent2words[x]
                        if x != (len(sent2words)-1):
                            temp = temp + ' '
                        
                        words.append(temp)

                    for w in words:
                        sentence_raw = sentence_raw.replace(w,'',1)

                    inputs = self.tokenizer(text=sentence_raw, add_special_tokens=True, padding="max_length", truncation=True, max_length=self.max_tokens)
                    padded_input_ids = inputs['input_ids']
                    attention_mask = inputs['attention_mask']
                    sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                    attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))
            
                self.input_ids[ref['ref_id']]=sentences_for_ref
                self.attention_masks[ref['ref_id']]=attentions_for_ref

            else:
                for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                    sentence_raw = el['sent']
                    inputs = self.tokenizer(text=sentence_raw, add_special_tokens=True, padding="max_length", truncation=True, max_length=self.max_tokens)
                    padded_input_ids = inputs['input_ids']
                    attention_mask = inputs['attention_mask']
                    sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                    attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))
                self.input_ids_val.append(sentences_for_ref)
                self.attention_masks_val.append(attentions_for_ref)
            

        if not self.eval_mode:
            ones_list = []
            for k,v in self.new_ref_ids.items():
                
                if len(v)==12:
                    for _ in range(2):
                        vv = np.random.choice(12, 12, replace=False)
                        l = []
                        for idx_v in vv[:4]:
                            l.append(v[idx_v])  
                        self.new_items.append(l)
                        l = []
                        for idx_v in vv[4:8]:
                            l.append(v[idx_v])
                        self.new_items.append(l)
                        l = []
                        for idx_v in vv[8:]:  
                            l.append(v[idx_v])
                        self.new_items.append(l)

                elif len(v)==11:
                    for _ in range(2):
                        vv = np.random.choice(11, 11, replace=False)
                        l = []
                        for idx_v in vv[:4]:
                            l.append(v[idx_v])  
                        self.new_items.append(l)
                        l = []
                        for idx_v in vv[4:8]:
                            l.append(v[idx_v])
                        self.new_items.append(l)
                        l = []
                        for idx_v in vv[8:]:  
                            l.append(v[idx_v])
                        self.new_items.append(l)

                elif len(v)==10:
                    for _ in range(2):
                        vv = np.random.choice(10, 10, replace=False)
                        l = []
                        for idx_v in vv[:4]:
                            l.append(v[idx_v])  
                        self.new_items.append(l)
                        l = []
                        for idx_v in vv[4:7]:
                            l.append(v[idx_v])
                        self.new_items.append(l)
                        l = []
                        for idx_v in vv[7:]:
                            l.append(v[idx_v])
                        self.new_items.append(l)

                elif len(v)==9:
                    for _ in range(2):
                        vv = np.random.choice(9, 9, replace=False)
                        l = []
                        for idx_v in vv[:3]:
                            l.append(v[idx_v])  
                        self.new_items.append(l)
                        l = []
                        for idx_v in vv[3:6]:
                            l.append(v[idx_v])
                        self.new_items.append(l)
                        l = []
                        for idx_v in vv[6:]:
                            l.append(v[idx_v])
                        self.new_items.append(l)

                elif len(v)==8:
                    for _ in range(2):
                        vv = np.random.choice(8, 8, replace=False)
                        l = []
                        for idx_v in vv[:4]:
                            l.append(v[idx_v])  
                        self.new_items.append(l)
                        l = []
                        for idx_v in vv[4:]:
                            l.append(v[idx_v])
                        self.new_items.append(l)

                elif len(v)==7:
                    for _ in range(2):
                        vv = np.random.choice(7, 7, replace=False)
                        l = []
                        for idx_v in vv[:4]:
                            l.append(v[idx_v])  
                        self.new_items.append(l)
                        l = []
                        for idx_v in vv[4:]:
                            l.append(v[idx_v])
                        self.new_items.append(l)

                elif len(v)==6:
                    for _ in range(2):
                        vv = np.random.choice(6, 6, replace=False)
                        l = []
                        for idx_v in vv[:3]:
                            l.append(v[idx_v])  
                        self.new_items.append(l)
                        l = []
                        for idx_v in vv[3:]:
                            l.append(v[idx_v])
                        self.new_items.append(l)
                
                elif len(v)==5: 
                        self.new_items.append(v)
                        self.new_items.append(v)
                
                elif len(v)==4:
                        self.new_items.append(v)
                        self.new_items.append(v)

                elif len(v)==3:
                        self.new_items.append(v)
                        self.new_items.append(v)
                elif len(v)==2:
                        self.new_items.append(v)
                elif len(v)==1:
                    ones_list.append(v[0])
            
            for _ in range(2):
                vv = np.random.choice(len(ones_list), len(ones_list), replace=False)
                l = []
                for v_ in vv:
                    if len(l) != 4:
                        l.append(ones_list[v_])
                    else:
                        self.new_items.append(l)
                        l = []
                        l.append(ones_list[v_])
                if len(l) != 1:
                    self.new_items.append(l)

            print(len(self.new_items))


    def get_classes(self):
        return self.classes

    def __len__(self):
        if self.eval_mode:
            return len(self.ref_ids)
        else:
            return len(self.new_items)
        
    def __getitem__(self, index):
        if self.eval_mode:
            this_ref_id = self.ref_ids[index]
            sent = self.refer.Refs[this_ref_id]['sentences']
            this_img_id = self.refer.getImgIds(this_ref_id)
            this_img = self.refer.Imgs[this_img_id[0]]

            img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")
            img_ndarray = np.array(img)
            ref = self.refer.loadRefs(this_ref_id)

            ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
            annot = np.zeros(ref_mask.shape)
            annot[ref_mask == 1] = 1

            annot = Image.fromarray(annot.astype(np.uint8), mode="P")
            
            if self.image_transforms is not None:
                # resize, from PIL to tensor, and mean and std normalization
                img, target = self.image_transforms(img, annot)


            embedding = []
            att = []
            sent_list=[]
            for s in range(len(self.input_ids_val[index])):
                sent_raw = sent[s]['sent']
                e = self.input_ids_val[index][s]
                a = self.attention_masks_val[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))
                sent_list.append(sent_raw)

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
            #print(tensor_embeddings)
            return img, target, tensor_embeddings, attention_mask, sent_list, img_ndarray



        else:
            ref_id_list = self.new_items[index]
            id1, id2 = np.random.choice(len(ref_id_list), 2, replace=False)

            this_ref_id1 = ref_id_list[id1]
            this_ref_id2 = ref_id_list[id2]

            this_img_id1 = self.refer.getImgIds(this_ref_id1)
            this_img_id2 = self.refer.getImgIds(this_ref_id2)
            this_img1 = self.refer.Imgs[this_img_id1[0]]
            this_img2 = self.refer.Imgs[this_img_id2[0]]

            img1 = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img1['file_name'])).convert("RGB")
            img2 = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img2['file_name'])).convert("RGB")

            ref1 = self.refer.loadRefs(this_ref_id1)
            ref2 = self.refer.loadRefs(this_ref_id2)

            ref_mask1 = np.array(self.refer.getMask(ref1[0])['mask'])
            annot1 = np.zeros(ref_mask1.shape)
            annot1[ref_mask1 == 1] = 1

            ref_mask2 = np.array(self.refer.getMask(ref2[0])['mask'])
            annot2 = np.zeros(ref_mask2.shape)
            annot2[ref_mask2 == 1] = 1

            annot1 = Image.fromarray(annot1.astype(np.uint8), mode="P")
            annot2 = Image.fromarray(annot2.astype(np.uint8), mode="P")

            if self.image_transforms is not None:
                # resize, from PIL to tensor, and mean and std normalization
                img1, target1 = self.image_transforms(img1, annot1)
                img2, target2 = self.image_transforms(img2, annot2)

            choice_sent1 = np.random.choice(len(self.input_ids[id1]))
            choice_sent2 = np.random.choice(len(self.input_ids[id2]))
            tensor_embeddings1 = self.input_ids[id1][choice_sent1]
            attention_mask1 = self.attention_masks[id1][choice_sent1]
            tensor_embeddings2 = self.input_ids[id2][choice_sent2]
            attention_mask2 = self.attention_masks[id2][choice_sent2]

            tensor_embeddings = torch.cat([tensor_embeddings1.unsqueeze(0), tensor_embeddings2.unsqueeze(0)], dim=0)
            attention_mask = torch.cat([attention_mask1.unsqueeze(0), attention_mask2.unsqueeze(0)], dim=0)

            img_batch = torch.cat([img1.unsqueeze(0), img2.unsqueeze(0)], dim=0)
            target_batch = torch.cat([target1.unsqueeze(0), target2.unsqueeze(0)], dim=0)
        return img_batch, target_batch, tensor_embeddings, attention_mask
