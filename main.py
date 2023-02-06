from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
import Levenshtein
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")


import torch
#single wordのみ使用
single_words=[]
print("vocabrary size:",len(processor.tokenizer.vocab.keys()))
for w in processor.tokenizer.vocab.keys():
  if "</w>" in w:
    single_words.append(w[:-4])
print("single words:",len(single_words))

#alphabetのみ抽出
cleaned_single_words=[]
for s in single_words:
  if s.isalpha():
    if len(s)<6:
      cleaned_single_words.append(s)

#promptで入力しても1トークンになっている場合のみ使用
prefix="A road sign with the word \""
suffix= "\""
prompts=processor(
    text=list(map(lambda w:prefix + w +suffix,cleaned_single_words))
    ,return_tensors="pt"
    ,padding=True
).input_ids

cleaned_prompts=[]
#promptから単語部分のみ抽出
for i,p in enumerate(prompts):
  if p[7]==257 and p[9]==257:
    #全てシーケンス長は同じ
    cleaned_prompts.append(prefix + cleaned_single_words[i] + suffix)
print("cleaned prompts:",len(cleaned_prompts))



import gc
#一気に処理するとOOMが発生
batch_size=4000
for i in range(0,len(cleaned_prompts),batch_size):
  print(i)
  inpt=processor(text=cleaned_prompts[i:i+batch_size],return_tensors="pt")
  torch.save(model.text_model(**inpt.to("cuda")).last_hidden_state[:,8,:].to("cpu"),str(i))
  del inpt
  gc.collect()
  
  
  
  embs=torch.tensor([])
for i in range(0,len(cleaned_prompts),batch_size):
  embs=torch.cat([embs,torch.load(str(i))])
embs=embs.to("cpu")


dist_edit=[]
dist_euclid=[]
def return_dists(pairs):
  i,j=pairs
  if abs(len(cleaned_single_words[i])-len(cleaned_single_words[j]))<3:
    set_a=set(list(cleaned_single_words[i]))
    set_b=set(list(cleaned_single_words[j]))
    if (len(set_a & set_b)) != 0:
      a_and_b=len(set_a)/(len(set_a & set_b))
      if a_and_b > 0.9 and a_and_b < 1.11111:        
        edit=Levenshtein.distance(cleaned_single_words[i],cleaned_single_words[j])
        if edit<4:
          return [edit,torch.sqrt(torch.sum((embs[i]-embs[j])**2))]
  return [0,0]


import itertools
lis = list(range(len(embs)))
pairs=[]
for pair in itertools.combinations(lis, 2):
	pairs.append(pair)
  
  
  
xylist=list(map(return_dists,tqdm(pairs)))


x_list=[0]
y_list=[0]
for a in xylist:
  if(a==[0,0]):
    pass
  else:
    x_list.append(a[0])
    y_list.append(a[1])
    
    

y_list=[y.detach().numpy().item() for y in  y_list[1:]]
y_list=[0]+y_list



zeros=[0]
ones=[]
twoes=[]
threes=[]
for i in range(len(x_list)):
   if x_list[i]==0:
     zeros.append(y_list[i])
   elif x_list[i]==1:
     ones.append(y_list[i])
   elif x_list[i]==2:
     twoes.append(y_list[i])
   elif x_list[i]==3:
     threes.append(y_list[i])

import numpy as np
np.mean(zeros),np.mean(ones),np.mean(twoes),np.mean(threes),np.var(zeros),np.var(ones),np.var(twoes),np.var(threes)


with open("dists2.txt", "w") as output:
    output.write(str(xylist))
