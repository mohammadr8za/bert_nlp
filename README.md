# Bidirectional Encoder Representations from Transformers (BERT)

![LGIlP](https://github.com/mohammadr8za/bert_nlp/assets/72736177/a7bcfd5f-6417-43ca-a260-961436741e04)

BERT is the state of the art Embedding algorithm in Natural Language Processing introduced by Google researchers in 2018. In short, embedding techniques aim to bring tokens (words or subwords in the sentences) into the numerical representation for feeding to Neural Networks for further processings. BERT reaches the state-of-the-art performance compared to all other embedding methods, such as GloVe by stanford, Fasttext by Facebook (Meta), Word2Vec by Google and the rest. For further info on other embedding techniques refer to another Github repository in my account, named [nlprocessing]().  

**NOTE:** BERT is named bidirectional, however, it is better to call it non-directional, because it uses transformer architecture that embeds each token according to all other tokens in the sequence without considering any specific order or direction. 

Let's detail BERT.

## 1. Network Architecture

![elmo-eemmbeddings-(1)](https://github.com/mohammadr8za/bert_nlp/assets/72736177/20695649-c7f8-4e7b-9815-e46d00587c2f)

BERT is an attention-based network introduced by Google researchers for the embedding task in Natural Language Processing. BERT is introduced in two particular versions, BASE and LARGE. The difference, as the names suggest is in the number of layers in the network architecture (network size). BASE version comprises 12 attention-based encoder layers, 512 hidden units and 12 attention heads. Dimension of embedded tokens in the output is 768 (110M parameters). However, BERT's LARGE version includes 24 attention-based encoder layers, with 512 hidden units and 16 attention heads. Moreover, its embedding size in 1024 (345M parameters). In both versions, maximum length of the input sequence is 512 (equal to the number of hidden units) and first token for both is the <CLS> that represents class token (it is stacked in the beginning of the input sequence, and its embedded corresponding is the first embedded output). In the output, we get the embeddings of tokens and the class embedding that can be used for tasks like sentiment analysis which falls within classification, Q-A, and Named Entity Recognition (NER). 


**Q) How to utilize?** BERT is challenging to train. It is a large network with a lot of attention encoder layers and subsequently requires a large dataset for training. Text datasets are mainly challenging to be handled, too, as they need to be specifically labeled for each task. So, it is always recommended to utilize the pre-trained versions of BERT. In order to adopt BERT architecture and embeddings in any particular task, we may add layers in the output of this architecture and transfer its learned features to our specific task (freeze bert before the definition of our model and use it as the core of our architecture).  
</p>

## 2. Tokenization: 

In the first step of every NLP project, raw text must be tokenized (splitted into words or even subwords). Every algorithm has its own way of tokenization. In BERT, this process takes place by splitting sentences into words and then subwords if required. For instance, consider the sentence below: 

**"Hello, how are you?"**

This sentence is splitted into words like this:

**('Hello', ',', 'how', 'are', 'you', '?')**

in the meanwhile, a word like "unhappiness" after the tokenization (the step mentioned above) is splitted into subwords: 

**('un', 'hap', 'pi', 'ness')**


After this step, according to a Vocabulary (a list of words that is already defined including a large number of words), an ID is assigned to each token (word or subword). IDs are predefined in BERT tokenizer. So, at the end of this step, we will have a list of IDs representing each sentence and each element of the list represents a particular word or subword. But, the main challenge here, like all other text processing is the unstructured from of the data. This problem leads to inputs of different length to the network. To solve this issue, a padding is considered to determine a fixed size for inputs to the network. 

### Padding

In this step, after the provision of a list for each sentence, due to the inconsistency in the list lengths, a padding system is defined (accoridng to average length of input lists). So, padding_len is then defined. In this process, inputs of length larger than padding_len are truncated from both ending and beginning until the reach the padding_len, also, input of length less than padding_len pads until they reach the defined fixed length. Obviously, the truncation step leads to loss of information and it is inevitable. However, considering an appropriate padding_len, this problem is tackled to a reasonable extend. Finally, this step provides us with inputs of fixed length appropriate for feeding to the network. There need to be a **trade-off** between padding_len and the computational cost. Needless to say, large padding_len leads to high computational cost and on the other handm small one causes more loss on input information. 


**Q) Is there any particular scheme to find an approriate padding_len?** The answer is YES! You may shape a histogram of input lengths and consider a weighted average of input length as your initial padding_len. Then if necessary, you may modify the initial padding_len (Sample code is provided in this repo).  

**Q) It is mentioned earlier that the network's input size (number of hidden units) is 512. What will happen to the rest of units when define a fixed input size (padding_len) less than 512?** The answer is straightforward: empty units will not be utilized! The network works with padding_len number of units!


### Attention Mask

BERT tokenizer also defines an attention mask. This mask aims to tell the network (which is attention-based) which tokens to attend during the embedding process. This mask is a binary tensor of size padding_len (which is the fixed size for input tensors). This binary mask represents 1 for each input token and 0 if padded inputs. In fact, the network during embedding attends input tokens corresponding to 1 elements in binary attention mask and neglect those equivalent to 0 elements. 



## 3. Embedding

Embedding means transforming tokens into the numerical represenation for further computations (feeding to the network). BERT first converts input tokens into IDs, as mentioned earlier. In the next step, these IDs are transformed into their corresponding dense vector representations (using pre-trained Word2Vec embeddings). Then inputs are ready to be utilized for any specific task.

**NOTE:** Other than the <CLS> class token, there is another specific tokens <SEP> that separates sentences in the input sequence. Its existence improves the embedding results by letting network understand the ending and beginning of sentences in the input.


## 4. Fine-Tuning BERT for Sentiment Analysis/Classification

In this section, we aim to utilize BERT, its pre-trained BASE version, for the task of sentiment analysis. We will use the abovementioned information, define a network with pretrained bert (from transformers library) as its core and fine-tune it for sentiment analysis/classification. Let's code and detail each section if require. 

## Step 1: Load Data

```
import pandas as pd

df = pd.read_csv('/content/sentiment_train.xls')
df.head(3)

```

## Step 2: Split Data into Train, Validation, and Test Sets

```
from sklearn.model_selection import train_test_split
import pandas as pd

X, y = df['sentence'].tolist(), df['label'].tolist()
print(f"sample sentence and its corresponding label are presented below: \n{X[0], y[0]}")

# we set random_state=1 for this tutorial to make sure we get the same split in each execution of the code (for the sake of consistency)
X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=0.4, random_state=1)

X_valid, X_test, y_valid, y_test = train_test_split(X_test_valid, y_test_valid, test_size=0.3, random_state=1)

# Write Train, Valid, and Test CSV annotations
train_dict = {'sentence': X_train, 'label': y_train}
train_df = pd.DataFrame(train_dict)
train_df.to_csv(r'/content/train.csv')

valid_dict = {'sentence': X_valid, 'label': y_valid}
valid_df = pd.DataFrame(valid_dict)
valid_df.to_csv(r'/content/valid.csv')

test_dict = {'sentence': X_test, 'label': y_test}
test_df = pd.DataFrame(test_dict)
test_df.to_csv(r'/content/test.csv')

```

## Step 3: Evaluate Length of Train Data Using Histogram

```
import matplotlib.pyplot as plt

train_len = [len(text.split()) for text in X_train]
plt.hist(train_len)
plt.title('Text Length Histogram')
plt.xlabel('Text Length')
plt.ylabel('Number of Text')

```

![histlength](https://github.com/mohammadr8za/bert_nlp/assets/72736177/6ddcd3da-6e4a-4217-b05b-2ee6c24fc1f7)



## Step 4: Load BERT Model and its Tokenizer from *transformers* Library

```
!pip install transformers

from transformers import AutoModel, BertTokenizerFast

bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

```

## Step 5: Define a Custom Dataset

```
from torch.utils.data import Dataset
import pandas as pd
import torch
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

class BertCustomDataset(Dataset):

  def __init__(self, path_csv_file, tokenizer, pad_len):
    super(BertCustomDataset, self).__init__()

    self.df = pd.read_csv(path_csv_file)

    self.texts, self.labels = df['sentence'].tolist(), df['label'].tolist()

    self.tokens = tokenizer.batch_encode_plus(self.texts,
                                              max_length=pad_len,
                                              pad_to_max_length=True,  # Pad short inputs to reach the fixed input size (pad_len)
                                              truncation=True)  # truncate longer inputs to keep them in fixed input size (pad_len)

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):

    sent_id, mask = torch.tensor(self.tokens['input_ids'][idx]), torch.tensor(self.tokens['attention_mask'][idx])
    label = torch.tensor(self.labels[idx])

    return sent_id, mask, label

```


## Step 6: Define the BERT-Based Model

```
import torch
from torch import nn

for param in bert.parameters():
  param.requires_grad = False


class BERTArchitecture(nn.Module):

  def __init__(self, base, num_classes: int=2):

    super(BERTArchitecture, self).__init__()

    self.bert = bert

    self.classifier = nn.Sequential(
        nn.Linear(in_features=768, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=num_classes)
    )

  def forward(self, sent_id, mask):

    _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

    # x = self.relu(cls_hs)

    x = self.classifier(cls_hs)

    return x

```



## Step 7: Define Datasets

```
train_dataset = BertCustomDataset(path_csv_file='/content/train.csv', tokenizer=tokenizer, pad_len=pad_len)
valid_dataset = BertCustomDataset(path_csv_file='/content/valid.csv', tokenizer=tokenizer, pad_len=pad_len)

sent_id, mask, label = train_dataset[0]
print(f"sentence ID: \t{sent_id}\nattention mask: \t{mask}\nlabel: \t{label}")

```

## Step 8: Define DataLoaders

```
from torch.utils.data import DataLoader

batch_size = 5

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size)

batch, mask, labels = next(iter(train_dataloader))
batch, mask, labels, len(train_dataloader)

```

## Step 9: Define Optimizer and Loss Function

```
import torch
from torch import nn

loss_fn = nn.CrossEntropyLoss()
optimzier = torch.optim.Adam(params=model.parameters(), lr=1e-03)

```

## Step 10: Define an Accuracy Function

```
def accuracy_fn(pred, true):

  count = torch.sum(pred == true)

  return (count/pred.shape[0]) * 100

```

## Step 11: Device Diagnostic Code


```
import torch

device = 'cuda' is torch.cuda.is_available() else 'cpu'

```


## Step 12: Define a Training Loop

```
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


Epochs = 4

train_loss, train_acc = [], []
valid_loss, valid_acc = [], []
for epoch in range(Epochs):

  print(f">>>>>>>>>>\tEpoch: {epoch+1}\t<<<<<<<<<<")

  true_labels, pred_labels = [], []  # for confusion matrix

  model.train()

  loss_t, acc_t = 0, 0
  for b_count, batch in enumerate(train_dataloader):

    sent_id, attention_mask, labels = batch
    sent_id, attention_mask, labels = sent_id.to(device), attention_mask.to(device), labels.to(device)

    model.zero_grad()

    preds = model(sent_id, attention_mask)

    loss = loss_fn(preds, labels)

    loss_t += loss

    acc = accuracy_fn(preds.argmax(dim=1), labels)
    acc_t += acc

    optimzier.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimzier.step()

    if (b_count + 1) % 500 == 0:
      print(f">>>>>>> batch: loss: {loss}\tacc: {acc}")

  loss_t /= len(train_dataloader)
  acc_t /= len(train_dataloader)

  model.eval()

  loss_v, acc_v = 0, 0
  with torch.inference_mode():

    for batch in valid_dataloader:

      sent_id, attention_mask, labels = batch
      sent_id, attention_mask, labels = sent_id.to(device), attention_mask.to(device), labels.to(device)

      preds = model(sent_id, mask)

      loss = loss_fn(preds.argmax(dim=1).type(torch.float), labels.type(torch.float))

      loss_v += loss

      acc = accuracy_fn(preds.argmax(dim=1), labels)

      acc_v += acc

      true_labels.expand(labels.cpu().numpy())
      pred_labels.expand(labels.argmax(dim=1).cpu().numpy())

    loss_v /= len(valid_dataloader)
    acc_v /= len(valid_dataloader)

  # Keep Metrics in Separated Lists
  train_loss.append(loss_t.item())
  train_acc.append(acc_t.item())
  valid_loss.append(loss_v.item())
  valid_acc.append(acc_v.item())

  print(f"Train:\t Loss: {loss_t} | Accuracy: {acc_t} \nValid:\t Loss: {loss_v} | Accuracy {acc_v}")

  # Create Confusion Matrix
  cm = confusion_matrix(true_labels, pred_labels)
  df_cm = pd.DataFrame(cm/np.sum(cm, axis=1),
                    index=['Positive', 'Negative'],
                    columns=['Positive', 'Negative'])

  plt.Figure(figsize=(8, 8))
  sn.heatmap(df_cm)
  plt.savefig('/content/cm.png')
  plt.close()


```













