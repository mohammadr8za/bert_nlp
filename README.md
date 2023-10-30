# Bidirectional Encoder Representations from Transformers (BERT)

BERT is the state of the art Embedding algorithm in Natural Language Processing introduced by Google researchers in 2018. In short, embedding techniques aim to bring tokens (words or subwords in the sentences) into the numerical representation for feeding to Neural Networks for further processings. BERT reaches the state-of-the-art performance compared to all other embedding methods, such as GloVe by stanford, Fasttext by Facebook (Meta), Word2Vec by Google and the rest. For further info on other embedding techniques refer to another Github repository in my account, named [nlprocessing]().  

Let's detail BERT.

## 1. BERT Network

BERT is an attention-based network introduced by Google researchers for the embedding task in Natural Language Processing. BERT is introduced in two particular versions, base and large. The difference, as the names suggest is in the number of layers in the network architecture (network size). Base version comprises 12 attention layers, 512 hidden units and 8 attention head and its embedding dimension is 768 (110M parameters). However, BERT's Large version includes 24 Multi Head Self Attention (MSA) Layers, with 512 hidden units and 8 attention head. Moreover, its embedding size in 1024 (310M parameters). In both versions, length of the input sequence is 512 (equal to the number of hidden units) and first token for both is the <CLS> that represents class token. In the output, we get the embeddings of tokens and the class embedding that can be used for tasks like sentiment analysis which falls within classification. 

**Q) How to train?** BERT is challenging to train. It is a large network with a lot of attention layers and susequqntly requires a large dataset. Text dataset are mainly challenging to be handled, too. So, it is always recommendded to utilize the pre-trained versions of BERT. In order to adopt BERT architecture and embeddings in any particular task, we may add layers in the output of this architecture and transfer its learned features to our specific task (freeze bert before the definition of our model and use it as the base of our architecture).  

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

### Attention Mask
BERT tokenizer also defines an attention mask. This mask aims to tell the network (which is attention-based) which tokens to attend during the embedding process. This mask is a binary tensor of size padding_len (which is the fixed size for input tensors). This binary mask represents 1 for each input token and 0 if padded inputs. In fact, the network during embedding attends input tokens corresponding to 1 elements in binary attention mask and neglect those equivalent to 0 elements. 



