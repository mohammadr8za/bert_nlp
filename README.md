# Bidirectional Encoder Representations from Transformers (BERT)

BERT is the state of the art Embedding algorithm in Natural Language Processing introduced by Google researchers in 2018. In short, embedding techniques aim to bring tokens (words or subwords in the sentences) into the numerical representation for feeding to Neural Networks for further processings. BERT reaches the state-of-the-art performance compared to all other embedding methods, such as GloVe by stanford, Fasttext by Facebook (Meta), Word2Vec by Google and the rest. For further info on other embedding techniques refer to another Github repository in my account, named [nlprocessing]().  

Let's divide BERT into three subsections and describe each of them in detail.

## 1. Tokenization: 

In the first step of every NLP project, raw text must be tokenized (splitted into words or even subwords). Every algorithm has its own way of tokenization. In BERT, this process takes place by splitting sentences into words and then subwords if required. For instance, consider the sentence below: 

**"Hello, how are you?"**

This sentence is splitted into words like this:

**('Hello', ',', 'how', 'are', 'you', '?')**

in the meanwhile, a word like "unhappiness" after the tokenization (the step mentioned above) is splitted into subwords: 

**('un', 'hap', 'pi', 'ness')**

After this step, according to a Vocabulary (a list of words that is already defined including a large number of words), an ID is assigned to each token (word or subword). IDs are predefined in BERT tokenizer. So, at the end of this step, we will have a list of IDs representing each sentence and each element of the list represents a particular word or subword. But, the main challenge here, like all other text processing is the unstructured from of the data. This problem leads to inputs of different length to the network. To solve this issue, a padding is considered to determine a fixed size for inputs to the network. 
### Padding
In this step, after the provision of a list for each sentence, due to the inconsistency in the list lengths, a padding system is defined (accoridng to average length of input lists). So, padding_len is then defined. In this process, inputs of length larger than padding_len are truncated from both ending and beginning until the reach the padding_len, also, input of length less than padding_len pads until they reach the defined fixed length. Needless to say, the truncation step leads to loss of information and is inevitable. However, considering an appropriate padding_len, this problem is tackled to a good extend. Finally, this step provides us with inputs of fixed length appropriate for feeding to the network.  



