# Bidirectional Encoder Representations from Transformers (BERT)

BERT is the state of the art Embedding algorithm in Natural Language Processing introduced by Google researchers in 2018. In short, embedding techniques aim to bring tokens (words or subwords in the sentences) into the numerical representation for feeding to Neural Networks for further processings. BERT reaches the state-of-the-art performance compared to all other embedding methods, such as GloVe by stanford, Fasttext by Facebook (Meta), Word2Vec by Google and the rest. For further info on other embedding techniques refer to another Github repository in my account, named [nlprocessing]().  

Let's divide BERT into three subsections and describe each of them in detail.

## 1. Tokenization: In the first step of every NLP project, raw text must be tokenized (splitted into words). Every algorithm has its own way of tokenization. In BERT, this process forms by splitting sentences into words and then subwords if required. For instance, consider the sentence below: 

"Hello, how are you?"

This sentence is splitted into words like this:

('Hello', ',', 'how', 'are', 'you', '?')

also, a word like "unhappiness" after the step above is splitted into subwords: ('un', 'hap', 'pi', 'ness')



