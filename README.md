# NMT_Tenforflow
A simple Neural Machine Translation implementation using [Tensorflow seq2seq](https://www.tensorflow.org/api_guides/python/contrib.seq2seq).

## 1. Dependency
- Python 3.5.2  
- tensorflow (1.12.0)  
- gensim (3.6.0)  
- nltk (3.3)  

## 2. DATA
Using IWSLT'15 English-Vietnamese dataset from [Stanford NMT Project Web](https://nlp.stanford.edu/projects/nmt/).  
- Train (133K sentence pairs): [data/iwslt15/train_2.en] [data/iwslt15/train_2.vi] 
- Test:  [data/iwslt15/tst2012.en] [data/iwslt15/tst2012.vi] [data/iwslt15/tst2013.en] [data/iwslt15/tst2013.vi] 

## 3. Usage
- Python3 NMT.py train for training;
- Python3 NMT.py test for testing;
- Python3 NMT.py translate for loading trained model and translate in interaction mode.

## 4. Results

After training for 10 epochs, BLEU score on testset is 16.835.

### Training log:
```
Iteration starts.
Number of batches per epoch : 1041
----------------------------------- epoch:  0 -----------------------------------  
step: 100, batch: 99, loss: 5.618868827819824  
step: 200, batch: 199, loss: 5.080287456512451  
step: 300, batch: 299, loss: 4.897921562194824  
step: 400, batch: 399, loss: 4.498010635375977  
step: 500, batch: 499, loss: 4.025182247161865  
step: 600, batch: 599, loss: 3.793147563934326  
step: 700, batch: 699, loss: 3.8505334854125977  
step: 800, batch: 799, loss: 4.4062113761901855  
step: 900, batch: 899, loss: 3.9689314365386963  
step: 1000, batch: 999, loss: 3.99228835105896  
 Epoch 1: Model is saved. Elapsed: 03:39:19.19   
  
----------------------------------- epoch:  1 -----------------------------------  
step: 1100, batch: 58, loss: 3.9536454677581787  
step: 1200, batch: 158, loss: 3.6259477138519287  
step: 1300, batch: 258, loss: 3.4871222972869873  
step: 1400, batch: 358, loss: 2.780867099761963  
step: 1500, batch: 458, loss: 3.1616005897521973  
step: 1600, batch: 558, loss: 2.9285504817962646  
step: 1700, batch: 658, loss: 2.8706881999969482  
step: 1800, batch: 758, loss: 2.4239518642425537  
step: 1900, batch: 858, loss: 2.863511323928833  
step: 2000, batch: 958, loss: 2.6131107807159424  
 Epoch 2: Model is saved. Elapsed: 07:24:17.88   

......  
  
----------------------------------- epoch:  9 -----------------------------------  
step: 9400, batch: 30, loss: 1.168839693069458  
step: 9500, batch: 130, loss: 0.7480009198188782  
step: 9600, batch: 230, loss: 0.7185280323028564  
step: 9700, batch: 330, loss: 1.2181029319763184  
step: 9800, batch: 430, loss: 0.9367976784706116  
step: 9900, batch: 530, loss: 1.1975377798080444  
step: 10000, batch: 630, loss: 0.5992949604988098  
step: 10100, batch: 730, loss: 0.46061110496520996  
step: 10200, batch: 830, loss: 1.0416465997695923  
step: 10300, batch: 930, loss: 0.8975763916969299  
step: 10400, batch: 1030, loss: 1.1043615341186523  
 Epoch 10: Model is saved. Elapsed: 37:11:54.28   
```

## 5. REFERENCE

1.  [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt)  
2.  [Tensorflow seq2seq Implementation of Text Summarization.](https://github.com/dongjun-Lee/text-summarization-tensorflow)
 

Note: Please feel free to contact if there are any questions, bugs or improvement suggestions.
