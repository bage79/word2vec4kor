### Make workspace directory.
```
mkdir ~/workspace
```

### Install word2vec open-source (for Pytorch)
```
cd ~/workspace
git clone https://github.com/theeluwin/pytorch-sgns
```

### Download korean wikipedia corpus (text format)
```
cd ~/workspace
git clone http://github.com/bage79/word2vec4kor
mkdir ~/workspace/word2vec4kor/data
cd ~/workspace/word2vec4kor/data
wget https://gitlab.com/bage79/nlp4kor-ko.wikipedia.org/raw/master/data/sample.ko.wikipedia.org.sentences.gz
wget https://gitlab.com/bage79/nlp4kor-ko.wikipedia.org/raw/master/data/ko.wikipedia.org.sentences.gz
gzip -d ko.wikipedia.org.sentences.gz
```

### Generate a vocaburary and a a train dataset (from the raw corpus)
```
cd ~/workspace/pytorch-sgns
python3 ./preprocess.py --window 5 --max_vocab 20000 --vocab ~/workspace/word2vec4kor/data/sample.ko.wikipedia.org.sentences --corpus ~/workspace/word2vec4kor/data/sample.ko.wikipedia.org.sentences --data_dir ~/workspace/word2vec4kor/data/
```

### Train and Create a word2vec (numpy format)
```
cd ~/workspace/pytorch-sgns
```
```
# run on GPU
python3 ./train.py --cuda --name sample.ko.wikipedia --e_dim 300 --n_negs 20 --epoch 10 --mb 4096 --ss_t 1e-5 --weights --save_dir ~/workspace/word2vec4kor/models --data_dir ~/workspace/word2vec4kor/data
```
```
# run on CPU
python3 ./train.py --name sample.ko.wikipedia --e_dim 300 --n_negs 20 --epoch 10 --mb 4096 --ss_t 1e-5 --weights --save_dir ~/workspace/word2vec4kor/models --data_dir ~/workspace/word2vec4kor/data
```

### Convert word2vec from numpy format into tensorboard format.
```
cd ~/workspace/word2vec4kor
python3 ./word2vec_tensorboard.py --top_n 10000 --data_dir ~/workspace/word2vec4kor/data

```
