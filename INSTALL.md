### Make workspace directory.
```
mkdir ~/workspace
cd ~/workspace
git clone http://github.com/bage79/word2vec4kor
mkdir ~/workspace/word2vec4kor/corpus
```

### Install word2vec open-source (for Pytorch)
```
cd ~/workspace
git clone https://github.com/theeluwin/pytorch-sgns
```

### Download korean wikipedia corpus (text format)
```
# sample corpus
cd ~/workspace/word2vec4kor/corpus
wget https://gitlab.com/bage79/nlp4kor-ko.wikipedia.org/raw/master/data/sample.ko.wikipedia.org.sentences
```
```
# full corpus
cd ~/workspace/word2vec4kor/corpus
wget https://gitlab.com/bage79/nlp4kor-ko.wikipedia.org/raw/master/data/ko.wikipedia.org.sentences.gz
gzip -d ko.wikipedia.org.sentences.gz
```

### Generate a vocaburary and a a train dataset (from the raw corpus)
```
cd ~/workspace/pytorch-sgns
python3 ./preprocess.py --window 5 --max_vocab 20000 --vocab ~/workspace/word2vec4kor/corpus/sample.ko.wikipedia.org.sentences --corpus ~/workspace/word2vec4kor/corpus/sample.ko.wikipedia.org.sentences --data_dir ~/workspace/word2vec4kor/data/
```

### Train and Create a word2vec (numpy format)
```
# run on GPU
cd ~/workspace/pytorch-sgns
python3 ./train.py --cuda --name sample.ko.wikipedia --e_dim 300 --n_negs 20 --epoch 10 --mb 4096 --ss_t 1e-5 --weights --data_dir ~/workspace/word2vec4kor/data --save_dir ~/workspace/word2vec4kor/models
```
```
# run on CPU
cd ~/workspace/pytorch-sgns
python3 ./train.py --name ko.wikipedia --e_dim 300 --n_negs 20 --epoch 10 --mb 4096 --ss_t 1e-5 --weights --data_dir ~/workspace/word2vec4kor/data --save_dir ~/workspace/word2vec4kor/models
```

### Convert word2vec from numpy format into tensorboard format.
```
mkdir ~/tensorboard_dir
cd ~/workspace/word2vec4kor
python3 ./word2vec_tensorboard.py --name sample.ko.wikipedia --top_n 10000 --data_dir ~/workspace/word2vec4kor/data --tensorboard_dir ~/tensorboard_log
```

### Start Tensorboard
```
tensorboard --logdir=~/tensorboard_log/ --port=6006
```
http://localhost:6006
