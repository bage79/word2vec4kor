# word2vec4kor
- Word2vec for Korean
- Keynote: https://github.com/bage79/word2vec4kor/raw/master/docs/korean_embedding_20180224_KCD.pdf
- Install & Run: https://github.com/bage79/word2vec4kor/blob/master/INSTALL.md
    - corpus -> dataset -> word2vec(numpy format) -> visualization data(tensorboard format)

### `tensorboard_log` word2vec visualization data 
- tensorboard embedding(projector) file format
- window size: 1
- Total unique words: 10,000
- Tokenized: white-space
- Embedding Dimension: 300
- Skip-Gram + Negative Sampling + Subsampling
```
mkdir ~/workspace
cd ~/workspace

git clone https://github.com/bage79/word2vec4kor
tensorboard --logdir=~/workspace/word2vec4kor/tensorboard_log
```
![demo](https://github.com/bage79/word2vec4kor/raw/master/img/demo.png)

### `ko.wikipedia.org.sentences` raw corpus 
- from `https://ko.wikipedia.org`
- Total sentences: about 3,115,431
```angular2html
wget https://gitlab.com/bage79/nlp4kor-ko.wikipedia.org/raw/master/data/ko.wikipedia.org.sentences.gz
gzip -d ko.wikipedia.org.sentences.gz
```

# Tips    
### Download korean Wikipedia dump file
- `https://dumps.wikimedia.org/kowiki/20180220/`

### Parse dump file(mediawiki format) to text file
- `https://pypi.python.org/pypi/mediawiki-parser/`

### Word2vec open source
- `https://github.com/theeluwin/pytorch-sgns`