# chainer_devise
implementation of DeViSE in Chainer

# 画像の収集
util/myrunを使って、500クラス、各クラス1000枚として収集。必ずしも1000枚とは限らない。詳細はhistogram.pngを参照のこと。
`
/Volumes/TOSHIBA EXT/mac/image_net/images/histogram.png
`

# enwikiの加工
## ruby関係のインストール。
```
$> sudo port install rbenv ruby-build
$> rbenv install --list
$> rbenv install 2.3.1
$> rbenv local 2.3.1
$> rbenv global 2.3.1
$> rbenv exec gem install nokogiri -- --use-system-libraries 
$> rbenv exec gem install wp2txt bundler
$> rbenv rehash
```

## 変換
```
$> rbenv exec wp2txt --input-file enwiki-20101011-pages-articles.xml.bz2
```
途中で、こんなエラーが出る。
```
stack level too deep (SystemStackError)
```
[ここ](http://medialab.di.unipi.it/wiki/Wikipedia_Extractor)にあるWikiExtractor.pyを使う。
```
$> ./WikiExtractor.py -cb 250K -o extracted enwiki-20101011-pages-articles.xml.bz2
```
ファンがフル回転する。7つのプロセスが起動する。

