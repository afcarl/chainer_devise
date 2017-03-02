# chainer_devise
implementation of DeViSE in Chainer

# 画像の収集
util/myrunを使って、500クラス、各クラス1000枚として収集。必ずしも1000枚とは限らない。詳細はhistogram.pngを参照のこと。
`
/Volumes/TOSHIBA EXT/mac/image_net/images/histogram.png
`
残念ながら、これ以上画像を増やすことができません。
`collectImagenet_selected.py`で試しましたが、だめでした。

# caffe modelの変換
`visual/load_caffemodel.py`を使って、caffemodelをchainerのモデルに変換する。

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
$> ./WikiExtractor.py --processes 3 -cb 10M -o extracted enwiki-20101011-pages-articles.xml.bz2
```
3つのプロセスが起動する。--processesに指定するプロセス数のデフォルトは7、このまま実行するとファンがフル回転します。10M単位でテキストファイルに変換し、そのあと圧縮します。途中で止まりました。

[ここ](https://markroxor.github.io/gensim/static/notebooks/online_w2v_tutorial.html)を参考にします。
20101011のwikidumpに適用すると以下のエラーが出ます。

```
AttributeError: 'NoneType' object has no attribute 'text'
```
最近のwikidumpに適用してみます。
できましたが、８日ほどかかりました。
そのあと、gensim.models.word2vec.LineSentenceのオブジェクトを作る。
一回呼び出すと、一文を単語のリストにして返してくれる。


