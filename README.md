## How to use
### Data
[日本放射線技術学会　miniJSRT_database](http://imgcom.jsrt.or.jp/minijsrtdb/)

git clone した`Unet-sample`ディレクトリ下で以下のコマンドを実行してデータをダウンロードして解凍する。
```
wget http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/2018/11/Segmentation01.zip
unzip Segmentation01.zip
```

### execution
`setting_data.py`で入力データを整えたのちに`unet_main.py`を実行すれば動作する。

### result
[6](https://github.com/user-attachments/assets/3b129de1-d050-4125-b8a7-0397f7336463)