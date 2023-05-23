# 图像相似度检测
利用VGGnet的预训练模型来实现图像的检索，先用预训练模型来抽取图片的特征，然后把待检索的图像和数据库中的所有图像进行匹配，找出相似度最高的

## Install packages
```sh
pip install -r requirements.txt
```

## run
```sh
python streamlit run .\main.py
```

