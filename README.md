# Text classification

The goal of the project is to create a text classification with a dataset of [Amazon product reviews](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews)


Run app 

```
  -h,     --help              show this help message and exit
  -dd,    --download-data     If passed, the dataset for text classification will be downloaded.
  -pp,    --pre-processing    If passed, run text processing and save the processed text.
  --svm   {default,best}      Train SVM model. Options: 'default', 'best'. You can specify both.
  --bert  {default,best}      Train BERT model. Options: 'default', 'best'. You can specify both.
  -s,     --save              Save fine-tuned run models.
```

example run:
```
python src/main.py --download-data --pre-processing --svm {default, best} --bert {default, best} --save
```
