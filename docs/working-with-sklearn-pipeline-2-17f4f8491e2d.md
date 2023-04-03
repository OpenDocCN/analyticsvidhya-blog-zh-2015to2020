# 使用 Sklearn Pipeline-2

> 原文：<https://medium.com/analytics-vidhya/working-with-sklearn-pipeline-2-17f4f8491e2d?source=collection_archive---------39----------------------->

![](img/34c2f465c43a4c19befb8268801bd6ab.png)

今天的帖子是我们将了解 Sklearn 管道的三部分中的第二部分。你可以在这里看到第一部分:

1.  [https://medium . com/@ vikashprasad _ 16952/working-with-sk learn-pipeline-part 1-419 b 32 fc 8 b 1](/@vikashprasad_16952/working-with-sklearn-pipeline-part1-419b32fc8b1)

在这一部分，我们将探索 sklearn 管道，观察管道的属性，并在数据集上拟合不同的模型。继续上一篇文章，这是我们最终使用 ExtraTreesClassifier 作为估计器并使用不同的自定义和传统功能开发的管道(我假设您已经阅读了数据，请参考上一篇博客了解数据细节和功能)。

```
ET_pipeline_pos_tag = Pipeline([
   ('u1', FeatureUnion([
      ('tfdif_features', Pipeline([('cleaner', FeatureCleaner()),
                            ('tfidf', TfidfVectorizer(max_features=40000, ngram_range=(1, 3))),
                            ])),
      ('numerical_features', Pipeline([('numerical_feats', FeatureMultiplierCount()),
                               ('scaler', StandardScaler()), ])),

      ('pos_features', Pipeline([
         ('pos', PosTagMatrix(tokenizer=nltk.word_tokenize)),
      ])),
   ])),
   ('clf', ExtraTreesClassifier()),
])
```

让我们分解上面的管道，看看管道中涉及的各个步骤，您应该看到如下内容，基本上它显示了管道中的各个步骤，您也可以单独使用这些步骤来查看转换:

```
ET_pipeline_pos_tag.named_steps{'feature_union': FeatureUnion(n_jobs=None,
              transformer_list=[('tfdif_features',
                                 Pipeline(memory=None,
                                          steps=[('cleaner',
                                                  FeatureCleaner(clean=True)),
                                                 ('tfidf',
                                                  TfidfVectorizer(analyzer='word',
                                                                  binary=False,
                                                                  decode_error='strict',
                                                                  dtype=<class 'numpy.float64'>,
                                                                  encoding='utf-8',
                                                                  input='content',
                                                                  lowercase=True,
                                                                  max_df=1.0,
                                                                  max_features=40000,
                                                                  min_df=1,
                                                                  ngram_range=(1,
                                                                               3),
                                                                  norm='l2',
                                                                  pr...
                                                                         word_count=True,
                                                                         word_density=None,
                                                                         word_unique_percent=None,
                                                                         words_vs_unique=None)),
                                                 ('scaler',
                                                  StandardScaler(copy=True,
                                                                 with_mean=True,
                                                                 with_std=True))],
                                          verbose=False)),
                                ('pos_features',
                                 Pipeline(memory=None,
                                          steps=[('pos',
                                                  PosTagMatrix(normalize=True,
                                                               tokenizer=<function word_tokenize at 0x7f92f68fc620>))],
                                          verbose=False))],
              transformer_weights=None, verbose=False),
 'clf': ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
                      max_depth=None, max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators='warn',
                      n_jobs=None, oob_score=False, random_state=None, verbose=0,
                      warm_start=False)}
```

或者，您也可以通过以下方式查看这些步骤，“clf”是估计值:

```
In [28]: ET_pipeline_pos_tag.named_steps.keys()
Out[28]:dict_keys(['feature_union', 'clf'])
```

让我们来看看 FeatureUnion，以及我们如何访问它并分别拟合我们的数据，当您执行“ET_pipeline_pos_tag.predict”时，管道会在内部负责转换数据并进行预测。

```
In[54]: pipeline_fit = ET_pipeline_pos_tag.named_steps['feature_union'].fit(X_train)
pipeline_fit_train = pipeline_fit.transform(X_train)
pipeline_fit_trainOut[54]: <6090x40019 sparse matrix of type '<class 'numpy.float64'>'
	with 204786 stored elements in Compressed Sparse Row format>In[55]: trans_test= pipeline_fit.transform(X_test)
```

如您所见，管道完全适合训练数据集，并在测试数据集上进行转换。我们现在需要从 pipeline 访问模型，并对其进行拟合和预测。了解管道的各个步骤很重要，这样您就可以在需要时进行调试。

```
In[61]: model = ET_pipeline_pos_tag['clf'].fit(pipeline_fit_train,y=y_train)
modelOut[61]: ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
                     max_depth=None, max_features='auto', max_leaf_nodes=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=1, min_samples_split=2,
                     min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
                     oob_score=False, random_state=None, verbose=0,
                     warm_start=False)In[49]: model.predict(trans)
```

让我们继续下一步，我们需要在管道中安装不同的模型，以找到最佳模型。在这一部分中，我们将关注如何添加不同的模型，在最后一部分中，我们将关注使用管道在模型上应用 gridsearch cv。让我们看看如何在 pipeline 中添加不同的模型:

步骤 1:用管道声明模型:

```
def models():
 SVC_pipeline_pos_tag = Pipeline([
  ('feature_union', FeatureUnion([
   ('tfdif_features', Pipeline([('cleaner', FeatureCleaner()),
           ('tfidf', TfidfVectorizer(max_features=40000, ngram_range=(1, 3))),
           ])),
   ('numerical_features', Pipeline([('numerical_feats', FeatureMultiplierCount()),
            ('scaler', StandardScaler()), ])),('pos_features', Pipeline([
    ('pos', PosTagMatrix(tokenizer=nltk.word_tokenize)),
   ])),
  ])),
  ('clf', LinearSVC()),
 ])ET_pipeline_pos_tag = Pipeline([
  ('feature_union', FeatureUnion([
   ('tfdif_features', Pipeline([('cleaner', FeatureCleaner()),
           ('tfidf', TfidfVectorizer(max_features=40000, ngram_range=(1, 3))),
           ])),
   ('numerical_features', Pipeline([('numerical_feats', FeatureMultiplierCount()),
            ('scaler', StandardScaler()), ])),('pos_features', Pipeline([
    ('pos', PosTagMatrix(tokenizer=nltk.word_tokenize)),
   ])),
  ])),
  ('clf', ExtraTreesClassifier()),
 ])AdaBoost_pipeline_pos_tag = Pipeline([
  ('feature_union', FeatureUnion([
   ('tfdif_features', Pipeline([('cleaner', FeatureCleaner()),
           ('tfidf', TfidfVectorizer(max_features=40000, ngram_range=(1, 3))),
           ])),
   ('numerical_features', Pipeline([('numerical_feats', FeatureMultiplierCount()),
            ('scaler', StandardScaler()), ])),('pos_features', Pipeline([
    ('pos', PosTagMatrix(tokenizer=nltk.word_tokenize)),
   ])),
  ])),
  ('clf', AdaBoostClassifier()),
 ])GRD_pipeline_pos_tag = Pipeline([
  ('feature_union', FeatureUnion([
   ('tfdif_features', Pipeline([('cleaner', FeatureCleaner()),
           ('tfidf', TfidfVectorizer(max_features=40000, ngram_range=(1, 3))),
           ])),
   ('numerical_features', Pipeline([('numerical_feats', FeatureMultiplierCount()),
            ('scaler', StandardScaler()), ])),('pos_features', Pipeline([
    ('pos', PosTagMatrix(tokenizer=nltk.word_tokenize)),
   ])),
  ])),
  ('clf', GradientBoostingClassifier()),
 ])pipelines = [SVC_pipeline_pos_tag,ET_pipeline_pos_tag,AdaBoost_pipeline_pos_tag,GRD_pipeline_pos_tag]return pipelines
```

第二步:我们现在需要在一个循环中运行它来观察最佳模型，我们使用 F1 分数作为衡量标准来测试并找到最佳模型:

```
In[66]: grids = models()
grid_dict = {0: 'svc', 1: 'extratrees',
             2: 'adaboost', 3: 'gradientboost'}

print('Performing model optimizations...')
best_acc = 0.0
best_clf = 0
best_gs = ''
for idx, gs in enumerate(grids):
    print('\nEstimator: %s' % grid_dict[idx])
    gs.fit(X_train, y_train)
    # Predict on test data with best params
    y_pred = gs.predict(X_test)
    # Test data accuracy of model with best params
    print('Test set f1 score : %.3f ' % f1_score(y_test, y_pred))
    # Track best (highest test accuracy) model
    if f1_score(y_test, y_pred) > best_acc:
        best_acc = f1_score(y_test, y_pred)
        best_gs = gs
        best_clf = idx
print('\nClassifier with best test set f1 score: %s' % grid_dict[best_clf])
# Save best grid search pipeline to file
dump_file = 'best_gs_pipeline.joblib'
joblib.dump(best_gs, dump_file, compress=1)
print('\nSaved %s pipeline to file: %s' % (grid_dict[best_clf], dump_file))Performing model optimizations...

Estimator: svc
Test set f1 score : 0.749 

Estimator: extratrees
Test set f1 score : 0.674 

Estimator: adaboost
Test set f1 score : 0.698 

Estimator: gradientboost
Test set f1 score : 0.699 

Classifier with best test set f1 score: svc

Saved svc pipeline to file: best_gs_pipeline.joblib
```

正如你所看到的，最好的模型是 F1 分数约为 0.75 的 LinearSVC，在下一篇文章中，我们将尝试在 GridsearchCV 和 pipelines 的帮助下对模型进行微调。如果这篇文章有任何帮助，请鼓掌:)

感谢阅读！！