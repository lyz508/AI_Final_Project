# Writing with AI

## Proposal

- NYCU 2022 spring AI Final Project

- The main proposal is to use the GPT-2 model trained & generating text

## Simple Use

- Download the release.

- Follow the step below to construct directories.

- Put files to the right position

- Train model

  ```shell
  python train.py
  ```

- Write with the model

	```shell
	python write.py [--dir <model_path>] [--max_len <expected_len>]
	```

## Code Structure

### Structure of Program

- Structure of the program

<img src="media/sample-code-structure-before.jpg" alt="image-20220613190118332"  />

- files

  - `src/*`

    - `config.py`: Store program configuration and will be instantiated in `train.py` or anywhere it need to be called.

    - `model.py`: Including model initialize, train, save, visualize and log output. A combination of core functions.

    - `tokenization.py`: Use this to trained an BPE tokenizer.

      ```shell
      python tokenization.py
      ```

      > This need to be run if there are no corresponding tokenizer.

  - `train.py`

    - It will load tokenizer, build model, setup all project configuration and start training the module.

  - `write.py`

    - It can be used to generate text with existed models, which will stored in trained_model directory.

### Mkdir

- Some directories may need to be constructed before the program runs

  ```shell
  mkdir trained_data
  mkdir tokenized_data
  mkdir trained_model
  ```

  > make sure to put data willing to train under the trained_data directory

- An example structure with the provided pretrained model and put the data will be like.

  <img src="media/sample-code-structure-release.png" alt="sample" style="zoom:80%;" />

  

### Modify the config

- Some codes may need to be modified for local use

- `train.py`

  ```python
  """ Metadata
  ...
  """
  # ...
  
  config = ProjectConfig(
  	...,
  	data_name="simplebooks-2"
  )
  ```

  > data_name can be modified

## Preprocessing

### BPE Tokenizer

- implement BPE tokenizer to pre-processing the text data

  - [Summary of the tokenizers (huggingface.co)](https://huggingface.co/docs/transformers/tokenizer_summary)
  - Aim's to translate between human-readable text and numeric indices
  - Indices will be mapped to word embeddings (numerical representations of words) -> This will be done by an embedding layer within the model. 

- Load the tokenizer

  - This Tokenizer will be loaded in model initialization

    ```python
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    ```

## Trainning

### TFGPT2LMHeadModel

- Use `transformers` to construct GPT Model

### History

- Stored history object will be used to visualize the trainning history.
- Use `matplotlib.pyplot` to visualized data

## Result

- This visualization result will be stored in `trained_model/figure/`

### Loss Value Per Epoch

<img src="media/samplebooks2-50-epoch.png" alt="loss per epoch" style="zoom:80%;" />

```
## Loss (Per Epoch)    ##
        Epoch 0 -> Loss: 3.903648853302002
        Epoch 1 -> Loss: 3.2500603199005127
        Epoch 2 -> Loss: 2.9946632385253906
        Epoch 3 -> Loss: 2.821579933166504
        Epoch 4 -> Loss: 2.6852798461914062
        Epoch 5 -> Loss: 2.5663347244262695
        Epoch 6 -> Loss: 2.4579238891601562
        Epoch 7 -> Loss: 2.3542587757110596
        Epoch 8 -> Loss: 2.251448154449463
        Epoch 9 -> Loss: 2.148042678833008
        Epoch 10 -> Loss: 2.043449878692627
        Epoch 11 -> Loss: 1.9365277290344238
        Epoch 12 -> Loss: 1.8293315172195435
        Epoch 13 -> Loss: 1.719415307044983
        Epoch 14 -> Loss: 1.6119022369384766
        Epoch 15 -> Loss: 1.5034215450286865
        Epoch 16 -> Loss: 1.3979310989379883
        Epoch 17 -> Loss: 1.2955776453018188
        Epoch 18 -> Loss: 1.1981788873672485
        Epoch 19 -> Loss: 1.1070220470428467
        Epoch 20 -> Loss: 1.0202499628067017
        Epoch 21 -> Loss: 0.9405309557914734
        Epoch 22 -> Loss: 0.8663972020149231
        Epoch 23 -> Loss: 0.7998376488685608
        Epoch 24 -> Loss: 0.7391602993011475
        Epoch 25 -> Loss: 0.6846773028373718
        Epoch 26 -> Loss: 0.6370853781700134
        Epoch 27 -> Loss: 0.5935728549957275
        Epoch 28 -> Loss: 0.5558265447616577
        Epoch 29 -> Loss: 0.5230922102928162
        Epoch 30 -> Loss: 0.4917517602443695
        Epoch 31 -> Loss: 0.46451228857040405
        Epoch 32 -> Loss: 0.4422658681869507
        Epoch 33 -> Loss: 0.4201878607273102
        Epoch 34 -> Loss: 0.4020017683506012
        Epoch 35 -> Loss: 0.38453686237335205
        Epoch 36 -> Loss: 0.3693166971206665
        Epoch 37 -> Loss: 0.356445848941803
        Epoch 38 -> Loss: 0.3433244228363037
        Epoch 39 -> Loss: 0.33187514543533325
        Epoch 40 -> Loss: 0.3205644190311432
        Epoch 41 -> Loss: 0.31170225143432617
        Epoch 42 -> Loss: 0.3019041121006012
        Epoch 43 -> Loss: 0.29318127036094666
        Epoch 44 -> Loss: 0.28581875562667847
        Epoch 45 -> Loss: 0.27874401211738586
        Epoch 46 -> Loss: 0.27176862955093384
        Epoch 47 -> Loss: 0.26560863852500916
        Epoch 48 -> Loss: 0.260423868894577
        Epoch 49 -> Loss: 0.253560870885849
```



### Loss Value Per Batch

<img src="media/samplebooks2-50-batch.jpg" alt="loss per batch" style="zoom:80%;" />

```
## Loss (Per Batch)    ##
        Batch 0 -> Loss: 9.497234344482422
        Batch 1000 -> Loss: 4.277572154998779
        Batch 2000 -> Loss: 3.9940218925476074
        Batch 3000 -> Loss: 3.3456242084503174
        Batch 4000 -> Loss: 3.321021318435669
        Batch 5000 -> Loss: 3.0202085971832275
        Batch 6000 -> Loss: 3.050510883331299
        Batch 7000 -> Loss: 3.0121395587921143
        Batch 8000 -> Loss: 2.8706321716308594
        Batch 9000 -> Loss: 2.873934030532837
        Batch 10000 -> Loss: 2.699038028717041
        Batch 11000 -> Loss: 2.7171552181243896
        Batch 12000 -> Loss: 2.691805839538574
        Batch 13000 -> Loss: 2.603060483932495
        Batch 14000 -> Loss: 2.5933454036712646
        Batch 15000 -> Loss: 2.5013160705566406
        Batch 16000 -> Loss: 2.4846417903900146
        Batch 17000 -> Loss: 2.4636943340301514
        Batch 18000 -> Loss: 2.390069007873535
        Batch 19000 -> Loss: 2.3672285079956055
        Batch 20000 -> Loss: 2.278238534927368
        Batch 21000 -> Loss: 2.288430690765381
        Batch 22000 -> Loss: 2.1553590297698975
        Batch 23000 -> Loss: 2.184856653213501
        Batch 24000 -> Loss: 2.155637741088867
        Batch 25000 -> Loss: 2.0815889835357666
        Batch 26000 -> Loss: 2.0744104385375977
        Batch 27000 -> Loss: 1.9860414266586304
        Batch 28000 -> Loss: 1.9614776372909546
        Batch 29000 -> Loss: 1.9377615451812744
        Batch 30000 -> Loss: 1.8658758401870728
        Batch 31000 -> Loss: 1.850908875465393
        Batch 32000 -> Loss: 1.771216869354248
        Batch 33000 -> Loss: 1.743480920791626
        Batch 34000 -> Loss: 1.6579285860061646
        Batch 35000 -> Loss: 1.6453077793121338
        Batch 36000 -> Loss: 1.6207081079483032
        Batch 37000 -> Loss: 1.5390323400497437
        Batch 38000 -> Loss: 1.531586766242981
        Batch 39000 -> Loss: 1.421772837638855
        Batch 40000 -> Loss: 1.4289096593856812
        Batch 41000 -> Loss: 1.402809739112854
        Batch 42000 -> Loss: 1.3266408443450928
        Batch 43000 -> Loss: 1.3127678632736206
        Batch 44000 -> Loss: 1.2425144910812378
        Batch 45000 -> Loss: 1.2159185409545898
        Batch 46000 -> Loss: 1.2372543811798096
        Batch 47000 -> Loss: 1.1357413530349731
        Batch 48000 -> Loss: 1.1169850826263428
        Batch 49000 -> Loss: 1.0511211156845093
        Batch 50000 -> Loss: 1.0351990461349487
        Batch 51000 -> Loss: 0.9733182787895203
        Batch 52000 -> Loss: 0.9627713561058044
        Batch 53000 -> Loss: 0.9453473091125488
        Batch 54000 -> Loss: 0.8898150324821472
        Batch 55000 -> Loss: 0.8794136047363281
        Batch 56000 -> Loss: 0.8232561945915222
        Batch 57000 -> Loss: 0.8125141263008118
        Batch 58000 -> Loss: 0.8010779619216919
        Batch 59000 -> Loss: 0.755605936050415
        Batch 60000 -> Loss: 0.7468817830085754
		...
        Batch 95000 -> Loss: 0.33387047052383423
        Batch 96000 -> Loss: 0.3343263864517212
        Batch 97000 -> Loss: 0.3236560523509979
        Batch 98000 -> Loss: 0.32374653220176697
        Batch 99000 -> Loss: 0.3211926817893982
        Batch 100000 -> Loss: 0.31454187631607056
        Batch 101000 -> Loss: 0.3139667510986328
        Batch 102000 -> Loss: 0.3027575612068176
        Batch 103000 -> Loss: 0.30423781275749207
        Batch 104000 -> Loss: 0.30198535323143005
        Batch 105000 -> Loss: 0.2958706021308899
        Batch 106000 -> Loss: 0.2944484353065491
        Batch 107000 -> Loss: 0.28725558519363403
        Batch 108000 -> Loss: 0.28654876351356506
        Batch 109000 -> Loss: 0.28274139761924744
        Batch 110000 -> Loss: 0.28014689683914185
        Batch 111000 -> Loss: 0.27918609976768494
        Batch 112000 -> Loss: 0.2728959619998932
        Batch 113000 -> Loss: 0.27407777309417725
        Batch 114000 -> Loss: 0.2634708285331726
        Batch 115000 -> Loss: 0.2673775851726532
        Batch 116000 -> Loss: 0.26590201258659363
        Batch 117000 -> Loss: 0.26176029443740845
        Batch 118000 -> Loss: 0.26142510771751404
        Batch 119000 -> Loss: 0.2556767165660858
        Batch 120000 -> Loss: 0.2547377049922943
```



## Reference

- [Text generation with GPT-2 - Model Differently](https://www.modeldifferently.com/en/2021/12/generación-de-fake-news-con-gpt-2/)

- [Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. (github.com)](https://github.com/huggingface/transformers)
- [TensorFlow API](https://www.tensorflow.org/api_docs/python/tf?hl=zh-tw)
- [Model training APIs (keras.io)](https://keras.io/api/models/model_training_apis/)
- [TFGPT2LMHeadModel (huggingface.co)](https://huggingface.co/docs/transformers/v4.19.4/en/model_doc/gpt2#transformers.TFGPT2LMHeadModel)
- [TFPreTrainedModel (huggingface.co)](https://huggingface.co/docs/transformers/v4.19.4/en/main_classes/model#transformers.TFPreTrainedModel)
- [tf.keras.callbacks.History  | TensorFlow Core v2.9.1](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
- [tf.keras.callbacks.Callback  | TensorFlow Core v2.9.1](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback)
- [Pipelines (huggingface.co)](https://huggingface.co/docs/transformers/v4.19.4/en/main_classes/pipelines#transformers.TextGenerationPipeline)
- [Difference between Sparse Cross Entropy and Categorical Cross Entropy](https://ithelp.ithome.com.tw/articles/10271081)

- [gpt2 · Hugging Face](https://huggingface.co/gpt2)
- [Visualize the hyperparameter tuning process (keras.io)](https://keras.io/guides/keras_tuner/visualize_tuning/)
- [python - How to disable printing reports after each epoch in Keras? - Stack Overflow](https://stackoverflow.com/questions/44931689/how-to-disable-printing-reports-after-each-epoch-in-keras)
- [Module: tf.keras.metrics  | TensorFlow Core v2.9.1](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
- [machine learning - What does from_logits=True do in SparseCategoricalcrossEntropy loss function? - Data Science Stack Exchange](https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do-in-sparsecategoricalcrossentropy-loss-function)
- [How to add some new special tokens to a pretrained tokenizer?](https://github.com/huggingface/tokenizers/issues/247)

