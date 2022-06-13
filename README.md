# AI Final Project

## Proposal

- The main proposal is to use the GPT-2 model trained & generating text

## Code Structure



## Preprocessing

### BPE Tokenizer

- implement BPE tokenizer to pre-processing the text data

  - [BPE tokenizer]([Summary of the tokenizers (huggingface.co)](https://huggingface.co/docs/transformers/tokenizer_summary))
  - Aim's to translate between human-readable text and numeric indices
  - Indices will be mapped to word embeddings (numerical representations of words) -> This will be done by an embedding layer within the model. 

- Load the tokenizer

  - This Tokenizer will be loaded in model initialization

    ```python
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    ```

## Reference

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

