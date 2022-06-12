# AI Final Project

## Proposal

- The main proposal is to use the GPT-2 model trained & generating text
  - 

- Compare the self-trained model with the pre-trained model



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

- ref
  - [BPE simple implementation](https://blog.csdn.net/qq_44574333/article/details/110749997)

## Reference

- [Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. (github.com)](https://github.com/huggingface/transformers)
- [TensorFlow API](https://www.tensorflow.org/api_docs/python/tf?hl=zh-tw)
- [gpt2 · Hugging Face](https://huggingface.co/gpt2)
- [预训练模型专题_GPT2_模型代码学习笔记](https://blog.csdn.net/qq_35128926/article/details/111399679)

- [自然语言处理时，通常的文本清理流程是什么？](https://www.zhihu.com/question/268849350/answer/488000403)

  ```
  Normalization
  Tokenization
  Stop words
  Part-of-Speech Tagging
  Named Entity Recognition
  Stemming and Lemmatization
  ```

- [How to add some new special tokens to a pretrained tokenizer?](https://github.com/huggingface/tokenizers/issues/247)