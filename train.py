import os
import tensorflow as tf
from transformers import GPT2Config, GPT2Tokenizer, TextGenerationPipeline
from src.config import ProjectConfig
from src.model import TextModel
import matplotlib.pyplot as plt

""" Metadata
1. config
    - Store the configuration about this project
2. gpt_config
    - Aim to have the GPT configuration to build GPT2Model
    - It will be initialized after loading tokenizer
3. set_seed
    - For, test. Ensure the statistic of behavior.
"""
print("==> Setting Metadata:")
config = ProjectConfig(
    block_size=100,
    batch_size=12,
    buffer_size=1000,
    data_name="simplebooks-2",
    epoch_times=1
)
gpt_config : GPT2Config
# Set seed for static behavior
tf.random.set_seed(42)
print("\tMetadata set...")


def load_tokenizer(type: str) -> GPT2Tokenizer:
    """ Load tokenizer
    - load tokenizer from saved path
    """
    # Load tokenizer
    token_path = config.token_pos if type == "train" else config.test_token_pos
    BPE_tokenizer = GPT2Tokenizer.from_pretrained(token_path)
    # Add special tokens to tokenizer
    BPE_tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    })
    return BPE_tokenizer


def make_dataset(tokenizer: GPT2Tokenizer, type: str) -> tf.data.Dataset:
    """ Make Dataset
    1. Encode the text by tokenizer.
    2. Slice tokenized data to blocks
    3. Make dataset 
        - shuffle:
            Randomly shuflles the elements of the dataset,
            which will randomly samples the elements from buffer, 
            replacing the selected elements with new elements.
        - batch:
            Combines elements of the dataset into batches.
            Since this program depends on the batches to have the same outer dimension,
            set the drop_remainder argument to `True`, preventing the smaller batch from 
            being produced.
    """
    # Use tokenizer to encode the data as single string
    data_tokenized : str
    data_pos = config.train_pos if type == "train" else config.test_pos
    with open(data_pos, "r", encoding='utf-8') as f:
        tmp = f.read()
        tmp = tmp.replace("\n", " ")    # clean up the change line characters
    # Add token, this can be done when having multiple input files or text
    # add_special_tokoen = tokenizer.bos_token + tmp + tokenizer.eos_token
    data_tokenized = tokenizer.encode(tmp)
    print("\tText Encoded...")
    
    # Slice data with equal quantity
    examples, inputs, labels = [], [], []
    for i in range(0, len(data_tokenized) - config.block_size + 1, config.block_size):
        examples.append(data_tokenized[i:i + config.block_size])
    for ex in examples:
        inputs.append(ex[:-1])
        labels.append(ex[1:])

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    # Shuffles & Batch
    dataset = dataset.shuffle(buffer_size=config.buffer_size) \
        .batch(config.batch_size, drop_remainder=True)
    return dataset


def combination_visualize(train_model: TextModel, test_model: TextModel):
    """ Combinational visualization
    - Provide combinational visualized function of test model and train model
    """
    # Per epoch
    plt.figure(figsize=(10, 5))
    plt.plot(train_model.history.history['loss'], color='red', label='train')
    plt.plot(test_model.history.history['loss'], color='blue', label='test')
    plt.legend(loc='upper left')
    plt.title('Comparison of test & train (Per epoch)')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f"{config.pltfigure_pos}/{config.data_name}-{config.epoch_times}-train-test-epoch.png")
    plt.show()
    plt.close()
    # Per batch
    plt.figure(figsize=(10, 5))
    plt.plot(train_model.batch_end_loss[:len(test_model.batch_end_loss)], color='red', label='train')
    plt.plot(test_model.batch_end_loss, color='blue', label='test')
    plt.legend(loc='upper left')
    plt.title('Comparison of test & train (Per batch)')
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.savefig(f"{config.pltfigure_pos}/{config.data_name}-{config.epoch_times}-train-test-batch.png")
    plt.show()
    plt.close()


def main():
    ## Load tokenizer   ##
    print("==> Loading tokenizer:")
    BPE_tokenizer = load_tokenizer(type='train')
    test_BPE_tokenizer = load_tokenizer(type='test')
    print("\tTokenizer loaded...")

    ## Make dataset     ##
    print("==> Making dataset:")
    train_dataset = make_dataset(tokenizer=BPE_tokenizer, type="train")
    test_dataset = make_dataset(tokenizer=test_BPE_tokenizer, type="test")
    print("\tDataset made...")

    ## Init Model       ##
    print("==> Init Model:")
    train_model = TextModel(config=config, tokenizer=BPE_tokenizer, model_name="train")
    test_model = TextModel(config=config, tokenizer=test_BPE_tokenizer, model_name="test")
    print("\tModel initialized...")

    ## Train Model      ##
    print("==> Trainning Model:")
    train_model.train(dataset=train_dataset)
    test_model.train(dataset=test_dataset)
    print("\tModel trained...")

    ## Visualized       ##
    print("==> Visualizing:")
    # Per Model Visualization
    train_model.visualize()
    test_model.visualize()
    print("\tPer Model visualized...")
    # Combinational Visualization
    combination_visualize(train_model=train_model, test_model=test_model)
    print("\tComparison visualized...")
    print("\tModel visualized...")

    ## Output Training Result ##
    print("==> Training Result:")
    train_model.trainning_output()
    test_model.trainning_output()
    print("\tTraining result output....")


    """ Text Generating
    Make prediction by using text generating function
    -> This blank is for test
    """
    text = "Did you hear that ?"
    tokenizer = BPE_tokenizer
    text_generator = TextGenerationPipeline(train_model.model, tokenizer)
    print (f'''Result: 
        {text_generator(
            text_inputs=text,
            max_length=128,
            do_sample=True,
            top_k=10,
            eos_token_id=tokenizer.get_vocab().get("</s>", 0)
        )[0]['generated_text']}
        '''
    )


if __name__ == "__main__":
    main()