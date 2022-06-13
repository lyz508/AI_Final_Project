import os
import tensorflow as tf
from transformers import GPT2Config, GPT2Tokenizer, TextGenerationPipeline
from src.config import ProjectConfig
from src.model import TextModel

""" Metadata
1. config
    - Store the configuration about this project
2. gpt_config
    - Aim to have the GPT configuration to build GPT2Model
    - It will be initialized after loading tokenizer
3. set_seed
    - For, test. Ensure the statistic of behavior.
"""
print("Setting Metadata:")
config = ProjectConfig(
    block_size=100,
    batch_size=12,
    buffer_size=1000,
    data_name="simplebooks-2",
    epoch_times=50
)
gpt_config : GPT2Config
# Set seed for static behavior
tf.random.set_seed(42)
print("\tMetadata set...")


def load_tokenizer() -> GPT2Tokenizer:
    """ Load tokenizer
    - load tokenizer from saved path
    """
    # Load tokenizer
    BPE_tokenizer = GPT2Tokenizer.from_pretrained(config.token_pos)
    # Add special tokens to tokenizer
    BPE_tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    })
    return BPE_tokenizer


def make_dataset(tokenizer: GPT2Tokenizer) -> tf.data.Dataset:
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
    with open(config.train_pos, "r", encoding='utf-8') as f:
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

def main():
    ## Load tokenizer   ##
    print("==> Loading tokenizer:")
    BPE_tokenizer = load_tokenizer()
    print("\tTokenizer loaded...")

    ## Make dataset     ##
    print("==> Making dataset:")
    dataset = make_dataset(BPE_tokenizer=BPE_tokenizer)
    print("\tDataset made...")

    ## Init Model       ##
    print("==> Init Model:")
    model = TextModel(config=config, tokenizer=BPE_tokenizer)
    print("\tModel initialized...")

    ## Train Model      ##
    print("==> Trainning Model:")
    model.train(dataset=dataset)
    print("\tModel trained...")

    ## Visualized       ##
    print("==> Visualizing:")
    model.visualize()
    print("\tModel visualized...")

    ## Trainning Result ##
    print("==> Train Result:")
    model.trainning_output()
    print("\tTrain result output....")

    """ Text Generating
    Make prediction by using text generating function
    -> This blank is for test
    """
    text = "Did you hear that ?"
    tokenizer = BPE_tokenizer
    text_generator = TextGenerationPipeline(model.model, tokenizer)
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