import os
import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer, TextGenerationPipeline
from tokenization import tokenization
from config import ProjectConfig

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
    epoch_times=10
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


def make_dataset(BPE_tokenizer: GPT2Tokenizer) -> tf.data.Dataset:
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
    single_string = tmp + BPE_tokenizer.eos_token
    data_tokenized = BPE_tokenizer.encode(tmp)
    print("\tText Encoded...")

    examples, inputs, labels = [], [], []
    # Slice data with equal quantity
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


def init_model(tokenizer: tokenization):
    """ Train
    1. Initialize the model
    2. Define
        - optimizer
            Choose adam optimizer.
            Control the clipnorm parameters to clip the gradient.
        - loss -> Using SparseCategoricalCrossentropy
            Since there will be only one attr in the dataset,
            use sparse categorial crossentropy.
            Set from_logits to True may improve numerical stablility.
        - metric
    3. Compile into the model
    """
    # initial the Model
    gpt_config = GPT2Config(
        architectures=["TFGPT2LMHeadModel"],
        model_type="TFGPT2LMHeadModel",
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    model = TFGPT2LMHeadModel(config=gpt_config)

    # Define
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-05, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    # compiling the model
    model.compile(
        optimizer=optimizer, 
        loss=[loss, *[None] * model.config.n_layer], 
        metrics=[metric]
    )

    return model


def train(model: TFGPT2LMHeadModel, dataset: tf.data.Dataset):
    """ Train model & Save it to the path
    1. Set callback function to store model per epoch
        - inherited from  tf.keras.callbacks.Callback
        - override to save model in config.model_pos
    2. Check directories for tfboard and saved_model existed 
    3. Use TensorBoard to visualize the result 
    """
    class SaveCallback (tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            self.model.save_pretrained(f"{config.model_pos}-{epoch}")

    # Check directory existed
    if not os.path.exists(config.model_pos):
        os.makedirs(config.model_pos)
    if not os.path.exists(config.tfboard_pos):
        os.makedirs(config.tfboard_pos)

    model.fit(
        dataset, 
        epochs=config.epoch_times,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=f"{config.model_pos}/logs"
            ),
            SaveCallback(),
        ]
    )
    return model


def main():
    ## Load tokenizer   ##
    print("Loading tokenizer:")
    BPE_tokenizer = load_tokenizer()
    print("\tTokenizer loaded...")

    ## Make dataset     ##
    print("Making dataset:")
    dataset = make_dataset(BPE_tokenizer=BPE_tokenizer)
    print("\tDataset made...")

    ## Init Model       ##
    print("Init Model:")
    model = init_model(tokenizer=BPE_tokenizer)
    print("\tModel initialized...")

    ## Train Model      ##
    print("Trainning Model:")
    model = train(model=model, dataset=dataset)
    print("\tModel trained...")
    

    """ Text Generating
    Make prediction by using text generating function
    """
    text = "Did you hear that ?"
    tokenizer = BPE_tokenizer
    text_generator = TextGenerationPipeline(model, tokenizer)
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