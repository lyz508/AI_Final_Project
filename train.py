from matplotlib.pyplot import hist
import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel
from transformers import TFGPT2LMHeadModel
from transformers import BertTokenizer
from transformers import GPT2Config
import configs
from transformers import TextGenerationPipeline
import time
import pickle


max_length = configs.model.max_length


def load_tokenizer() -> BertTokenizer:
    tokenizer = BertTokenizer.from_pretrained(
        configs.data.path, max_len=max_length-1)
    return tokenizer


def get_dataset() -> tf.data.Dataset:
    print(f"loading {configs.data.pickle}")
    ids, labels = pickle.load(open(configs.data.pickle, 'rb'))
    print(ids[:5])
    print(labels[:5])
    print(ids.shape, labels.shape, ids.dtype, labels.dtype)
    dataset = tf.data.Dataset.from_tensor_slices((
        ids,
        labels
    )).shuffle(100000, reshuffle_each_iteration=True).batch(configs.model.batch_size)
    return dataset


def init_model(tokenizer) -> TFGPT2LMHeadModel:

    try:
        model = TFGPT2LMHeadModel.from_pretrained(
            f'{configs.model_path}', return_dict=True)
    except EnvironmentError:
        config = GPT2Config(
            architectures=["TFGPT2LMHeadModel"],
            vocab_size=tokenizer.vocab_size,
            n_positions=configs.model.n_positions,
            n_ctx=configs.model.n_ctx,
            n_embd=configs.model.n_embd,
            n_layer=configs.model.n_layer,
            n_head=configs.model.n_head,
            pad_token_id=tokenizer.pad_token_id,
            task_specific_params={
                "text-generation": {
                    "do_sample": True,
                    "max_length": 120
                }
            },
        )
        model = TFGPT2LMHeadModel(config)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    ]

    model.compile(
        optimizer=optimizer,
        loss=[loss, *[None] * model.config.n_layer],
        # metrics=metrics,
    )

    return model


def train():
    tokenizer = load_tokenizer()
    train_dataset = get_dataset()
    model = init_model(tokenizer)

    text_generator = TextGenerationPipeline(model, tokenizer)

    class AutoSaveCallback(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs=None):
            if (batch + 1) % 2000 == 0:
                self.model.save_pretrained(f'{configs.model_path}{batch}')

        def on_epoch_end(self, epoch, logs=None):
            self.model.save_pretrained(f'{configs.model_path}')
    class ShowLossCallback(tf.keras.callbacks.Callback):
            interval = 100
            counter = 0
            epoch = 0
            # begin of one epoch
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch = epoch
                self.counter = 0
            def on_train_batch_end(self, batch, logs=None):
                if self.counter % self.interval == 0:
                    print(f"Epoch: {self.epoch+1} \n\tBatch: {self.counter}/{train_dataset.__len__()} \n\tLoss: {str(logs['loss'])}")
                self.counter += 1


    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath='./models',
                                           save_weights_only=True),
        AutoSaveCallback(),
        ShowLossCallback()
    ]

    t1 = time.time()
    history = model.fit(train_dataset, epochs=50, callbacks=callbacks, verbose=0, batch_size=12)
    print(history.history['loss'])


    print(text_generator("Did you hear that ?", max_length=int(max_length / 2), do_sample=False))
    print(f'total train time {t1 - time.time()}')

    model.save_pretrained(configs.model_path)


if __name__ == '__main__':
    train()
