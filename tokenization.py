import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


# Byte pair encoding tokenizor
class tokenization(object):
    def __init__(self):
        # Initialoze tokenizer
        self.tokenizer = Tokenizer(BPE())
        # Provide the equality for unicode charactors
        # It may seen same words in different forms as equal
        self.tokenizer.normalizer = Sequence([
            # Normalization Form Compatibility Composition
            NFKC(),
            Lowercase()
        ])
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()

    # trained BPE model and output the vocab size
    def train(self, location="."):
        # Add special tokens 
        trainer = BpeTrainer(vocab_size=50000,
            show_progress=True,
            inital_alphabet=ByteLevel.alphabet(),
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>"
            ]
        )
        # Specify the file location to avoid TypeError
        self.tokenizer.train(trainer=trainer, files=location)
        print (f"trained vocab size: {self.tokenizer.get_vocab_size()}")

    def save(self, location, prefix=None):
        if not os.path.exists(location):
            os.makedirs(location)
        self.tokenizer.model.save(location, prefix)


# for generating tokenizer
if __name__ == "__main__":
    data_name = "simplebooks-2"

    # allocate path
    text_path = [f"trained_data/{data_name}/train.txt"]
    save_path = f"tokenized_data/{data_name}/"
    
    # Init & train tokenizer
    tokenizer = tokenization()
    tokenizer.train(text_path)

    # Save the tokenized data
    tokenizer.save(save_path)