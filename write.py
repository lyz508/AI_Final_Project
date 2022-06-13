import argparse
from transformers import TFGPT2LMHeadModel, TextGenerationPipeline
from train import load_tokenizer
from train import config

def param_parser():
    """ Param_parser
    - Using argparse module
    - Parse argument
        - dir: model path
        - max_len: output length
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default=f"{config.model_pos}-30")
    # parser.add_argument('-h', '--help', default=False)
    parser.add_argument('-max', '--max_len', default=128)
    args = parser.parse_args()
    return args


def main():
    """ main
    1. Show welcome prompt
    2. Parse Arguments
    3. Load Model & Tokenizer
    4. Accept input string
    5. Generate Text
    """
    # Variables
    model_path = f"{config.model_pos}-30"
    max_len = 128

    # Prompt
    print("## Write.py      ##")
    print(f"\tBy default, it will trace use the model stored in -> {model_path} \n\tIt is possible to use the --d option to specify the model directory")

    # Parsing Argument
    print("==> Parsing Argument:")
    args = param_parser()
    if args.dir != model_path:
        model_path = args.dir
        print(f"==> Set model path: {model_path}")
    if args.max_len != max_len:
        max_len = args.max_len
    print("\tArgument parsed...")

    # Load Model & Tokenizer 
    print("==> Loading Model & Tokenizer:")
    try:
        tokenizer = load_tokenizer()
        print("\tTokenizer Loaded...")
        model = TFGPT2LMHeadModel.from_pretrained(model_path)
    except OSError:
        print(f"The direcoty of default model path: {model_path} may not exist.")
        print ("\tUsing `python write.py --d <dir>` to set model path if you need.")
        exit(255)
    print("\tModel loaded...")
    
    # Accept Input
    print("## Input Start Sequence      ##")
    text = input("Input: ")
    text_generator = TextGenerationPipeline(model, tokenizer)
    print("\tText generator generated...")

    # Generate Text
    print ("==> Generating Text:")
    print(f'''## Result        ##
        {text_generator(
            text_inputs=text,
            max_length=max_len,
            do_sample=True,
            top_k=10,
            eos_token_id=tokenizer.get_vocab().get("</s>", 0)
        )[0]['generated_text']}
        '''
    )
    print ("\tText Generated...")


if __name__ == '__main__':
    main()