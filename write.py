from transformers import TFGPT2LMHeadModel, TextGenerationPipeline
from train import load_tokenizer
from train import config

def main():
    model_path = f"{config.model_pos}-30"
    tokenizer = load_tokenizer()
    model = TFGPT2LMHeadModel.from_pretrained(model_path)

    print(f"By default, it will trace use the model stored in -> {model_path} \
        It is possible to use the --d option to specify the model directory")
    text = input("Input: ")
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

if __name__ == '__main__':
    main()