from datasets import load_dataset
import datasets
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import AutoTokenizer


def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    # {'en': 'That is because at present the disparity in wealth between the regions is huge, even exceeding a disparity of 10:1.', 'ro': 'Aceasta se datorează faptului că, în prezent, există discrepanţe enorme între regiuni în ceea ce priveşte nivelul de bogăţie, depăşindu-se chiar raportul de 10:1.'}
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))


if __name__ == '__main__':
    raw_datasets = load_dataset("wmt16", "ro-en")
    # show_random_elements(raw_datasets["train"])
    model_checkpoint = "Helsinki-NLP/opus-mt-en-ro"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    if "mbart" in model_checkpoint:
        tokenizer.src_lang = "en-XX"
        tokenizer.tgt_lang = "ro-RO"
    with tokenizer.as_target_tokenizer():
        print(tokenizer("Hello, this one sentence!"))
        model_input = tokenizer("Hello, this one sentence!")
        tokens = tokenizer.convert_ids_to_tokens(model_input['input_ids'])
        # 打印看一下special toke
        print('tokens: {}'.format(tokens))
    # {'input_ids': [10334, 1204, 3, 15, 8915, 27, 452, 59, 29579, 581, 23, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    # tokens: ['▁Hel', 'lo', ',', '▁', 'this', '▁o', 'ne', '▁se', 'nten', 'ce', '!', '</s>']
    if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
        prefix = "translate English to Romanian: "
    else:
        prefix = ""

    max_input_length = 128
    max_target_length = 128
    source_lang = "en"
    target_lang = "ro"

    def preprocess_function(examples):
        inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


