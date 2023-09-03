import logging
import random
from random import randrange
import datasets
import nltk
import numpy as np
import torch
import evaluate
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
import transformers

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}

def setup_logging(accelerator):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
    return logger

def train(config):
    accelerator = Accelerator()
    logger = setup_logging(accelerator)
    output_dir = config['output_dir']
    logger.info(accelerator.state)

    raw_datasets = load_dataset(config['dataset_name'])
    model_ckpt = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(config['model_ckpt'])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSeq2SeqLM.from_pretrained(config['model_ckpt']).to(device)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = ""

    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get('samsum', None)
    text_column_name = dataset_columns[0] if dataset_columns is not None else column_names[0]

    summary_column = 'summary'


    def preprocess_function(examples):
        inputs = examples[text_column_name]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=1024, padding=config['padding'], truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=config['max_target_length'], padding=config['padding'], truncation=True)

        if config['padding'] == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=column_names
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=config["batch_size"]
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=config["batch_size"])

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config["learning_rate"])

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = config['num_train_epochs'] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config["num_warmup_steps"]
    )

    metric = evaluate.load("rouge")

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config['num_train_epochs']}")

    progress_bar = tqdm(range(max_train_steps))
    completed_steps = 0

    for epoch in range(config['num_train_epochs']):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch.to(device))
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

            if completed_steps >= max_train_steps:
                break

        model.eval()

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **config['gen_kwargs'],
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        result = metric.compute(use_stemmer=True)

        # Extract a few results from ROUGE
        result = {key: value * 100 for key, value in result.items()}

        result = {k: round(v, 4) for k, v in result.items()}

        logger.info(result)

    if output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

    # Model inference with transformers.pipeline

    summarizer = transformers.pipeline(
        "summarization",
        model=unwrapped_model,
        tokenizer=tokenizer,
        max_length=100,
        device=3
    )

    sample = raw_datasets['test'][randrange(len(raw_datasets["test"]))]

    print(f"dialogue: \n{sample['dialogue']}\n---------------")

    # summarize dialogue
    res = summarizer(sample["dialogue"])

    print(f"Our bart-samsum-model summary:\n{res[0]['summary_text']}")


if __name__ == '__main__':
    # Configuration Dictionary
    config = {
        "output_dir": "bart-samsum-model",
        "model_ckpt": "facebook/bart-large-cnn",
        "dataset_name": "samsum",
        "num_train_epochs": 1,
        "max_target_length": 128,
        "padding": "max_length",
        "preprocess_max_length": 1024,
        "learning_rate": 5e-5,
        "batch_size": 1,
        "num_warmup_steps": 10,
        "gen_kwargs": {
            "max_length": 128,
            "num_beams": 2,
        },
    }

    train(config)
