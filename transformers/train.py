import torch
import torch.nn as nn
from config import ModelConfig
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from model_config import ModelArgs, build_transformer
from dataset import BilingualDataset
from torch.utils.data import Dataset, DataLoader, random_split

import warnings
from tqdm import tqdm
import os
from pathlib import Path
from config import *

# Tokenizer

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config: ModelConfig, ds, lang):
    tokenizer_path = Path(config.tokenizer_file.format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

# Model and datasets
def get_model(config: ModelConfig, vocab_src_len, vocab_tgt_len):
    model_args = ModelArgs(
        vocab_src_len,
        vocab_tgt_len,
        config.seq_len,
        config.seq_len,
        config.d_model
    )
    return build_transformer(model_args)

def get_ds(config: ModelConfig):
    dataset = load_dataset(config.datasource, f"{config.lang_src}-{config.lang_tgt}", split='train')
    # print(len(dataset))

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset, config.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config, dataset, config.lang_tgt)

    # Train/test splits
    train_ds_size = int(0.9 * len(dataset))
    val_ds_sisze = len(dataset) - train_ds_size
    train_ds, val_ds = random_split(dataset, [train_ds_size, val_ds_sisze])

    train_ds = BilingualDataset(train_ds, tokenizer_src, tokenizer_tgt, config.lang_src, config.lang_tgt, config.seq_len)
    val_ds = BilingualDataset(val_ds, tokenizer_src, tokenizer_tgt, config.lang_src, config.lang_tgt, config.seq_len)

    # print(len(train_ds), len(val_ds))

    max_len_src = 0
    max_len_tgt = 0
    for data in dataset:
        data = data['translation']
        tokens_src = tokenizer_src.encode(data[config.lang_src]).ids
        tokens_tgt = tokenizer_tgt.encode(data[config.lang_tgt]).ids
        max_len_src = max(max_len_src, len(tokens_src))
        max_len_tgt = max(max_len_tgt, len(tokens_tgt))
    
    # print(max_len_src, max_len_tgt)
    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    # double check this
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

# Inference and eval
def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    # print(eos_idx, sos_idx)
    encoder_output = model.encode(source, source_mask)
    # print(f"encoder output is {encoder_output.shape}")
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            # self, tgt, encoder_output, src_mask, tgt_mask
            break

        # build mask for the target
        decoder_mask = BilingualDataset.causal_mask(decoder_input.size(1)).type_as(source_mask).unsqueeze(0).to(device)
        # print(decoder_input.shape, decoder_mask.shape)
        # (1, seq_len, d_model)
        decoder_output = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
        prob = model.project(decoder_output[:, -1])
        # print(decoder_output.shape, prob.shape)
        _, next_token = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.ones(1, 1).fill_(next_token.item()).type_as(source).to(device)], dim=1
        )
        # print(f"new decoder input is {decoder_input.shape}")

        if next_token.item() == eos_idx:
            break
    
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_tgt, max_len, device, global_step=-1, writer=None, num_examples=2):
    # model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in validation_ds:
            # batch size is 1 here
            # print(batch.keys())
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (b, 1, 1, seq_len)
            # print(encoder_input.shape, encoder_mask.shape)

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)
            # print(model_out)

            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_txt'][0]
            model_out_text = tokenizer_tgt.decode(model_out.tolist())
            # print(model_out_text)

            source_texts.append(src_text)
            expected.append(tgt_text)
            predicted.append(model_out_text)

            if count == num_examples:
                print(source_texts)
                print(expected)
                print(predicted)
                break
    
    if writer:
        # Log metrics
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        metric = torchmetrics.BLEUScore()
        wer = metric(predicted, expected)
        writer.add_scalar('validation bleu', wer, global_step)
        writer.flush()


# Train the model
def train_model(config: ModelConfig):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() or torch.backend.mps.is_available() else 'cpu'
    device = torch.device(device)
    # print(device)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    Path(f"{config.datasource}_{config.model_folder}").mkdir(parents=True, exist_ok=True)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # tensorboard
    writer = SummaryWriter(config.experiment_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)

    # preload the model
    initial_epoch = 0
    global_step = 0
    preload = config.preload
    # print(preload)

    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    # print(model_filename)

    if model_filename:
        print(f"preloading model: {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        # return model
    else:
        print('No model to preload')
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config.num_epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # [B, seq_len]
            decoder_input = batch['decoder_input'].to(device) # [B, seq_len]
            encoder_mask = batch['encoder_mask'].to(device) # [B, 1, 1, seq_len]
            decoder_mask = batch['decoder_mask'].to(device) # [B, 1, seq_len, seq_len]
            label = batch['label'].to(device) # [B, seq_len]

            encoder_output = model.encode(encoder_input, encoder_mask) # [B, seq_len, d_model]
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # [B, seq_len, d_model]
            proj_out = model.project(decoder_output) # [B, seq_len, vocab_size]

            # print(encoder_input.shape, decoder_input.shape, encoder_mask.shape, decoder_mask.shape,
            #       label.shape, encoder_output.shape, decoder_output.shape, proj_out.shape, tokenizer_tgt.get_vocab_size())
            
            proj_out = proj_out.view(-1, tokenizer_tgt.get_vocab_size())
            label = label.view(-1)
            # print(proj_out.shape, label.shape)
            # print(proj_out[0].sum()) unnormalized as we are just projecting, loss function expects unnormalized
            # print(proj_out[0, :10], label[0]) # each proj out has probs, while labels are class labels

            loss = loss_fn(proj_out, label)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            # break
        
        # run validation
        run_validation(model, val_dataloader, tokenizer_tgt, config.seq_len, device, global_step, writer)

        # save model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        print(model_filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        # break

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)