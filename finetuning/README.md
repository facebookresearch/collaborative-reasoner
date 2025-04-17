# Llama Finetuning Pipeline using fairseq2

Following this readme, you learn to:
* Prepare the data in the way that `fairseq2` wants
* Finetune Llama-3.1 8B and 70B models on such data
* Tune different hyperparameters for the finetuning

Good things for `fairseq2`
* Configs are easy to read and can be organized as yaml files with overriding setups
* Tested and running on AWS & model checkpoints already on AWS
* Supports recipes for SFT and preference tuning
* FSDP + tensor parallel, GPU utilization ~100%

## Fairseq2 Workflow

### Installation
Follow the [fairseq2 doc here](https://facebookresearch.github.io/fairseq2/stable/getting_started/installation/index.html) for installation.

### Finetuning Configs
`fairseq2` has a variety of built-in recipes, including instruction-tuning (what we use)
here, preference-tuning, etc.

You can write the configs in a series yamls, merging from left to right, and you can
also put `k=v` pairs in the same command for overriding as well.

You can see all the options from [fairseq2/src/fairseq2/recipes/lm/instruction_finetune.py](https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/recipes/lm/instruction_finetune.py), you can also find relevant docs on preference finetuning as well.


### Create the training data
`fairseq2` expects training/val data in the following format in a `*.jsonl` file:
```json
{"src": "...", "tgt": "..."}
```

> [!IMPORTANT]
> By default, fairseq2 will add the special tokens for you, that includes the BOS token at the beginning of the src, and the EOS token at the end of the tgt. You can customize the behavior, for more check out [this PR](https://github.com/facebookresearch/fairseq2/pull/846).

You should create a dir to store the training and validation data, for example:
```bash
mkdir /home/$USER/llama_data
```

Then you can put the training and validation data like this:
```text
llama_data
├── train/train.jsonl
└── val/val.jsonl
```

And in the config file, add the following two lines to indicate the folder for train/val:
```yaml
train_split: train
valid_split: val
```

### Start Training
Check the example configs in `finetuning/configs/sft.yaml` for the basic SFT config, and
remember to (at least) change the following fields:
* dataset
* wandb_project

> [!TIP]
> When finetuning for a specific model size, the configs `sft.yaml` and `sft_[70b|8b].yaml`
> will be used in tandem and the configs in `sft_[70b|8b].yaml` will override `sft.yaml`

We have also written a python script to bypass all the slurm configs to give you an
easier way to start finetuning, here is an example command:
```bash
python finetuning/run_fairseq2_training.py \
    --exp_name test_8B_dpo \
    --size 8b \
    --data /home/$user/llama_data/ \
    --task mmlu_pro \
    --save_dir /home/$user/coral_results \
    --training_method dpo
```

### Serving the finetuned model with vLLM for inference/eval
As of `vllm>=0.7.3`, we no longer need to convert the `fairseq2` ckpts into huggingface
format for it to be served in vllm, however, you do need to copy the corresponding
tokenizers from the local huggingface copy. And you need to feed it into an env var
`$tokenizer_dir` (see L31 of `finetuning/convert_ckpt.sh`).

Your `$tokenizer_dir` should look like this:
```
$tokenizer_dir
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json
```

Run `bash finetuning/convert_ckpt.sh <size> <ckpt_dir>` to "convert" the ckpt, which
will create a new folder named `*_hf_converted`.

And you can use something like `step_500_hf_converted` folder as a dir for vLLM to serve:
```bash
vllm serve -c .../step_500_hf_converted
```

Or if you are using [matrix](https://github.com/facebookresearch/matrix), you can
directly add it to the serving applications list, check the
[docs](https://github.com/facebookresearch/matrix?tab=readme-ov-file#incremental-deployment) for details.
