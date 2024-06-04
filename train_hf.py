"""
To run on a single GPU small debug run, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8
"""

from copy import copy
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from tokenizer import Tokenizer
from utils import print_trainable_parameters, create_llama_pause_tokenizer

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
eval_interval = 100
log_interval = 1
eval_iters = 100
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "hf"  # 'scratch'|'resume'|'hf'
dataset = "math"  # "tiny" | "pajama" | "ultra_chat" | "math"
# wandb logging
wandb_log = True  # disabled by default
wandb_project = "pause meta_meta_qa"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# data
batch_size = 2  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 2048
# model
hf_path = "TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T"
use_flash_attention_2 = True
# LoRA
lora_training = True
# adamw optimizer
gradient_accumulation_steps = 16  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
max_iters = 100000  # total number of training iterations
weight_decay = 0
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 200  # how many steps to warm up for
# system
device = "cuda"
dtype = "bfloat16"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# various inits, derived attributes, I/O setup
seed_offset = 0

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model = AutoModelForCausalLM.from_pretrained(
    hf_path,
    use_flash_attention_2=True,
    torch_dtype=ptdtype,
)
tokenizer, pause_token_id = create_llama_pause_tokenizer()
model.resize_token_embeddings(len(tokenizer))

model.train()
model.to(device)

# Apply LoRA
if lora_training:
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

max_seq_len = model.config.max_position_embeddings
tokens_per_iter = gradient_accumulation_steps * batch_size * max_seq_len
print(f"tokens per iteration will be: {tokens_per_iter:}")
print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {batch_size} batch size * {max_seq_len} max seq len")

# task-specific setup
if dataset == "tiny":
    from tinystories import Task
elif dataset == "pajama":
    from slim_pajama import Task
elif dataset == "ultra_chat":
    from ultra_chat import Task
elif dataset == "math":
    from metamathqa import Task

task = Task

iter_batches = partial(
    task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    pause_token_id=pause_token_id,
    device=device,
    num_workers=0,
)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2))

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    loss_dict = {}
    model.eval()
    for split in ["train", "test"]:
        batch_iter = iter_batches(split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                with torch.no_grad():
                    out = model(input_ids=X, labels=X)
                    loss = out.loss
            losses[k] = loss.item()
        loss_dict[split] = losses.mean()
    model.train()
    return loss_dict

@torch.no_grad()
def generate_sample(max_new_tokens=150, temperature=1.0, top_k=5):
    model.eval()
    start_ids = tokenizer.encode("What is the value of the ceiling function applied to -2.4?")
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    with ctx:
        with torch.no_grad():
            kwargs = {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_k": top_k}
            y = model.generate(input_ids=x, **kwargs)
    
    decoded = tokenizer.decode(y[0].tolist())
    decoding_table.add_data("", decoded)
    model.train()
    
def calculate_grad_norm():
    return sum([torch.norm(p.grad) for p in model.parameters() if p.grad is not None])

def calculate_param_norm():
    return sum([torch.norm(p) for p in model.parameters() if p.grad is not None])

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    decoding_table = wandb.Table(columns=["original text", "continuation"])

train_batch_iter = iter_batches("train")
X, Y = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0
# for _ in epochs:
# for X, Y in tqdm(next(train_batch_iter))
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            out = model(input_ids=X, labels=X)
            loss = out.loss
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(train_batch_iter)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    grad_norm = calculate_grad_norm()
    param_norm = calculate_param_norm()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    # Eval
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        if wandb_log:
            generate_sample()
            try:
                wandb.log({
                    "loss/train": losses["train"].item(),
                    "loss/test": losses["test"].item(),
                    "decoded_table": copy(decoding_table),  # workaround so wandb table updates
                })
            except Exception as e:
                print(f"logging to wandb failed: {e}")

        # Save model if good
        if losses["test"] < best_val_loss or always_save_checkpoint:
            if True or iter_num > 0:
                model.save_pretrained(out_dir)
                tokenizer.save_pretrained(out_dir)
                print(f"Saving checkpoint to {out_dir}")
    
    if iter_num % log_interval == 0:
        # wandb logging
        if wandb_log:
            try:
                wandb.log({
                        "iter": iter_num,
                        "tokens": iter_num * tokens_per_iter,
                        "loss/train": loss.item() * gradient_accumulation_steps,
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "param_norm": param_norm,
                        "forward time (ms)": dt*1000,
                })
            except Exception as e:
                    print(f"logging to wandb failed: {e}")
        
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | {round(torch.cuda.max_memory_allocated()/1e9, 1)}GB memory"
        )
    iter_num += 1
    local_iter_num += 1
