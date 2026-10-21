"""Generate -> Verify -> Reward -> Update  (GRPO tối giản, G = 4).

Cài:   pip install torch transformers peft
Bước 0 (đo baseline, chưa train):  python train_grpo.py --steps 0
Bước 1 (train thử):                python train_grpo.py --steps 30
Bước 2 (train thật):               python train_grpo.py --steps 300

Đơn giản hóa có chủ ý: mỗi batch chỉ update 1 lần (on-policy) nên tỉ số
pi_new/pi_old = 1 -> không cần clipping; bỏ KL penalty. Đây là bản "trực giác"
của slide, chưa phải GRPO đầy đủ.
"""
import argparse
import csv
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from countdown import build_messages, group_advantages, make_dataset, verify

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--steps", type=int, default=30)
p.add_argument("--G", type=int, default=4)
p.add_argument("--prompts_per_step", type=int, default=4)
p.add_argument("--n_numbers", type=int, default=3)
p.add_argument("--max_new_tokens", type=int, default=64)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--eval_n", type=int, default=50)
p.add_argument("--eval_every", type=int, default=25)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
    torch.float16 if device == "cuda" else torch.float32)

# ------------------------------------------------------------ model + LoRA
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
for prm in model.parameters():            # tham số LoRA giữ fp32 cho ổn định
    if prm.requires_grad:
        prm.data = prm.data.float()
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr)

# ------------------------------------------------------------ data
train = make_dataset(5000, seed=args.seed, n_numbers=args.n_numbers)
test = make_dataset(args.eval_n, seed=999, n_numbers=args.n_numbers)   # seed khác -> không lẫn train


def encode_prompt(numbers, target):
    text = tok.apply_chat_template(build_messages(numbers, target),
                                   tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").to(device)


# ------------------------------------------------------------ GENERATE
@torch.no_grad()
def generate(numbers, target, n, greedy=False):
    enc = encode_prompt(numbers, target)
    out = model.generate(
        **enc, max_new_tokens=args.max_new_tokens, num_return_sequences=n,
        do_sample=not greedy, temperature=None if greedy else args.temperature,
        top_p=None if greedy else 1.0, pad_token_id=tok.pad_token_id)
    plen = enc["input_ids"].shape[1]
    texts = tok.batch_decode(out[:, plen:], skip_special_tokens=True)
    return out, plen, texts


def completion_logprobs(seqs, plen):
    """log pi(token | context) cho từng token của completion. Shape [G, T]."""
    mask = (seqs != tok.pad_token_id).long()
    logits = model(input_ids=seqs, attention_mask=mask).logits[:, :-1].float()
    logp = torch.log_softmax(logits, dim=-1).gather(-1, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
    return logp[:, plen - 1:], mask[:, plen:].float()   # chỉ lấy phần completion


# ------------------------------------------------------------ EVAL (greedy, pass@1)
def evaluate():
    model.eval()
    ok, reasons = 0, {}
    for nums, tgt in test:
        _, _, txt = generate(nums, tgt, 1, greedy=True)
        r, why = verify(txt[0], nums, tgt)
        ok += r
        reasons[why] = reasons.get(why, 0) + 1
    model.train()
    return ok / len(test), reasons


acc, why = evaluate()
print(f"[eval step 0] pass@1 = {acc:.2%}  {why}")   # = baseline Prompting của đề tài

# ------------------------------------------------------------ TRAIN LOOP
log = open("train_log.csv", "w", newline="")
w = csv.writer(log)
w.writerow(["step", "mean_reward", "groups_used", "loss"])
model.train()
for step in range(1, args.steps + 1):
    opt.zero_grad()
    rewards_all, used, total_loss = [], 0, 0.0
    for nums, tgt in random.sample(train, args.prompts_per_step):
        seqs, plen, texts = generate(nums, tgt, args.G)                        # 1. GENERATE
        rewards = [verify(t, nums, tgt)[0] for t in texts]                      # 2-3. VERIFY -> REWARD
        rewards_all += rewards
        adv = group_advantages(rewards)                                         # 4. ADVANTAGE
        if all(a == 0 for a in adv):
            continue                                                            # cả group như nhau -> bỏ
        used += 1
        logp, cmask = completion_logprobs(seqs, plen)                           # 5. UPDATE
        a = torch.tensor(adv, device=device).unsqueeze(1)
        loss = -(a * logp * cmask).sum() / cmask.sum()
        (loss / args.prompts_per_step).backward()
        total_loss += loss.item()
    if used:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    mr = sum(rewards_all) / len(rewards_all)
    w.writerow([step, mr, used, total_loss]); log.flush()
    print(f"step {step:4d} | mean_reward {mr:.2f} | groups_used {used}/{args.prompts_per_step}")
    if step % args.eval_every == 0 or step == args.steps:
        acc, why = evaluate()
        print(f"[eval step {step}] pass@1 = {acc:.2%}  {why}")

model.save_pretrained("lora_countdown")
