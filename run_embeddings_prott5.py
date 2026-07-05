#!/usr/bin/env python3
"""
ProtT5 layer-wise embedding extraction.
"""
import argparse, os, math, re
import numpy as np
import torch


def read_fasta(path: str):
    # fasta of form >ID seq
    seqs = {}
    cur_id, cur = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur)
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line)
        if cur_id is not None:
            seqs[cur_id] = "".join(cur)
    return seqs


def read_pairs(path: str):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                a, b = line.split(",")[:2]
            else:
                a, b = line.split()[:2]
            pairs.append((a.strip(), b.strip()))
    return pairs


def parse_layers(s: str):
    s = s.strip()
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def shard_list(items, shard_id: int, num_shards: int):
    return [x for i, x in enumerate(items) if (i % num_shards) == shard_id]


def prott5_prepare(seq: str) -> str:
    """Uppercase, map rare residues U/Z/O/B -> X, insert spaces between residues."""
    seq = seq.upper()
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))


def embed_chunked(
    seqs_dict,
    protein_ids,
    out_dir,
    model_name,
    layers,
    batch_size,
    device,
    chunk_size,
    max_len=1024,
    stride=512,
):
    from transformers import T5EncoderModel, T5Tokenizer

    os.makedirs(out_dir, exist_ok=True)

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    if device.startswith("cuda"):
        model = model.half()

    n_hidden = model.config.num_layers  # encoder layers (24 for ProtT5-XL)
    print(f"[model] {model_name}  encoder_layers={n_hidden}  "
          f"hidden_states_available=0..{n_hidden}", flush=True)
    for L in layers:
        if L < 0 or L > n_hidden:
            raise ValueError(f"--layers index {L} out of range 0..{n_hidden}")

    def forward_and_pool(batch):
        """batch: list[(pid, raw_seq)] -> dict pid -> torch [len(layers), C] float32 (GPU)."""
        pids = [pid for pid, _ in batch]
        prepared = [prott5_prepare(seq) for _, seq in batch]
        enc = tokenizer.batch_encode_plus(
            prepared, add_special_tokens=True, padding="longest", return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)  # [B,T]; 1 for real tokens incl </s>

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn,
                        output_hidden_states=True)
        hs = out.hidden_states  # tuple len (n_hidden+1), each [B,T,C]

        # Build a pooling mask that excludes padding AND the trailing </s>.
        # attention_mask includes </s>=1; the </s> is the last real token of each row.
        mask = attn.clone()                       # [B,T]
        lengths = attn.sum(dim=1)                 # includes </s>
        for bi in range(mask.shape[0]):
            eos_pos = int(lengths[bi].item()) - 1  # index of </s>
            if eos_pos >= 0:
                mask[bi, eos_pos] = 0             # drop </s> from pooling
        mask_f = mask.unsqueeze(-1).float()       # [B,T,1]
        denom = mask_f.sum(1).clamp_min(1.0)      # [B,1]

        pooled_layers = []
        for L in layers:
            reps = hs[L].float()                  # [B,T,C]
            pooled = (reps * mask_f).sum(1) / denom  # [B,C]
            pooled_layers.append(pooled)
        pooled_stack = torch.stack(pooled_layers, dim=1)  # [B, len(layers), C]

        del out, hs, input_ids, attn
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        return {pid: pooled_stack[bi] for bi, pid in enumerate(pids)}

    def embed_one_tiled(pid: str):
        seq = seqs_dict[pid]
        Lseq = len(seq)
        if Lseq <= max_len:
            return forward_and_pool([(pid, seq)])[pid]

        starts = list(range(0, max(1, Lseq - max_len + 1), stride))
        last = max(0, Lseq - max_len)
        if not starts or starts[-1] != last:
            starts.append(last)

        sum_emb = None
        sum_w = 0.0
        for s in starts:
            e = min(s + max_len, Lseq)
            subseq = seq[s:e]
            w = float(e - s)
            emb = forward_and_pool([(pid, subseq)])[pid]
            sum_emb = emb * w if sum_emb is None else sum_emb + emb * w
            sum_w += w
            del emb
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        return sum_emb / max(sum_w, 1.0)

    protein_ids = list(protein_ids)
    total = len(protein_ids)
    n_chunks = math.ceil(total / chunk_size)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(f"[embed] device={device} proteins={total} chunks={n_chunks} "
          f"chunk_size={chunk_size} batch={batch_size} max_len={max_len} stride={stride}",
          flush=True)

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min((ci + 1) * chunk_size, total)
        chunk_ids = protein_ids[start:end]

        out_path = os.path.join(out_dir, f"chunk_{ci:05d}.npz")
        if os.path.exists(out_path):
            print(f"[skip] {out_path}", flush=True)
            continue

        results = {}
        short_ids = [pid for pid in chunk_ids if len(seqs_dict[pid]) <= max_len]
        long_ids  = [pid for pid in chunk_ids if len(seqs_dict[pid]) >  max_len]

        for i in range(0, len(short_ids), batch_size):
            sub = short_ids[i:i + batch_size]
            batch = [(pid, seqs_dict[pid]) for pid in sub]
            try:
                pooled = forward_and_pool(batch)
                for pid, emb_LC in pooled.items():
                    results[pid] = emb_LC.detach().cpu().numpy().astype(np.float32)
            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] short batch {sub}; retrying individually", flush=True)
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                for pid in sub:
                    try:
                        emb_LC = embed_one_tiled(pid)
                        results[pid] = emb_LC.detach().cpu().numpy().astype(np.float32)
                    except torch.cuda.OutOfMemoryError:
                        print(f"[SKIP-OOM] {pid} len={len(seqs_dict[pid])}", flush=True)
                        if device.startswith("cuda"):
                            torch.cuda.empty_cache()

        for pid in long_ids:
            try:
                emb_LC = embed_one_tiled(pid)
                results[pid] = emb_LC.detach().cpu().numpy().astype(np.float32)
            except torch.cuda.OutOfMemoryError:
                print(f"[SKIP-OOM] {pid} len={len(seqs_dict[pid])} (even with tiling)", flush=True)
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

        np.savez(out_path, **results)
        print(f"[saved] {out_path} proteins={len(results)} ({end}/{total})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta")
    ap.add_argument("--train_pos")
    ap.add_argument("--train_neg")
    ap.add_argument("--all_proteins", action="store_true",
                    help="Embed all proteins in fasta (skip pair filtering)")
    ap.add_argument("--model", default="Rostlab/prot_t5_xl_half_uniref50-enc")
    ap.add_argument("--layers", required=True,
                    help="e.g. 0-24 for all ProtT5-XL encoder hidden states")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--chunk_size", type=int, default=200)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    shard_dir = os.path.join(args.out_root, f"shard_{args.shard_id}")
    os.makedirs(shard_dir, exist_ok=True)

    print("[load] fasta...", flush=True)
    seqs = read_fasta(args.fasta)
    print(f"[load] sequences={len(seqs)}", flush=True)

    if args.all_proteins:
        train_proteins = sorted(seqs.keys())
        print(f"[data] all_proteins mode: proteins={len(train_proteins)}", flush=True)
    else:
        print("[load] train pairs...", flush=True)
        pos = read_pairs(args.train_pos)
        neg = read_pairs(args.train_neg)
        pairs = [(a, b) for (a, b) in (pos + neg) if a in seqs and b in seqs]
        train_proteins = sorted({a for a, b in pairs} | {b for a, b in pairs})
        print(f"[data] pairs_after_filter={len(pairs)} train_proteins={len(train_proteins)}", flush=True)

    my_proteins = shard_list(train_proteins, args.shard_id, args.num_shards)
    print(f"[shard] shard_id={args.shard_id}/{args.num_shards} "
          f"proteins={len(my_proteins)} out_dir={shard_dir}", flush=True)

    embed_chunked(
        seqs_dict=seqs,
        protein_ids=my_proteins,
        out_dir=shard_dir,
        model_name=args.model,
        layers=layers,
        batch_size=args.batch_size,
        device=args.device,
        chunk_size=args.chunk_size,
        max_len=args.max_len,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()