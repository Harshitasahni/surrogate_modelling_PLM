#!/usr/bin/env python3
import argparse, os, math, glob
import numpy as np
import torch

def read_fasta(path: str):
    #fasta is of form >ID seq 
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

def mean_pool(reps_btC, toks_bT, alphabet):
    pad = alphabet.padding_idx
    bos = alphabet.cls_idx
    eos = alphabet.eos_idx
    mask = (toks_bT != pad) & (toks_bT != bos) & (toks_bT != eos)    # [B,T]
    mask_f = mask.unsqueeze(-1).float()                              # [B,T,1]
    summed = (reps_btC * mask_f).sum(1)                              # [B,C]
    denom  = mask_f.sum(1).clamp_min(1.0)                            # [B,1]
    return summed / denom

def parse_layers(s: str):
    s = s.strip()
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]

def shard_list(items, shard_id: int, num_shards: int):
    # deterministic sharding
    return [x for i, x in enumerate(items) if (i % num_shards) == shard_id]

def embed_chunked(
    seqs_dict,
    protein_ids,
    out_dir,
    model_name,
    layers,
    batch_size,
    use_fp16,
    device,
    chunk_size,
    max_len=1024,
    stride=512,
):
    import esm

    os.makedirs(out_dir, exist_ok=True)

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    def forward_and_pool(batch):
        """batch: list[(pid, seq)] -> returns dict pid -> torch [L,C] float32 on GPU"""
        labels, strs, toks = batch_converter(batch)
        toks = toks.to(device, non_blocking=True)

        with torch.no_grad():
            if use_fp16 and device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(toks, repr_layers=layers, return_contacts=False)
            else:
                out = model(toks, repr_layers=layers, return_contacts=False)

        pooled_layers = []
        for layer in layers:
            reps = out["representations"][layer]          # [B,T,C]
            pooled = mean_pool(reps, toks, alphabet)      # [B,C]
            pooled_layers.append(pooled)

        pooled_stack = torch.stack(pooled_layers, dim=1)  # [B,L,C]
        pooled_stack = pooled_stack.float()               # [B,L,C] float32

        del out, pooled_layers, toks
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        # return as dict
        result = {}
        for bi, (pid, _) in enumerate(batch):
            result[pid] = pooled_stack[bi]  # [L,C] on GPU
        return result

    def embed_one_tiled(pid: str):
        """Returns torch [L,C] float32 on GPU using tiling if needed."""
        seq = seqs_dict[pid]
        Lseq = len(seq)

        # short: one pass
        if Lseq <= max_len:
            d = forward_and_pool([(pid, seq)])
            return d[pid]

        # long: windows + length-weighted average
        # ensure last window hits end
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

            d = forward_and_pool([(pid, subseq)])
            emb = d[pid]  # [L,C] float32 on GPU

            if sum_emb is None:
                sum_emb = emb * w
            else:
                sum_emb = sum_emb + emb * w
            sum_w += w

            # free window emb
            del emb
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        return sum_emb / max(sum_w, 1.0)

    protein_ids = list(protein_ids)
    total = len(protein_ids)
    n_chunks = math.ceil(total / chunk_size)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(
        f"[embed] device={device} proteins={total} chunks={n_chunks} "
        f"chunk_size={chunk_size} batch={batch_size} max_len={max_len} stride={stride}",
        flush=True,
    )

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

        # batch shorts
        for i in range(0, len(short_ids), batch_size):
            sub = short_ids[i : i + batch_size]
            batch = [(pid, seqs_dict[pid]) for pid in sub]
            try:
                pooled = forward_and_pool(batch)  # pid -> [L,C] GPU
                for pid, emb_LC in pooled.items():
                    results[pid] = emb_LC.detach().cpu().numpy().astype(
                        np.float16 if use_fp16 else np.float32
                    )
            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] short batch {sub}; retrying individually", flush=True)
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                for pid in sub:
                    try:
                        emb_LC = embed_one_tiled(pid)  # handles short too
                        results[pid] = emb_LC.detach().cpu().numpy().astype(
                            np.float16 if use_fp16 else np.float32
                        )
                    except torch.cuda.OutOfMemoryError:
                        print(f"[SKIP-OOM] {pid} len={len(seqs_dict[pid])}", flush=True)
                        if device.startswith("cuda"):
                            torch.cuda.empty_cache()

        # tile longs
        for pid in long_ids:
            try:
                emb_LC = embed_one_tiled(pid)
                results[pid] = emb_LC.detach().cpu().numpy().astype(
                    np.float16 if use_fp16 else np.float32
                )
            except torch.cuda.OutOfMemoryError:
                print(
                    f"[SKIP-OOM] {pid} len={len(seqs_dict[pid])} (even with tiling)",
                    flush=True,
                )
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

        np.savez(out_path, **results)
        print(f"[saved] {out_path} proteins={len(results)} ({end}/{total})", flush=True)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta")
    ap.add_argument("--train_pos")
    ap.add_argument("--train_neg")

    ap.add_argument("--model")
    ap.add_argument("--layers")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_shards", type=int, default=2)
    ap.add_argument("--shard_id", type=int, default=0)

    ap.add_argument("--chunk_size", type=int, default=200)
    ap.add_argument("--out_root")
   
    ap.add_argument("--max_len", type=int, default=1024, help="window length for long sequences")
    ap.add_argument("--stride", type=int, default=512, help="stride for tiling long sequences")

    args = ap.parse_args()

    layers = parse_layers(args.layers)
    
    # Check for missing sequences 
    missing = [name for name in ("fasta", "train_pos", "train_neg") if getattr(args, name) is None]
    if missing:
        print(f"[warning] {missing} not provided, will skip", flush=True)
    shard_dir = os.path.join(args.out_root, f"shard_{args.shard_id}")
    os.makedirs(shard_dir, exist_ok=True)
    print("[load] fasta...", flush=True)
    seqs = read_fasta(args.fasta)
    print(f"[load] sequences={len(seqs)}", flush=True)

    print("[load] train pairs...", flush=True)
    pos = read_pairs(args.train_pos)
    neg = read_pairs(args.train_neg)

    # Keep only proteins present in fasta 
    pairs = [(a, b) for (a, b) in (pos + neg) if a in seqs and b in seqs]
    train_proteins = sorted({a for a, b in pairs} | {b for a, b in pairs})

    print(f"[data] pairs_after_filter={len(pairs)} train_proteins={len(train_proteins)}", flush=True)

    my_proteins = shard_list(train_proteins, args.shard_id, args.num_shards)
    print(f"[shard] shard_id={args.shard_id}/{args.num_shards} proteins={len(my_proteins)} out_dir={shard_dir}", flush=True)

    
    embed_chunked(
        seqs_dict=seqs,
        protein_ids=my_proteins,
        out_dir=shard_dir,
        model_name=args.model,
        layers=layers,
        batch_size=args.batch_size,
        use_fp16=args.fp16,
        device=args.device,
        chunk_size=args.chunk_size,
        max_len=args.max_len,
        stride=args.stride,
    )



if __name__ == "__main__":
    main()
