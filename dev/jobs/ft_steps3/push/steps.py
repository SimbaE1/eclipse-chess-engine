# ── Eclipse fine-tune STEP-COUNT sweep (Kaggle GPU) ──────────────────────────
# Model/QAT code below is pasted VERBATIM from eclipse_wdl_train.ipynb cells 3+5.
# The training step mirrors cell 6 exactly: AMP fp16 + GradScaler, DataParallel
# over 2x T4, Adam(wd=1e-4), clip 1.0, padding-row re-zero, 500-step warmup.
# Each config restarts from ep2c44 (init_weights.pt) on the SAME batches.
import sys
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(line_buffering=True)
    except Exception: pass
import glob, json, math, os, struct, time
from pathlib import Path
import numpy as np, torch, torch.nn as _nn, torch.nn.functional as F
# HalfKAv2 feature indexing -- exact mirror of src/nnue.cpp::feature_index
# and scripts/train_halfkav2.py. Drift here = silently wrong inference.
import gzip, math, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

FT_IN_FEATURES = 45056          # HalfKAv2: 64 (king sq) * 64 (piece sq) * 11 (piece slots)
# ── Architecture ──────────────────────────────────────────────────────────
# Stockfish-modern shape, trained from random init (see the header cell for why
# the 2048x2->1024->256 net is being retired rather than fine-tuned).
#
# DEPLOY NOTE: these four numbers are the contract with the C++ engine. Before a
# net trained here will load:
#     src/accumulator.hpp : kFtOutSize  = FT_OUT
#     src/nnue.hpp        : kL1OutSize  = L1_OUT
#                           kL2OutSize  = L2_OUT
# (kL1InSize follows kFtOutSize automatically; the SIMD kernels in src/nnue.cpp
# already handle these widths.) convert_halfkav2_nnue.py reads the widths out of
# the checkpoint, so it needs no edit. The loader cross-checks the header and
# refuses a mismatched file rather than evaluating garbage.
#
# ARCH picks a preset so switching lineages is one edit, not three:
#   'wide' -- 2048x2 -> 1024 -> 256: the DEPLOYED net (eclipse_v9_ep2c44) and the
#             shape the engine's headers are built for today. Used for the
#             2026-11 warm restart on the quiet-filtered data.
#   'sf16' -- 1024x2 -> 16 -> 32: the from-scratch experiment of Oct 2026
#             (checkpoint lineage simbae11/eclipse-checkpoint-sf16). Needs the
#             C++ header edits above before it can be deployed.
ARCH = 'wide'
FT_OUT, L1_OUT, L2_OUT = {'wide': (2048, 1024, 256),
                          'sf16': (1024, 16, 32)}[ARCH]
L1_IN          = 2 * FT_OUT     # concat of both perspectives
L3_OUT         = 1              # Value head: single logit (sigmoid = win prob)
N_PIECE_SLOTS  = 11

_PIECE_LETTER_TO_TYPE = {'p':1, 'n':2, 'b':3, 'r':4, 'q':5, 'k':6}
_FLIP_RANK = lambda s: s ^ 56

def parse_fen_board(fen_board):
    pieces = []
    sq = 56
    for ch in fen_board:
        if ch == '/':
            sq -= 16
            continue
        if ch.isdigit():
            sq += int(ch)
            continue
        is_white = ch.isupper()
        pt = _PIECE_LETTER_TO_TYPE[ch.lower()]
        pieces.append((sq, 0 if is_white else 1, pt))
        sq += 1
    return pieces

def feature_index(king_sq, piece_sq, pt, piece_is_ours):
    # ours: P=0,N=1,B=2,R=3,Q=4 (own king skipped). theirs: P=5..K=10.
    if piece_is_ours and pt == 6:
        raise ValueError('own king is not a HalfKAv2 feature')
    slot = (pt - 1) if piece_is_ours else (5 + (pt - 1))
    return king_sq * (64 * N_PIECE_SLOTS) + piece_sq * N_PIECE_SLOTS + slot

def fen_to_features(fen):
    parts = fen.split()
    board, stm = parts[0], parts[1]
    pieces = parse_fen_board(board)
    stm_color = 0 if stm == 'w' else 1
    king_sq = [None, None]
    for sq, color, pt in pieces:
        if pt == 6:
            king_sq[color] = sq
    if king_sq[0] is None or king_sq[1] is None:
        raise ValueError(f'FEN missing a king: {fen}')
    indices = [[], []]
    for persp in (0, 1):
        ksq = king_sq[persp] if persp == 0 else _FLIP_RANK(king_sq[persp])
        for sq, color, pt in pieces:
            is_ours = (color == persp)
            if is_ours and pt == 6:
                continue
            psq = sq if persp == 0 else _FLIP_RANK(sq)
            indices[persp].append(feature_index(ksq, psq, pt, is_ours))
    return (indices[0], indices[1]) if stm_color == 0 else (indices[1], indices[0])

# Model -- float mirror of the C++ inference path in src/nnue.cpp.
# Improvement 1: Quantization-Aware Training (QAT).
#
# Eclipse deploys int16 FT weights and int8 hidden weights. Training in float32
# then converting creates a systematic error: every weight is rounded to the
# nearest quantization step, and the model never saw that rounding during
# training so it can't compensate for it.
#
# QAT inserts fake-quantization nodes (round → dequantize) in the forward pass
# using the Straight-Through Estimator (STE) for gradients. The model sees
# the same discretization error at training time as at inference time, so it
# learns weights that are robust to rounding.
#
# Quantization constants (must match nnue.hpp / convert_halfkav2_nnue.py):
#   FT weights/biases  → int16: stored = round(real * 127),  scale = 1/127
#   Hidden weights     → int8:  stored = round(real * 64),   scale = 1/64
#   Activations (CReLU output) are in [0, 1] = [0, 127/127], discretised to
#   128 levels: scale = 1/127, same as FT.

FT_QUANT     = 127       # kFtQuant in nnue.hpp
WEIGHT_SCALE = 64        # kWeightScale in nnue.hpp

FT_SCALE  = 1.0 / FT_QUANT
FT_LO     = -32768.0 / FT_QUANT
FT_HI     =  32767.0 / FT_QUANT
W_SCALE   = 1.0 / WEIGHT_SCALE
W_LO      = -128.0 / WEIGHT_SCALE   # = -2.0
W_HI      =  127.0 / WEIGHT_SCALE   # ≈ +1.984
ACT_SCALE = 1.0 / FT_QUANT
ACT_LO    = 0.0
ACT_HI    = 1.0


class FakeQuantSTE(torch.autograd.Function):
    """Fake quantization: round to grid in forward, identity (STE) in backward."""
    @staticmethod
    def forward(ctx, x, scale, lo, hi):
        return torch.clamp(torch.round(x / scale) * scale, lo, hi)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None  # STE: pass gradient through unchanged


def fake_quant(x, scale, lo, hi):
    return FakeQuantSTE.apply(x, scale, lo, hi)


class ClippedReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class HalfKAv2Net(nn.Module):
    def __init__(self, qat: bool = False):
        super().__init__()
        self.qat = qat

        # Last embedding row is the padding sentinel: zeroed at init,
        # pinned to zero after every step so padded slots contribute 0.
        self.ft = nn.Embedding(FT_IN_FEATURES + 1, FT_OUT,
                               padding_idx=FT_IN_FEATURES)
        nn.init.uniform_(self.ft.weight, -0.01, 0.01)
        with torch.no_grad():
            self.ft.weight[FT_IN_FEATURES].zero_()
        self.ft_bias = nn.Parameter(torch.zeros(FT_OUT))
        self.l1  = nn.Linear(L1_IN, L1_OUT)
        self.l2  = nn.Linear(L1_OUT, L2_OUT)
        self.l3  = nn.Linear(L2_OUT, L3_OUT)
        self.act = ClippedReLU()

    def forward(self, us_idx, them_idx):
        if self.qat:
            # Fake-quantize FT weights to int16 grid before lookup.
            ft_w_q = fake_quant(self.ft.weight, FT_SCALE, FT_LO, FT_HI)
            ft_b_q = fake_quant(self.ft_bias,   FT_SCALE, FT_LO, FT_HI)
            us   = F.embedding(us_idx,   ft_w_q).sum(dim=1) + ft_b_q
            them = F.embedding(them_idx, ft_w_q).sum(dim=1) + ft_b_q
            # Fake-quantize activations to 128 discrete levels ([0, 127/127]).
            us   = fake_quant(torch.clamp(us,   0.0, 1.0), ACT_SCALE, ACT_LO, ACT_HI)
            them = fake_quant(torch.clamp(them, 0.0, 1.0), ACT_SCALE, ACT_LO, ACT_HI)
        else:
            us   = self.ft(us_idx).sum(dim=1)   + self.ft_bias
            them = self.ft(them_idx).sum(dim=1) + self.ft_bias
            us   = self.act(us)
            them = self.act(them)

        x = torch.cat([us, them], dim=1)

        if self.qat:
            l1_w = fake_quant(self.l1.weight, W_SCALE, W_LO, W_HI)
            x = fake_quant(torch.clamp(F.linear(x, l1_w, self.l1.bias), 0.0, 1.0),
                           ACT_SCALE, ACT_LO, ACT_HI)
            l2_w = fake_quant(self.l2.weight, W_SCALE, W_LO, W_HI)
            x = fake_quant(torch.clamp(F.linear(x, l2_w, self.l2.bias), 0.0, 1.0),
                           ACT_SCALE, ACT_LO, ACT_HI)
            l3_w = fake_quant(self.l3.weight, W_SCALE, W_LO, W_HI)
            return F.linear(x, l3_w, self.l3.bias)
        else:
            x = self.act(self.l1(x))
            x = self.act(self.l2(x))
            return self.l3(x)

    def enable_qat(self):
        self.qat = True

    def disable_qat(self):
        self.qat = False

    def export_state_dict_for_quantization(self):
        sd = {}
        ft_w = self.ft.weight.detach()
        sd['ft.weight'] = ft_w[:FT_IN_FEATURES].T.contiguous()
        sd['ft.bias']   = self.ft_bias.detach()
        sd['l1.weight'] = self.l1.weight.detach()
        sd['l1.bias']   = self.l1.bias.detach()
        sd['l2.weight'] = self.l2.weight.detach()
        sd['l2.bias']   = self.l2.bias.detach()
        sd['l3.weight'] = self.l3.weight.detach()
        sd['l3.bias']   = self.l3.bias.detach()
        return sd


# The net2widernet helper that used to live here is gone along with the widen
# itself: it referenced FT_OUT_SRC / L1_OUT_SRC / L2_OUT_SRC, which no longer
# exist, and this run trains from random init. Recover it from git history if a
# function-preserving widen is ever wanted again.


# ───────────────────────────── step-count sweep driver ─────────────────────────
# Why: smoke (1000 steps @5e-5 from ep2c44, pure quiet data) beat the full 2-epoch
# quiet-v2 run by ~+41 Elo and its ep0c47 checkpoint by ~+49. Local check showed the
# quiet-position gain is complete by ~1000 steps while error on TACTICAL positions
# (the ones the quiet filter drops) keeps growing (MAE 183 -> 208 @1k -> 226 @ep0c23).
# SWEEP 2 (2026-11-10). Sweep 1 result: mix50_s2000 lost -37 Elo and quiet_s500 -45 to smoke
# (quiet, 1000 steps, constant 5e-5); mix50 curves flat from ~1.5k steps. Two untested levers:
#   (a) LOWER mix ratio -- keep most of the quiet gain, forget fewer tactics. Constant LR, so one run
#       snapshotted at several steps == separate runs.
#   (b) LR DECAY -- cosine from lr to lr*0.1 after the 500-step warmup. The schedule depends on the
#       total length, so each decay length is its own run (snapshot only at the end).
# SWEEP 3 (2026-11-12): does mix20 keep improving / start degrading past 2k steps?
# mix20_cos2000 (= v1.2.0 release net) is the reference.
CONFIGS = [
    dict(name='mix20',         lr=5e-5, mix=0.2, steps=8000, snaps=[2000, 4000, 6000, 8000]),
    dict(name='mix20_cos4000', lr=5e-5, mix=0.2, steps=4000, snaps=[4000], decay=True),
    dict(name='mix20_cos8000', lr=5e-5, mix=0.2, steps=8000, snaps=[8000], decay=True),
]
LR_FLOOR = 0.1                                  # decay ends at lr * LR_FLOOR

def lr_at(cfg, step):
    if step <= WARMUP: return cfg['lr'] * step / WARMUP
    if not cfg.get('decay'): return cfg['lr']
    f = (step - WARMUP) / max(1, cfg['steps'] - WARMUP)
    return cfg['lr'] * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * f)))

EVAL_EVERY, WARMUP, BS, WD = 500, 500, 16384, 1e-4
N_Q, N_A = 28, 8                               # 28 x 3.9M = 109M >= 8000 x 13107 quiet; 8 x 3.9M >= 8000 x 3277 unfiltered
WORK = Path('/kaggle/working')
LOCAL = os.environ.get('SMOKE_LOCAL') == '1'   # CPU dry-run of the driver logic only
if LOCAL:
    CONFIGS = [dict(c, steps=4, snaps=[2, 4]) for c in CONFIGS]
    EVAL_EVERY, WARMUP, BS, N_Q, N_A = 2, 2, 512, 1, 1
    WORK = Path(os.environ['SMOKE_WORK'])
    _orig_glob = glob.glob
    glob.glob = lambda pat, recursive=False: _orig_glob(pat.replace('/kaggle/input', os.environ['SMOKE_INPUT']), recursive=recursive)
    torch.cuda.synchronize = lambda *a, **k: None
dev = torch.device('cpu' if LOCAL else 'cuda')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
DT = np.dtype([('us', np.uint16, (32,)), ('them', np.uint16, (32,)), ('tgt', np.float16)])

def find(name, slug):
    """Both datasets use eval_chunk_NNN.bin names, so match on the dataset slug in the path."""
    c = sorted(p for p in glob.glob(f'/kaggle/input/**/{name}', recursive=True) if f'/{slug}/' in p)
    assert c, f'{name} from {slug} not mounted'
    return c[0]

def bin_len(p):
    with open(p, 'rb') as f:
        assert f.read(8) == b'HKAV2BIN'; return struct.unpack('<q', f.read(8))[0]

def load_many(paths):
    """Read chunks straight into one preallocated array (no concat -> no 2x RAM peak)."""
    ns = [bin_len(p) for p in paths]
    out = np.empty(sum(ns), dtype=DT); off = 0; t = time.time()
    for p, n in zip(paths, ns):
        with open(p, 'rb') as f:
            f.seek(16); f.readinto(memoryview(out[off:off + n]).cast('B'))
        off += n
    print(f'  loaded {len(paths)} chunks, {off:,} recs in {time.time()-t:.0f}s', flush=True)
    return out

def read_bin(p):
    return load_many([p])

def to_dev(r):
    return (torch.from_numpy(r['us'].astype(np.int64)).to(dev, non_blocking=True),
            torch.from_numpy(r['them'].astype(np.int64)).to(dev, non_blocking=True),
            torch.from_numpy(r['tgt'].astype(np.float32)).to(dev, non_blocking=True))

print('loading data...')
Q = load_many([find(f'eval_chunk_{i:03d}.bin', 'eclipse-chunks-quiet') for i in range(N_Q)])
A = load_many([find(f'eval_chunk_{i:03d}.bin', 'eclipse-chunks-a') for i in range(N_A)])
VAL = read_bin(find('val_holdout.bin', 'eclipse-chunks-quiet'))
LEG = read_bin(find('val_legacy.bin', 'eclipse-chunks-quiet'))
if LOCAL: VAL, LEG = VAL[:3000], LEG[:3000]
INIT = find('init_weights.pt', 'eclipse-chunks-quiet')
men = (VAL['us'] != FT_IN_FEATURES).sum(1) + 1          # features exclude own king
BUCKETS = {'<=12': men <= 12, '13-18': (men > 12) & (men <= 18),
           '19-25': (men > 18) & (men <= 25), '26+': men > 25}
print('val bucket sizes', {k: int(v.sum()) for k, v in BUCKETS.items()})

def fresh_net():
    net = HalfKAv2Net(qat=True).to(dev)
    sd = torch.load(INIT, map_location=dev)
    ft_w = sd['ft.weight'].T
    net.load_state_dict({'ft.weight': torch.cat([ft_w, torch.zeros(1, ft_w.shape[1], device=dev)], 0),
                         'ft_bias': sd['ft.bias'],
                         **{k: sd[k] for k in ('l1.weight','l1.bias','l2.weight','l2.bias','l3.weight','l3.bias')}})
    return net

@torch.no_grad()
def per_pos_loss(net, arr):
    net.eval(); out = []
    for i in range(0, len(arr), BS):
        us, them, t = to_dev(arr[i:i+BS])
        with torch.autocast('cuda', dtype=torch.float16, enabled=not LOCAL):
            lg = net(us, them)
        out.append(F.binary_cross_entropy_with_logits(lg.squeeze(1).float(), t, reduction='none').cpu())
    net.train()
    return torch.cat(out).numpy()

def evaluate(net):
    lv = per_pos_loss(net, VAL); ll = per_pos_loss(net, LEG)
    r = {'val': float(lv.mean()), 'leg': float(ll.mean())}
    for k, m in BUCKETS.items(): r['v' + k] = float(lv[m].mean())
    return r

def fmt(step, lr, tr, r, dt_step):
    return (f'step {step:5d} lr {lr:.2e} train {tr:.5f} | val {r["val"]:.5f} leg {r["leg"]:.5f} | '
            + ' '.join(f'{k} {r["v"+k]:.5f}' for k in BUCKETS) + f' | {dt_step*1e3:.0f}ms/step')

def run(cfg, record):
    base = fresh_net()
    net = _nn.DataParallel(base) if torch.cuda.device_count() > 1 else base
    opt = torch.optim.Adam(base.parameters(), lr=cfg['lr'], weight_decay=WD)
    scaler = torch.cuda.amp.GradScaler(enabled=not LOCAL)
    rng = np.random.default_rng(1234)              # same quiet stream for every config
    pq = rng.permutation(len(Q)); pa = rng.permutation(len(A))
    nb_a = int(round(BS * cfg['mix'])); nb_q = BS - nb_a
    iq = ia = 0
    r0 = evaluate(base); record.append(dict(step=0, lr=0.0, train=None, **r0))
    print(cfg['name'], fmt(0, 0.0, float('nan'), r0, 0), flush=True)
    tr_acc = []; t_acc = 0.0
    for step in range(1, cfg['steps'] + 1):
        t0 = time.time()
        idx_q = pq[np.arange(iq, iq + nb_q) % len(pq)]; iq += nb_q
        parts = [Q[np.sort(idx_q)]]
        if nb_a:
            idx_a = pa[np.arange(ia, ia + nb_a) % len(pa)]; ia += nb_a
            parts.append(A[np.sort(idx_a)])
        us, them, t = to_dev(np.concatenate(parts) if len(parts) > 1 else parts[0])
        for g in opt.param_groups: g['lr'] = lr_at(cfg, step)
        with torch.autocast('cuda', dtype=torch.float16, enabled=not LOCAL):
            lg = net(us, them)
            loss = F.binary_cross_entropy_with_logits(lg.squeeze(1), t)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        with torch.no_grad(): base.ft.weight[FT_IN_FEATURES].zero_()
        tr_acc.append(loss.item()); torch.cuda.synchronize(); t_acc += time.time() - t0
        if step % EVAL_EVERY == 0 or step in cfg['snaps']:
            r = evaluate(base); r.update(step=step, lr=opt.param_groups[0]['lr'], train=float(np.mean(tr_acc)))
            record.append(r)
            print(cfg['name'], fmt(step, r['lr'], r['train'], r, t_acc / max(1, len(tr_acc))), flush=True)
            tr_acc = []; t_acc = 0.0
        if step in cfg['snaps']:
            torch.save(base.export_state_dict_for_quantization(), WORK / f'ft_{cfg["name"]}_s{step}.pt')
            print(f'  saved ft_{cfg["name"]}_s{step}.pt', flush=True)
    return base

results = {}
for cfg in CONFIGS:
    print(f'\n== config {cfg["name"]} (mix {cfg["mix"]}, {cfg["steps"]} steps) ==', flush=True)
    rec = []; t = time.time()
    base = run(cfg, rec)
    print(f'  {cfg["name"]} done in {(time.time()-t)/60:.1f} min', flush=True)
    results[cfg['name']] = rec
    json.dump(results, open(WORK / 'steps_results.json', 'w'), indent=1)
    del base; torch.cuda.empty_cache()
print('STEP SWEEP 3 COMPLETE')
