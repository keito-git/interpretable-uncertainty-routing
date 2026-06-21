"""routing11.py - per-corruption routing AUC across ALL 11 corruptions @ s4,
for ens_epistemic vs single max-prob (1D baseline). Random tie-breaking, 20 seeds.
Also the aggregate routing AUC over the 11-corruption pool. Honest, from saved preds."""
import json
from pathlib import Path
import numpy as np

EPS = 1e-12
OLD = ["gnoise", "blur", "occlude", "bright", "pixelate"]
NEW = ["shot_noise", "impulse_noise", "motion_blur", "contrast", "jpeg", "elastic"]
ALL = OLD + NEW


def ent(p):
    pc = np.clip(p, EPS, 1.0)
    return -np.sum(pc * np.log(pc), axis=-1)


def epi_decomp(member):
    mp = member.mean(0)
    return np.maximum(ent(mp) - np.stack([ent(member[k]) for k in range(len(member))]).mean(0), 0.0)


def load_ens(tag):
    # old corruptions in shift_ens, new in shift_ens_ext
    base = "data/shift_ens" if (tag.split("_s")[0] in OLD or tag == "none_s0") else "data/shift_ens_ext"
    return np.load(f"{base}/{tag}/predictions.npz")


def routing_auc(si, so, amb, rg):
    n = len(si)
    jit = max(np.std(np.concatenate([si, so])), 1e-9) * 1e-6
    si = si + rg.normal(0, jit, n); so = so + rg.normal(0, jit, len(so))
    s = np.concatenate([si, so]); is_ood = np.concatenate([np.zeros(n), np.ones(len(so))])
    ambm = np.concatenate([amb, np.zeros(len(so), bool)])
    order = np.argsort(-s); rej = np.zeros(len(s), bool); rows = []
    na = max(int(ambm.sum()), 1); no = len(so); step = max(len(order)//300, 1)
    for k in range(0, len(order)+1, step):
        rej[:] = False; rej[order[:k]] = True
        rows.append(((rej & (is_ood == 1)).sum()/no, (~rej & ambm).sum()/na))
    rows = np.array(rows); idx = np.argsort(rows[:, 0])
    return float(np.trapz(rows[idx, 1], rows[idx, 0]))


meta = np.load("data/processed/ferplus_test.npz", allow_pickle=True)
ve = ent(meta["soft"].astype(np.float64))
clean = load_ens("none_s0")
n = len(clean["epistemic"]); amb = ve[:n] >= np.quantile(ve[:n], 2/3)
epi_c = clean["epistemic"]
smx_c = 1 - clean["member"][0].max(1)   # single = member[0] (s42), consistent across all

per = {"ens_epistemic": {}, "single_maxprob": {}}
for c in ALL:
    e = load_ens(f"{c}_s4")
    epi_o = epi_decomp(e["member"]); smx_o = 1 - e["member"][0].max(1)
    per["ens_epistemic"][c] = round(float(np.mean(
        [routing_auc(epi_c, epi_o, amb, np.random.default_rng(s)) for s in range(20)])), 3)
    per["single_maxprob"][c] = round(float(np.mean(
        [routing_auc(smx_c, smx_o, amb, np.random.default_rng(s)) for s in range(20)])), 3)

# aggregate over 11-corruption pool
epi_pool = np.concatenate([epi_decomp(load_ens(f"{c}_s4")["member"]) for c in ALL])
smx_pool = np.concatenate([1 - load_ens(f"{c}_s4")["member"][0].max(1) for c in ALL])
rg = np.random.default_rng(0)
agg_epi = np.mean([routing_auc(epi_c, epi_pool, amb, np.random.default_rng(s)) for s in range(20)])
agg_smx = np.mean([routing_auc(smx_c, smx_pool, amb, np.random.default_rng(s)) for s in range(20)])

out = {
    "pool": "clean + 11 corruptions @ severity 4; random tie-break; 20 seeds",
    "per_corruption_routing_auc": per,
    "ens_epi_above_chance": {"count": sum(v > 0.5 for v in per["ens_epistemic"].values()),
                             "of": len(ALL),
                             "which": [c for c, v in per["ens_epistemic"].items() if v > 0.5]},
    "ens_beats_single_count": sum(per["ens_epistemic"][c] > per["single_maxprob"][c] for c in ALL),
    "aggregate_routing_auc": {"ens_epistemic": round(float(agg_epi), 3),
                              "single_maxprob": round(float(agg_smx), 3)},
}
Path("data/analysis_shift").mkdir(parents=True, exist_ok=True)
json.dump(out, open("data/analysis_shift/routing11.json", "w"), indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
