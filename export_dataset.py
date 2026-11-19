"""Turn saved episodes into a supervised dataset (observation → expert command).

Typical flow:

    python benchmark.py --suite dataset --jobs 8 --out runs/dataset_pp
    python export_dataset.py runs/dataset_pp --out datasets/pp_v1
    python export_dataset.py runs/dataset_pp --out datasets/pp_v1 --holdout-maps map07

Output (datasets/<name>/):

    data.npz    X (N, F) features, Y (N, 2) expert [v, ω], episode (N,) episode index,
                split (N,) 0 = train / 1 = val / 2 = test, plus raw per-step arrays
                (obs_pose, ref_pose, ref_control, last_command, t) for custom features
    meta.json   feature names, the episodes used (map, scenario, controller, seed), split sizes

Splits are by *episode* (never by time step), so samples of one trajectory
never end up in both train and test. --holdout-maps puts whole maps in test
to measure generalisation to unseen paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FEATURES = ("e_x", "e_y", "sin_e_theta", "cos_e_theta", "v_ref", "omega_ref", "v_prev", "omega_prev")


def episode_features(ep) -> dict[str, np.ndarray]:
    """Per control tick: what the controller saw (pose measurement, reference) and did."""
    from diffdrive.core.geometry import body_frame_error

    obs, ref, u_ref, cmd = ep["obs_pose"], ep["ref_pose"], ep["ref_control"], ep["command"]
    ok = np.all(np.isfinite(obs), axis=1)  # before the first measurement arrives
    last = np.vstack([np.zeros((1, 2)), cmd[:-1]])  # command applied before this tick
    e = body_frame_error(obs, ref)
    X = np.column_stack([e[:, 0], e[:, 1], np.sin(e[:, 2]), np.cos(e[:, 2]), u_ref, last])
    return {
        "X": X[ok],
        "Y": cmd[ok],
        "obs_pose": obs[ok],
        "ref_pose": ref[ok],
        "ref_control": u_ref[ok],
        "last_command": last[ok],
        "t": ep["ctrl_t"][ok],
    }


def main(args: argparse.Namespace) -> Path:
    from diffdrive.evaluation import compute_metrics
    from diffdrive.sim import Episode, find_episodes

    paths = [p for d in args.runs for p in find_episodes(d)]
    if not paths:
        raise SystemExit("no episodes found")

    chunks: dict[str, list[np.ndarray]] = {}
    episodes, ep_index = [], []
    skipped = 0
    for p in paths:
        ep = Episode.load(p)
        if args.controller and ep.label not in args.controller:
            continue
        success = bool((ep.metrics or compute_metrics(ep)).get("success"))
        if args.only_success and not success:
            skipped += 1
            continue
        f = episode_features(ep)
        for k, v in f.items():
            chunks.setdefault(k, []).append(v)
        ep_index.append(np.full(len(f["X"]), len(episodes)))
        episodes.append(
            {
                "path": str(p),
                "map": ep.map_name,
                "scenario": ep.meta.get("scenario_label", ep.scenario_name),
                "controller": ep.label,
                "seed": ep.meta["seed"],
                "success": success,
                "steps": len(f["X"]),
            }
        )
    if not episodes:
        raise SystemExit("no episodes left after filtering")

    # split by episode: hold-out maps → test; the rest shuffled into train / val / test
    rng = np.random.default_rng(args.seed)
    split_of_ep = np.zeros(len(episodes), dtype=np.int8)
    pool = [i for i, e in enumerate(episodes) if e["map"] not in (args.holdout_maps or [])]
    split_of_ep[[i for i in range(len(episodes)) if i not in set(pool)]] = 2
    order = rng.permutation(pool)
    n_val = int(round(args.val * len(order)))
    n_test = 0 if args.holdout_maps else int(round(args.test * len(order)))
    split_of_ep[order[:n_val]] = 1
    split_of_ep[order[n_val : n_val + n_test]] = 2
    for e, s in zip(episodes, split_of_ep, strict=True):
        e["split"] = ["train", "val", "test"][s]

    data = {k: np.concatenate(v) for k, v in chunks.items()}
    data["episode"] = np.concatenate(ep_index)
    data["split"] = split_of_ep[data["episode"]]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "data.npz", **data)
    sizes = {name: int((data["split"] == i).sum()) for i, name in enumerate(("train", "val", "test"))}
    meta = {
        "features": list(FEATURES),
        "targets": ["v", "omega"],
        "sources": [str(r) for r in args.runs],
        "holdout_maps": args.holdout_maps or [],
        "split_seed": args.seed,
        "samples": sizes,
        "episodes": episodes,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(episodes)} episodes ({skipped} failed skipped) → {sum(sizes.values())} samples")
    print(f"  train {sizes['train']}   val {sizes['val']}   test {sizes['test']}")
    print(f"saved: {out}/  (data.npz, meta.json)")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("runs", nargs="+", help="benchmark output dir(s) or episode dir(s)")
    p.add_argument("--out", required=True, help="dataset directory, e.g. datasets/pp_v1")
    p.add_argument("--controller", action="append", help="only use this expert label (repeatable)")
    p.add_argument("--only-success", action="store_true", help="drop failed episodes")
    p.add_argument("--holdout-maps", nargs="+", metavar="MAP", help="maps used only for test")
    p.add_argument("--val", type=float, default=0.1, help="fraction of episodes for validation")
    p.add_argument("--test", type=float, default=0.1, help="fraction for test (no holdout maps)")
    p.add_argument("--seed", type=int, default=0, help="split shuffle seed")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
