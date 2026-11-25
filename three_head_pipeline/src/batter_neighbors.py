"""Nearest-neighbor explorer for a run's batter embedding space.

    python src/batter_neighbors.py --run runs/2024
    python src/batter_neighbors.py --run runs/2024 --batter "Judge"      # also print to the terminal

Writes <run>/batter_neighbors.html: a self-contained page (open it in any
browser) where you search a batter and see the closest batters by cosine
similarity, next to a scouting line for each (handedness, K%, BB%, whiff%,
chase%, exit velocity, launch angle, hard-hit%, wOBA, xwOBA) so you can judge
whether neighbors are alike. Vectors are centered on the pitch-weighted mean
before the cosine, so similarity is about how batters differ from the average.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paths import run_data_paths, savant_names, savant_woba

STRIKEOUTS = {"strikeout", "strikeout_double_play"}
WALKS = {"walk", "intent_walk"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
CONTACT = {"foul", "foul_tip", "foul_bunt", "hit_into_play", "bunt_foul_tip"}
STAT_COLUMNS = ["mlbam_id", "name", "bats", "train_pitches", "pa", "k_pct", "bb_pct", "whiff_pct", "chase_pct",
                "ev", "la", "hard_pct", "woba", "xwoba"]


def batter_profiles(paths):
    """Per-batter scouting stats from the raw pitches."""
    p = pd.concat([pd.read_parquet(path, columns=[
        "batter", "stand", "description", "events", "launch_speed", "launch_angle",
        "plate_x", "plate_z", "sz_top", "sz_bot"]) for path in paths], ignore_index=True)
    swing = p.description.isin(WHIFFS | CONTACT)
    located = p.plate_x.notna() & p.plate_z.notna()
    out_zone = located & ~((p.plate_x.abs() <= 0.83) & p.plate_z.between(p.sz_bot, p.sz_top))
    in_play = p.description.eq("hit_into_play")
    p = p.assign(pa=p.events.notna(), k=p.events.isin(STRIKEOUTS), bb=p.events.isin(WALKS), swing=swing,
                 whiff=p.description.isin(WHIFFS), out_zone=out_zone, chase=out_zone & swing,
                 ev=p.launch_speed.where(in_play), la=p.launch_angle.where(in_play),
                 hard=(p.launch_speed >= 95).where(in_play & p.launch_speed.notna()))
    g = p.groupby("batter")
    stats = pd.DataFrame({
        "bats": g.stand.agg(lambda s: "S" if s.nunique() > 1 else s.iloc[0]),
        "k_pct": g.k.sum() / g.pa.sum(), "bb_pct": g.bb.sum() / g.pa.sum(),
        "whiff_pct": g.whiff.sum() / g.swing.sum(), "chase_pct": g.chase.sum() / g.out_zone.sum(),
        "ev": g.ev.mean(), "la": g.la.mean(), "hard_pct": g.hard.mean(),
    })
    return stats.rename_axis("mlbam_id").reset_index()


def build_table(run_dir):
    """(batter table with stats, unit-length centered vectors, season)."""
    paths = run_data_paths(run_dir)
    season = int(pd.to_datetime(pd.read_parquet(paths[0], columns=["game_date"]).game_date).dt.year.max())
    emb = pd.read_csv(run_dir / "batter_embeddings.csv")
    dims = [c for c in emb.columns if c.startswith("dim_")]
    table = (emb.merge(batter_profiles(paths), on="mlbam_id", how="left")
                .merge(savant_woba(season).reset_index(), on="mlbam_id", how="left"))
    names = savant_names(season)
    table["name"] = [names.get(int(i), f"ID {int(i)}") for i in table.mlbam_id]
    x = table[dims].to_numpy(dtype=np.float64)
    x = x - np.average(x, axis=0, weights=table.train_pitches)
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    return table, x, season


def neighbors(table, unit, row, k, min_pitches):
    sims = unit @ unit[row]
    keep = (table.train_pitches >= min_pitches).to_numpy()
    keep[row] = False
    order = np.argsort(-np.where(keep, sims, -np.inf))[:k]
    return table.iloc[order].assign(similarity=sims[order])


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Batter Neighbors</title>
<style>
:root { --bg:#fbfaf8; --panel:#fff; --text:#1d1d1b; --muted:#6b6a66; --line:#e4e2dc;
        --accent:#2f5d8a; --hi:#e6efe7; --lo:#f6e7e3; --query:#eef3f9; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --panel:#1f1f1d; --text:#ecebe7;
        --muted:#9c9a94; --line:#33322f; --accent:#8fb6dc; --hi:#203226; --lo:#3a2522; --query:#1f2a36; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1180px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:20px; margin:0 0 4px; }
p.sub { color:var(--muted); margin:0 0 18px; }
.controls { display:flex; flex-wrap:wrap; gap:12px 18px; align-items:end; background:var(--panel);
            border:1px solid var(--line); border-radius:10px; padding:14px; margin-bottom:16px; }
label { display:flex; flex-direction:column; gap:4px; font-size:12px; color:var(--muted); }
input { font:inherit; color:var(--text); background:var(--bg); border:1px solid var(--line);
        border-radius:6px; padding:6px 8px; }
#search { width:300px; max-width:100%; }
.wrap { overflow-x:auto; background:var(--panel); border:1px solid var(--line); border-radius:10px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { padding:7px 10px; text-align:right; white-space:nowrap; border-bottom:1px solid var(--line); }
th { font-size:12px; color:var(--muted); font-weight:600; background:var(--panel); }
td.l, th.l { text-align:left; }
tr.query td { background:var(--query); font-weight:600; border-bottom:2px solid var(--accent); }
td.hi { background:var(--hi); } td.lo { background:var(--lo); }
.bar { display:inline-block; height:8px; background:var(--accent); border-radius:4px; vertical-align:middle; margin-right:6px; }
a { color:var(--accent); cursor:pointer; text-decoration:none; }
.note { color:var(--muted); font-size:12px; margin-top:10px; }
</style></head><body><main>
<h1>Batter embedding neighbors</h1>
<p class="sub">__SUBTITLE__</p>
<div class="controls">
  <label>Batter<input id="search" list="names" placeholder="Type a name, e.g. Aaron Judge"></label>
  <datalist id="names"></datalist>
  <label>Min training pitches<input id="minp" type="number" value="__MINP__" step="100" min="0" style="width:110px"></label>
  <label>Show<input id="k" type="number" value="15" min="1" max="100" style="width:70px"></label>
</div>
<div class="wrap"><table><thead><tr>
  <th>#</th><th class="l">Batter</th><th class="l">Bats</th><th class="l">Cosine</th>
  <th>Pitches</th><th>PA</th><th>K%</th><th>BB%</th><th>Whiff%</th><th>Chase%</th>
  <th>EV</th><th>LA</th><th>Hard%</th><th>wOBA</th><th>xwOBA</th>
</tr></thead><tbody id="rows"></tbody></table></div>
<p class="note">Cosine similarity of centered embeddings. Shading marks rate stats more than 25% above (green) or
below (red) the searched batter's value. Click a neighbor to search them. Few training pitches = noisy vector.</p>
</main>
<script>
const D = __DATA__;
const rows = D.rows, V = D.vectors;
const byName = new Map(rows.map((r, i) => [r.name, i]));
document.getElementById("names").innerHTML = rows.map(r => `<option value="${r.name}">`).join("");
const pct = v => v == null ? "" : (100 * v).toFixed(1);
const f1 = v => v == null ? "" : v.toFixed(1);
const f3 = v => v == null ? "" : v.toFixed(3).replace(/^0/, "");
const int = v => v == null ? "" : Math.round(v).toLocaleString();
const COLS = [["train_pitches", int], ["pa", int], ["k_pct", pct], ["bb_pct", pct], ["whiff_pct", pct],
  ["chase_pct", pct], ["ev", f1], ["la", f1], ["hard_pct", pct], ["woba", f3], ["xwoba", f3]];
const SHADE = new Set(["k_pct", "bb_pct", "whiff_pct", "chase_pct", "hard_pct"]);
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const search = document.getElementById("search"), minp = document.getElementById("minp"),
      kInput = document.getElementById("k");
let current = null;

function line(q, r, s, n) {
  const cells = COLS.map(([c, fmt]) => {
    let shade = "";
    if (n && SHADE.has(c) && r[c] != null && q[c]) {
      const ratio = r[c] / q[c]; shade = ratio > 1.25 ? "hi" : ratio < 0.8 ? "lo" : "";
    }
    return `<td class="${shade}">${fmt(r[c])}</td>`;
  }).join("");
  const bar = s == null ? "" : `<span class="bar" style="width:${Math.max(0, s) * 60}px"></span>${s.toFixed(3)}`;
  const name = n ? `<a data-name="${r.name}">${r.name}</a>` : r.name;
  return `<tr class="${n ? "" : "query"}"><td>${n || ""}</td><td class="l">${name}</td>`
    + `<td class="l">${r.bats || ""}</td><td class="l">${bar}</td>${cells}</tr>`;
}
function render(i) {
  const q = rows[i], sims = V.map(v => dot(v, V[i]));
  const cand = rows.map((_, j) => j).filter(j => j !== i && rows[j].train_pitches >= (+minp.value || 0));
  cand.sort((a, b) => sims[b] - sims[a]);
  document.getElementById("rows").innerHTML = line(q, q, null, 0)
    + cand.slice(0, +kInput.value || 15).map((j, n) => line(q, rows[j], sims[j], n + 1)).join("");
}
function go() { const i = byName.get(search.value); if (i !== undefined) { current = i; render(i); } }
search.addEventListener("input", go);
[minp, kInput].forEach(el => el.addEventListener("input", () => current !== null && render(current)));
document.getElementById("rows").addEventListener("click", e => {
  const a = e.target.closest("a[data-name]"); if (!a) return;
  search.value = a.dataset.name; go(); window.scrollTo(0, 0);
});
const start = rows.reduce((b, r, i) => r.train_pitches > rows[b].train_pitches ? i : b, 0);
search.value = rows[start].name; go();
</script></body></html>
"""


def write_html(table, unit, path, run_dir, season, min_pitches):
    stats = table[STAT_COLUMNS]
    records = stats.astype(object).where(stats.notna(), None)
    data = {"rows": records.to_dict(orient="records"), "vectors": np.round(unit, 4).tolist()}
    subtitle = f"{len(table):,} batters, {season} season, from {Path(run_dir).as_posix()}."
    html = (PAGE.replace("__DATA__", json.dumps(data, default=lambda v: v.item()))
                .replace("__SUBTITLE__", subtitle).replace("__MINP__", str(min_pitches)))
    path.write_text(html, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="A train.py output folder")
    parser.add_argument("--output", type=Path, default=None, help="Default <run>/batter_neighbors.html")
    parser.add_argument("--min-pitches", type=int, default=500,
                        help="Default minimum training pitches for neighbors (adjustable in the page)")
    parser.add_argument("--batter", default=None, help="Also print neighbors for names containing this text")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    table, unit, season = build_table(args.run)
    output = args.output or args.run / "batter_neighbors.html"
    write_html(table, unit, output, args.run, season, args.min_pitches)
    print(f"Wrote {output} ({len(table):,} batters)")

    if args.batter:
        hits = np.flatnonzero(table.name.str.contains(args.batter, case=False).to_numpy())
        if len(hits) == 0:
            print(f"No batter matches {args.batter!r}")
        show = ["name", "bats", "similarity", "train_pitches", "k_pct", "bb_pct", "whiff_pct", "ev", "la",
                "woba", "xwoba"]
        for row in hits:
            q = table.iloc[row]
            print(f"\n{q['name']}  ({int(q.train_pitches)} pitches, wOBA {q.woba:.3f}, xwOBA {q.xwoba:.3f})")
            near = neighbors(table, unit, row, args.k, args.min_pitches)
            print(near[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
