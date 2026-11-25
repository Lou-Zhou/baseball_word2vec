# Sequential embeddings: frozen, then fine-tuned

Seasons 2022, 2023, 2024, 2025, 2026; base season 2022 trained fully. Each later season starts from the previous season's heads: phase 1 fits fresh player vectors with the heads frozen (up to 10 epochs), phase 2 fine-tunes everything with the heads at 0.1 x the learning rate (up to 10 epochs).
Continuity uses batters with >= 1000 training pitches in both seasons; the wOBA columns use batters with >= 300 PA in both.

## How to read this

- **mean_dimension_r**: correlation of each embedding dimension with itself the next season, averaged.
- **mean_cosine** vs **shuffled_cosine**: how close a batter's two vectors are, against random pairs.
- **reidentified_top1 / top5**: share of batters whose own next-season vector is the closest (top 5) of all returning batters. Chance is about 1 / batters.
- **phase1 rows** are the vectors before the heads moved: if the final rows are much lower, fine-tuning drifted the space.
- **readout_woba_r**: the value-head wOBA readout year to year, next to actual wOBA and xwOBA. **readout_a_vs_actual_woba_b_r** is a no-aging forecast: this year's readout vs next year's wOBA.
- **cross_decode_r** (season table): last season's vectors read out through last season's heads vs this season's heads. Near 1 = the heads still decode the old vectors the same way.

## Year-to-year batter continuity

| pair | vectors | batters | mean_dimension_r | mean_cosine | shuffled_cosine | reidentified_top1 | reidentified_top5 | median_rank_percentile |
|---|---|---|---|---|---|---|---|---|
| 2022->2023 | final | 204 | 0.3781 | 0.3665 | 0.0248 | 0.3039 | 0.6225 | 0.0147 |
| 2022->2023 | phase1 (frozen heads) | 204 | 0.3782 | 0.3686 | 0.0289 | 0.2990 | 0.6275 | 0.0147 |
| 2023->2024 | final | 212 | 0.6104 | 0.5895 | -0.0197 | 0.4717 | 0.7594 | 0.0094 |
| 2023->2024 | phase1 (frozen heads) | 212 | 0.6364 | 0.6146 | -0.0168 | 0.5094 | 0.7642 | 0.0047 |
| 2024->2025 | final | 209 | 0.6522 | 0.6285 | -0.0138 | 0.5263 | 0.8421 | 0.0048 |
| 2024->2025 | phase1 (frozen heads) | 209 | 0.6522 | 0.6285 | -0.0138 | 0.5263 | 0.8421 | 0.0048 |
| 2025->2026 | final | 202 | 0.6721 | 0.6529 | -0.0016 | 0.4653 | 0.7921 | 0.0099 |
| 2025->2026 | phase1 (frozen heads) | 202 | 0.6884 | 0.6691 | -0.0038 | 0.4851 | 0.7921 | 0.0099 |

## Year-to-year wOBA

| pair | woba_batters | readout_woba_r | actual_woba_r | actual_xwoba_r | readout_a_vs_actual_woba_b_r |
|---|---|---|---|---|---|
| 2022->2023 | 199.0000 | 0.5105 | 0.5492 | 0.7307 | 0.5309 |
| 2023->2024 | 203.0000 | 0.4772 | 0.4683 | 0.6899 | 0.4932 |
| 2024->2025 | 198.0000 | 0.5175 | 0.5277 | 0.6535 | 0.5079 |
| 2025->2026 | 192.0000 | 0.3829 | 0.3957 | 0.6190 | 0.4186 |

## Drift from the base season (2022 -> each later season)

| pair | batters | mean_dimension_r | mean_cosine | reidentified_top1 |
|---|---|---|---|---|
| 2022->2023 | 204 | 0.3781 | 0.3665 | 0.3039 |
| 2022->2024 | 171 | 0.3437 | 0.3451 | 0.2690 |
| 2022->2025 | 145 | 0.3418 | 0.3282 | 0.2345 |
| 2022->2026 | 109 | 0.3191 | 0.3141 | 0.2110 |

## Seasons

| season | batters | phase1_epochs | phase2_epochs | finetune_kept | test_log_loss | readout_woba_rmse | readout_woba_r | phase1_valid_loss | phase2_valid_loss | cross_decode_r | cross_decode_mean_abs_diff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022.0000 | 589 | nan | 21.0000 | None | 1.1270 | 0.0148 | 0.9057 | nan | nan | nan | nan |
| 2023.0000 | 590 | 10.0000 | 6.0000 | True | 1.1093 | 0.0102 | 0.9554 | 3.1554 | 3.1485 | 0.9895 | 0.0093 |
| 2024.0000 | 573 | 10.0000 | 8.0000 | True | 1.1100 | 0.0102 | 0.9565 | 3.0857 | 3.0848 | 0.9981 | 0.0032 |
| 2025.0000 | 579 | 8.0000 | 4.0000 | False | 1.1178 | 0.0084 | 0.9668 | 3.1003 | 3.1004 | 1.0000 | 0.0031 |
| 2026.0000 | 577 | 9.0000 | 5.0000 | True | 1.0949 | 0.0107 | 0.9327 | 3.0358 | 3.0346 | 0.9942 | 0.0045 |
