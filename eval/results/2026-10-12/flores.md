# N-ATLaS evaluation — flores_plus:yo,ha,ig

Run at 2026-10-12 10:01:49 · 3036 items

| Group | n | base | tuned |
|---|---|---|---|
| en->ha | 1012 | chrF 48.89 | chrF 50.1 |
| en->ig | 1012 | chrF 40.45 | chrF 41.06 |
| en->yo | 1012 | chrF 26.55 | chrF 27.75 |

- **base** (`n-atlas`): 0 errors, median latency 1.325s
- **tuned** (`natlas-mt`): 0 errors, median latency 1.452s

## tuned vs base: paired bootstrap of corpus chrF

| Group | Δ chrF | 95% interval | resamples where the second system is better |
|---|---|---|---|
| en->ha | +1.21 | [+0.68, +1.75] | 100.0% of 1000 |
| en->ig | +0.61 | [+0.12, +1.11] | 99.6% of 1000 |
| en->yo | +1.20 | [+0.60, +1.82] | 100.0% of 1000 |
