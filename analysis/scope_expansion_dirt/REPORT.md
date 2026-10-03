# Scope development results

Known 2023–2024 development results; prospective confirmation pending.

| name     |   race_macro_brier |   race_macro_log_loss |   rows |   races |
|:---------|-------------------:|----------------------:|-------:|--------:|
| G0       |        0.149715516 |           0.459922772 |  34350 |    3080 |
| G3       |        0.149687716 |           0.459852653 |  34350 |    3080 |
| R3       |        0.149994619 |           0.460612259 |  34350 |    3080 |
| baseline |        0.177702106 |           0.540413367 |  34350 |    3080 |
| routed   |        0.149715516 |           0.459922772 |  34350 |    3080 |

```json
{
  "surface": "dirt",
  "global_selection": {
    "winner": "G0",
    "point_brier_best": "G3",
    "incumbent": "G0",
    "incumbent_retained": true,
    "reason": "incumbent retained: race-macro Brier is within one paired date-clustered SE of the point-Brier-best"
  },
  "whole_selection": {
    "winner": "G0",
    "point_brier_best": "G3",
    "incumbent": "G0",
    "incumbent_retained": true,
    "reason": "incumbent retained: race-macro Brier is within one paired date-clustered SE of the point-Brier-best"
  },
  "routes": [
    {
      "min_distance_m": 1000,
      "max_distance_m": 1400,
      "model": "G0",
      "blocks": []
    },
    {
      "min_distance_m": 1401,
      "max_distance_m": 2000,
      "model": "G0",
      "blocks": []
    },
    {
      "min_distance_m": 2001,
      "max_distance_m": 2600,
      "model": "G0",
      "blocks": []
    }
  ]
}
```
