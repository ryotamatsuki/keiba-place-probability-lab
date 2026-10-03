# Scope development results

Known 2023–2024 development results; prospective confirmation pending.

| name     |   race_macro_brier |   race_macro_log_loss |   rows |   races |
|:---------|-------------------:|----------------------:|-------:|--------:|
| G0       |        0.149311091 |           0.458055790 |  28926 |    2651 |
| G3       |        0.149315139 |           0.458173768 |  28926 |    2651 |
| R3       |        0.149301190 |           0.458165227 |  28926 |    2651 |
| baseline |        0.177907262 |           0.540206851 |  28926 |    2651 |
| routed   |        0.149016666 |           0.457404372 |  28926 |    2651 |

```json
{
  "surface": "turf",
  "global_selection": {
    "winner": "G0",
    "point_brier_best": "G0",
    "incumbent": "G0",
    "incumbent_retained": true,
    "reason": "incumbent retained: race-macro Brier is within one paired date-clustered SE of the point-Brier-best"
  },
  "whole_selection": {
    "winner": "G0",
    "point_brier_best": "R3",
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
      "model": "R3",
      "blocks": [
        "near",
        "regime",
        "course_distance"
      ]
    }
  ],
  "1200_selection": {
    "winner": "stage2",
    "point_brier_best": "stage2",
    "incumbent": "stage2",
    "incumbent_retained": true,
    "reason": "incumbent retained: race-macro Brier is within one paired date-clustered SE of the point-Brier-best"
  },
  "1200_override": {
    "scope": [
      1000,
      1400
    ],
    "blocks": [
      "regime"
    ]
  }
}
```
