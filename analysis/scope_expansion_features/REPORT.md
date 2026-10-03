# Scope development results

Known 2023–2024 development results; prospective confirmation pending.

| name                     |   race_macro_brier |   race_macro_log_loss |   rows |   races |
|:-------------------------|-------------------:|----------------------:|-------:|--------:|
| A_F0                     |        0.148958593 |           0.459405336 |   6313 |     495 |
| A_near                   |        0.149102464 |           0.459961083 |   6313 |     495 |
| A_regime                 |        0.149102464 |           0.459961083 |   6313 |     495 |
| A_course_distance        |        0.148963707 |           0.459430908 |   6313 |     495 |
| B_F0                     |        0.148774168 |           0.458557074 |   6313 |     495 |
| B_near                   |        0.148617965 |           0.458215475 |   6313 |     495 |
| B_regime                 |        0.148425742 |           0.457755146 |   6313 |     495 |
| B_regime_course_distance |        0.148516060 |           0.457926425 |   6313 |     495 |
| C_F0                     |        0.149394066 |           0.460243990 |   6313 |     495 |
| C_near                   |        0.149365415 |           0.460272546 |   6313 |     495 |
| C_regime                 |        0.149352689 |           0.460161317 |   6313 |     495 |
| C_course_distance        |        0.149368472 |           0.460221367 |   6313 |     495 |

```json
{
  "family_blocks": {
    "A": [],
    "B": [
      "regime"
    ],
    "C": []
  },
  "family_winners": {
    "A": "A_F0",
    "B": "B_regime",
    "C": "C_F0"
  },
  "winner": "B_regime",
  "winner_scope": [
    1000,
    1400
  ],
  "winner_blocks": [
    "regime"
  ],
  "block_gates": [
    {
      "candidate": "A_near",
      "reference": "A_F0",
      "mean_brier_delta": 0.00014387095285942178,
      "se_brier_delta": 0.00022758635457135546,
      "mean_log_loss_delta": 0.0005557470409122698,
      "se_log_loss_delta": 0.000596067701559035,
      "accepted": false
    },
    {
      "candidate": "A_regime",
      "reference": "A_F0",
      "mean_brier_delta": 0.00014387095285942178,
      "se_brier_delta": 0.00022758635457135546,
      "mean_log_loss_delta": 0.0005557470409122698,
      "se_log_loss_delta": 0.000596067701559035,
      "accepted": false
    },
    {
      "candidate": "A_course_distance",
      "reference": "A_F0",
      "mean_brier_delta": 5.113932946816108e-06,
      "se_brier_delta": 0.00015605417099483944,
      "mean_log_loss_delta": 2.5571390502797344e-05,
      "se_log_loss_delta": 0.0003824257317739728,
      "accepted": false
    },
    {
      "candidate": "B_near",
      "reference": "B_F0",
      "mean_brier_delta": -0.00015620359105209366,
      "se_brier_delta": 0.00019125650736717095,
      "mean_log_loss_delta": -0.0003415982866880481,
      "se_log_loss_delta": 0.00047255116642250364,
      "accepted": false
    },
    {
      "candidate": "B_regime",
      "reference": "B_F0",
      "mean_brier_delta": -0.0003484266704734136,
      "se_brier_delta": 0.00018733307315596542,
      "mean_log_loss_delta": -0.0008019279830047351,
      "se_log_loss_delta": 0.0004524104627470222,
      "accepted": true
    },
    {
      "candidate": "B_regime_course_distance",
      "reference": "B_regime",
      "mean_brier_delta": 9.031806812144457e-05,
      "se_brier_delta": 0.00016384673456555572,
      "mean_log_loss_delta": 0.00017127941403172108,
      "se_log_loss_delta": 0.0004033018415680221,
      "accepted": false
    },
    {
      "candidate": "C_near",
      "reference": "C_F0",
      "mean_brier_delta": -2.865112071098057e-05,
      "se_brier_delta": 0.0001549023711671942,
      "mean_log_loss_delta": 2.8555374041290928e-05,
      "se_log_loss_delta": 0.00038650966216903667,
      "accepted": false
    },
    {
      "candidate": "C_regime",
      "reference": "C_F0",
      "mean_brier_delta": -4.137668469380921e-05,
      "se_brier_delta": 0.0001312344829626438,
      "mean_log_loss_delta": -8.26738487257809e-05,
      "se_log_loss_delta": 0.00032450874672805313,
      "accepted": false
    },
    {
      "candidate": "C_course_distance",
      "reference": "C_F0",
      "mean_brier_delta": -2.5593498038613734e-05,
      "se_brier_delta": 0.00012513838742547596,
      "mean_log_loss_delta": -2.2623457271680353e-05,
      "se_log_loss_delta": 0.00031032125201529845,
      "accepted": false
    }
  ],
  "selection": {
    "winner": "B_regime",
    "point_brier_best": "B_regime",
    "incumbent": "A_F0",
    "incumbent_retained": false,
    "reason": "point-Brier-best selected: no incumbent was supplied or the incumbent fell outside the paired one-SE Brier retention band"
  }
}
```
