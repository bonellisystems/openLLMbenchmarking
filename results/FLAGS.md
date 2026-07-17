# FLAGS -- spread>2 and calibration-drift triage (table-time, regenerated every run)

## Spread flags (max-min > 2)

| Packet | Task | Run | Model | Scores by judge |
|---|---|---|---|---|
| ed4dca4b3e60 | b1.cybersecurity-01 | 1 | gemma-4-26b-a4b | claude=4, codex=2, gemini=1 |
| ed4dca4b3e60 | b1.cybersecurity-01 | 1 | gpt-oss-20b | claude=7, codex=3, gemini=3 |
| 724d2cd6fbb1 | b1.cybersecurity-03 | 1 | gpt-oss-20b | claude=3, codex=2, gemini=0 |
| 1136c16322da | b1.cybersecurity-04 | 1 | CAL-strong | claude=9, codex=8, gemini=4 |
| 1136c16322da | b1.cybersecurity-04 | 1 | gpt-oss-20b | claude=5, codex=4, gemini=9 |
| 6957320d33b0 | b1.cybersecurity-06 | 1 | CAL-strong | codex=6, gemini=10 |
| 6957320d33b0 | b1.cybersecurity-06 | 1 | CAL-weak | codex=3, gemini=0 |
| f673c951826d | b1.it_infra-01 | 1 | CAL-strong | claude=8, codex=6, gemini=10 |
| 2d805540514c | b1.it_infra-02 | 1 | CAL-strong | claude=9, codex=7, gemini=10 |
| 318f79159ec2 | b1.it_infra-03 | 1 | CAL-strong | claude=8, codex=7, gemini=10 |
| 318f79159ec2 | b1.it_infra-03 | 1 | CAL-weak | claude=2, codex=3, gemini=0 |
| 47011b266350 | b1.it_infra-04 | 1 | gpt-oss-20b | claude=6, codex=3, gemini=3 |
| 6e87630f7564 | b1.it_infra-05 | 1 | gpt-oss-20b | claude=6, codex=3, gemini=5 |
| 58ac56010ae6 | b1.it_infra-06 | 1 | gpt-oss-20b | claude=6, codex=3, gemini=4 |
| 267645a90d35 | b1.it_infra-07 | 1 | gpt-oss-20b | claude=7, codex=4, gemini=8 |
| e3d598b71edd | b1.it_infra-08 | 1 | gemma-4-26b-a4b | claude=8, codex=5, gemini=6 |

## Drift flags (|median - ref| > tolerance)

| Packet | Task | Run | CAL | Median | Ref | Delta |
|---|---|---|---|---|---|---|
| 9eba01d7d56a96ff5e34ee7be0e98c2341bb4060c22a574bd2bd84c865f3fc96 | b1.cybersecurity-01 | 1 | strong | 5.0 | 9.0 | -4.0 |
| 9eba01d7d56a96ff5e34ee7be0e98c2341bb4060c22a574bd2bd84c865f3fc96 | b1.cybersecurity-01 | 1 | weak | 5.0 | 2.0 | +3.0 |
| 47011b266350145ba86602e1ffcafd0afa8bade12268970345212f68d5c97727 | b1.it_infra-04 | 1 | strong | 6.0 | 9.0 | -3.0 |
| 58ac56010ae6d14e94fd3e475a56dc1374f021d660031bc536be51e18e47968d | b1.it_infra-06 | 1 | strong | 7.0 | 9.0 | -2.0 |
| e3d598b71edd50efdf4017cb9234fbdff76600e43798c169eab462915db239bc | b1.it_infra-08 | 1 | strong | 7.0 | 9.0 | -2.0 |
