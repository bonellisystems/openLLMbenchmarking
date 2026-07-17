# Serving (Battery 5) — timing_authoritative rows only

| Model | Condition | decode t/s | PP t/s | TTFT ms |
|---|---|---|---|---|
| unsloth/gpt-oss-20b-GGUF | runtime=fork;spec=ngram32;kv=q8;cond=PEAK | 158.9 | 411.9 | 206 |
| unsloth/gpt-oss-20b-GGUF | runtime=fork;spec=ngram32;kv=q8;cond=PEAK;conc=16 | 36.0 | 0.0 | 0 |
| unsloth/gpt-oss-20b-GGUF | runtime=fork;spec=ngram32;kv=q8;cond=PEAK;conc=2 | 132.0 | 0.0 | 0 |
| unsloth/gpt-oss-20b-GGUF | runtime=fork;spec=ngram32;kv=q8;cond=PEAK;conc=4 | 83.5 | 0.0 | 0 |
| unsloth/gpt-oss-20b-GGUF | runtime=fork;spec=ngram32;kv=q8;cond=PEAK;conc=8 | 55.3 | 0.0 | 0 |
| unsloth/gpt-oss-20b-GGUF | runtime=fork;spec=ngram32;kv=q8;cond=SUSTAINED32K | 124.6 | 4928.5 | 3844 |
| unsloth/gpt-oss-20b-GGUF | runtime=fork;spec=off;kv=q8;cond=PEAK | 159.0 | 458.0 | 186 |
| unsloth/gpt-oss-20b-GGUF | runtime=fork;spec=off;kv=q8;cond=SUSTAINED32K | 129.9 | 4934.8 | 3839 |
