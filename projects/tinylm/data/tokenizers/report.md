# tinylm tokenizer suite

Fixed 500M-char future-facing corpus; vocab size is the only variable.

## Aggregate

| metric | 8k | 12k | 16k |
| --- | ---: | ---: | ---: |
| chars/token (agg) | 3.759 | 3.973 | 4.116 |
| bytes/token (agg) | 3.766 | 3.981 | 4.123 |
| tokens/doc mean | 33.8 | 31.9 | 30.8 |
| tokens/doc p95 | 121 | 113 | 108 |
| frac ≤512 tok | 0.9996 | 0.9996 | 0.9996 |
| frac ≤2048 tok | 0.9999 | 0.9999 | 0.9999 |
| vocab utilization | 0.9814 | 0.977 | 0.9697 |
| round trips ok | True | True | True |
| specials atomic | True | True | True |

## chars/token per source

| source | 8k | 12k | 16k |
| --- | ---: | ---: | ---: |
| fineweb-edu | 3.778 | 4.021 | 4.172 |
| cosmopedia-v2 | 4.221 | 4.526 | 4.715 |
| tinystories-v2 | 3.861 | 3.96 | 4.017 |
| stackexchange | 3.195 | 3.361 | 3.474 |
| wikipedia | 3.541 | 3.783 | 3.952 |
| squad | 3.853 | 4.087 | 4.278 |
| ultrachat | 4.051 | 4.291 | 4.435 |
