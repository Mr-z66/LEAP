| signal | model | feature_dim | error_roc_auc | error_pr_auc | best_error_f1 | recall_at_trigger_0p05 | precision_at_trigger_0p05 | recall_at_trigger_0p1 | precision_at_trigger_0p1 | recall_at_trigger_0p2 | precision_at_trigger_0p2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| entropy | raw_score | 1 | 0.5705 | 0.3383 | 0.3975 | 0.1139 | 0.5455 | 0.1835 | 0.4394 | 0.2658 | 0.3206 |
| neg_top1_prob | raw_score | 1 | 0.5619 | 0.3014 | 0.3965 | 0.0759 | 0.3636 | 0.1519 | 0.3636 | 0.2658 | 0.3206 |
| neg_margin | raw_score | 1 | 0.5579 | 0.2806 | 0.3970 | 0.0570 | 0.2727 | 0.1203 | 0.2879 | 0.2722 | 0.3282 |
| boundary_logreg | logreg | 1536 | 0.8210 | 0.6992 | 0.6309 | 0.2089 | 1.0000 | 0.3671 | 0.8788 | 0.5633 | 0.6794 |
| mean_logreg | logreg | 1536 | 0.8409 | 0.7047 | 0.6400 | 0.2025 | 0.9697 | 0.3608 | 0.8636 | 0.5759 | 0.6947 |
| boundary+mean_logreg | logreg | 3072 | 0.8614 | 0.7485 | 0.6840 | 0.2089 | 1.0000 | 0.3608 | 0.8636 | 0.5949 | 0.7176 |
