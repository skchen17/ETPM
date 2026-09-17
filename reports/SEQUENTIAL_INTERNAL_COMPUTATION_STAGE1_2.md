# Sequential Internal Computation — Stage 1.2

Graph writes are followed by H scrub. Each transition emits exactly one query and receives one memory-read vector. L=6–8 is OOD.

| length_regime | path_length | internal_tick | accuracy | query_path_alignment | confidence | entropy |
|---|---|---|---|---|---|---|
| ood_longer | 6 | 0 | 0.5142 | 0.9620 | 0.6529 | 0.6093 |
| ood_longer | 6 | 1 | 0.5049 | 0.8826 | 0.5640 | 0.6737 |
| ood_longer | 6 | 2 | 0.5430 | 0.8647 | 0.5532 | 0.6802 |
| ood_longer | 6 | 4 | 0.5540 | 0.8254 | 0.5639 | 0.6717 |
| ood_longer | 6 | 8 | 0.5261 | 0.7710 | 0.5654 | 0.6689 |
| ood_longer | 6 | 16 | 0.5193 | 0.7104 | 0.5679 | 0.6649 |
| ood_longer | 7 | 0 | 0.4985 | 0.9614 | 0.6553 | 0.6080 |
| ood_longer | 7 | 1 | 0.5095 | 0.8817 | 0.5640 | 0.6739 |
| ood_longer | 7 | 2 | 0.5437 | 0.8640 | 0.5553 | 0.6797 |
| ood_longer | 7 | 4 | 0.5359 | 0.8047 | 0.5658 | 0.6704 |
| ood_longer | 7 | 8 | 0.5081 | 0.7523 | 0.5692 | 0.6671 |
| ood_longer | 7 | 16 | 0.4998 | 0.6941 | 0.5732 | 0.6614 |
| ood_longer | 8 | 0 | 0.5103 | 0.9614 | 0.6566 | 0.6063 |
| ood_longer | 8 | 1 | 0.5139 | 0.8802 | 0.5654 | 0.6730 |
| ood_longer | 8 | 2 | 0.5557 | 0.8616 | 0.5555 | 0.6797 |
| ood_longer | 8 | 4 | 0.5518 | 0.7973 | 0.5679 | 0.6687 |
| ood_longer | 8 | 8 | 0.5176 | 0.7483 | 0.5710 | 0.6655 |
| ood_longer | 8 | 16 | 0.5156 | 0.6946 | 0.5739 | 0.6612 |
| train_support | 1 | 0 | 0.5022 | 0.9615 | 0.5903 | 0.6457 |
| train_support | 1 | 1 | 0.5010 | 0.9028 | 0.5823 | 0.6558 |
| train_support | 1 | 2 | 0.5132 | 0.8683 | 0.5587 | 0.6741 |
| train_support | 1 | 4 | 0.5730 | 0.7744 | 0.5506 | 0.6760 |
| train_support | 1 | 8 | 0.5466 | 0.6731 | 0.5563 | 0.6732 |
| train_support | 1 | 16 | 0.5261 | 0.6033 | 0.5649 | 0.6722 |
| train_support | 2 | 0 | 0.4797 | 0.9617 | 0.6051 | 0.6401 |
| train_support | 2 | 1 | 0.5054 | 0.8995 | 0.5648 | 0.6705 |
| train_support | 2 | 2 | 0.5195 | 0.8862 | 0.5463 | 0.6802 |
| train_support | 2 | 4 | 0.5552 | 0.8215 | 0.5514 | 0.6747 |
| train_support | 2 | 8 | 0.5188 | 0.7194 | 0.5576 | 0.6694 |
| train_support | 2 | 16 | 0.5073 | 0.6546 | 0.5636 | 0.6653 |
| train_support | 3 | 0 | 0.4990 | 0.9617 | 0.6276 | 0.6266 |
| train_support | 3 | 1 | 0.5081 | 0.8897 | 0.5620 | 0.6735 |
| train_support | 3 | 2 | 0.5374 | 0.8875 | 0.5489 | 0.6805 |
| train_support | 3 | 4 | 0.5486 | 0.8302 | 0.5573 | 0.6731 |
| train_support | 3 | 8 | 0.5186 | 0.7348 | 0.5627 | 0.6676 |
| train_support | 3 | 16 | 0.5059 | 0.6709 | 0.5663 | 0.6636 |
| train_support | 4 | 0 | 0.5068 | 0.9616 | 0.6405 | 0.6172 |
| train_support | 4 | 1 | 0.5039 | 0.8857 | 0.5632 | 0.6737 |
| train_support | 4 | 2 | 0.5403 | 0.8734 | 0.5504 | 0.6809 |
| train_support | 4 | 4 | 0.5500 | 0.8369 | 0.5597 | 0.6724 |
| train_support | 4 | 8 | 0.5159 | 0.7519 | 0.5634 | 0.6679 |
| train_support | 4 | 16 | 0.5088 | 0.6926 | 0.5665 | 0.6641 |
| train_support | 5 | 0 | 0.5132 | 0.9617 | 0.6501 | 0.6110 |
| train_support | 5 | 1 | 0.5032 | 0.8852 | 0.5628 | 0.6738 |
| train_support | 5 | 2 | 0.5430 | 0.8676 | 0.5535 | 0.6799 |
| train_support | 5 | 4 | 0.5488 | 0.8425 | 0.5620 | 0.6727 |
| train_support | 5 | 8 | 0.5198 | 0.7580 | 0.5656 | 0.6685 |
| train_support | 5 | 16 | 0.5115 | 0.6961 | 0.5691 | 0.6638 |

G16: **FAIL**; hard bottleneck audit=True.
