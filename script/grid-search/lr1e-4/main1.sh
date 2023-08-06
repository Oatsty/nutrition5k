CUDA_VISIBLE_DEVICES=5 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --lp 0.4 --mask-weight 0.4 --save-path models/cross-swin-cls/grid-search/lr1e-4_lp0.4_mw0.4.pt
CUDA_VISIBLE_DEVICES=5 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --lp 0.4 --mask-weight 0.6 --save-path models/cross-swin-cls/grid-search/lr1e-4_lp0.4_mw0.6.pt
CUDA_VISIBLE_DEVICES=5 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --lp 0.4 --mask-weight 0.8 --save-path models/cross-swin-cls/grid-search/lr1e-4_lp0.4_mw0.8.pt
CUDA_VISIBLE_DEVICES=5 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --lp 0.6 --mask-weight 0.4 --save-path models/cross-swin-cls/grid-search/lr1e-4_lp0.6_mw0.4.pt
CUDA_VISIBLE_DEVICES=5 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --lp 0.6 --mask-weight 0.6 --save-path models/cross-swin-cls/grid-search/lr1e-4_lp0.6_mw0.6.pt
