CUDA_VISIBLE_DEVICES=1 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --mask-weight 0.6 --save-path models/cross-swin-cls/grid-search/lr1e-4_mw0.6.pt
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --mask-weight 0.7 --save-path models/cross-swin-cls/grid-search/lr1e-4_mw0.7.pt
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --mask-weight 0.8 --save-path models/cross-swin-cls/grid-search/lr1e-4_mw0.8.pt
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --mask-weight 0.9 --save-path models/cross-swin-cls/grid-search/lr1e-4_mw0.9.pt
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --mask-weight 1.0 --save-path models/cross-swin-cls/grid-search/lr1e-4_mw1.0.pt
