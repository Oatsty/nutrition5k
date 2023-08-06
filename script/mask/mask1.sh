CUDA_VISIBLE_DEVICES=0 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --mask-weight 0.0 --save-path models/cross-swin-cls/grid-search/lr1e-4_mw0.0.pt
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --mask-weight 0.1 --save-path models/cross-swin-cls/grid-search/lr1e-4_mw0.1.pt
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --mask-weight 0.2 --save-path models/cross-swin-cls/grid-search/lr1e-4_mw0.2.pt
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --mask-weight 0.3 --save-path models/cross-swin-cls/grid-search/lr1e-4_mw0.3.pt
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --mask-weight 0.4 --save-path models/cross-swin-cls/grid-search/lr1e-4_mw0.4.pt
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --cfg config/models/cross-swin-cls/base/base.yaml --lr 0.0001 --mask-weight 0.5 --save-path models/cross-swin-cls/grid-search/lr1e-4_mw0.5.pt
