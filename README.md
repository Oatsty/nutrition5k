## Environment

Create a python virtual environment. Here is an example using conda.

```
conda create -n nutriseg python=3.10
conda activate nutriseg
```

Install torch and other requirements.

```
python3 -m pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113
python3 -m pip install -r OpenSeeD/requirements.txt
conda install -c conda-forge --file requirements.txt -y
conda run python3 -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
```

## Dataset

Follow the instruction from Google Nutrition-5k.

## Mask generation

Food region masks are generated using OpenSeeD instance segmentation module. Run the following command to generate the food region masks in prior to training. You may need to change the system paths to absolute paths in `script/generate_openseed_mask.py`.

```
CUDA_VISIBLE_DEVICES=0 python3 script/generate_openseed_mask.py <dataset_directory>
```

After running the command food region mask `mask.pt` should be created under every image directory.

## Training

Change the configuration file `DATA.IMGS_DIR DATA.METADATAS_PATH DATA.SPLITS_TEST_PATH DATA.SPLITS_TRAIN_PATH` should corresponds to the where the Nutrition-5k dataset was saved.

Train the model using the following command. The final model should be saved at `models/base.pt`

```
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --cfg config/models/base.yaml
```

## Pre-trained models

Pre-trained models are provided at ...

## Evaluation

Use the following command to evaluate a pre-trained model.

```
CUDA_VISIBLE_DEVICES=0 python3 src/evaluate.py --cfg <path_to_model_config>
```
