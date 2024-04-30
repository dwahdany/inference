1. Install environment from env.yaml
```bash
conda env create -f env.yaml
```

2. Activate environment
```bash
conda activate inference
```

3. Configure
    - `out_dir`: set to the output directory
    - `dataset_name`: leave unchanged
    - `dataset_dir`: set to the directory that contains imagenet (should contain the unpacked `train_data_batch_x` files)
    - `size_target`: set to desired upscaling
    - `variants`: set to desired models (supports pytorch `vit_b_16`, `vit_l_16`, `vit_h_14`, as timm and huggingface identifiers)
    - `devices`: set to desired devices, supports virtually unlimited GPUs to parallelize the inference.

3. Inference
```bash
python predict.py
```