# SCANet

- 🎉🎉🎉 This paper has been accepted by the [31th International Conference on Neural Information Processing (ICONIP)](https://iconip2024.org/)!

The official PyTorch implementation of Paper "Split Coordinate Attention for Building Footprint Extraction".

## Usage
### Requirements
1. Create a conda environment with python 3.11.
    ```Bash
    conda create -n YourEnvName python=3.11
    ```
2. Activate the environment.
    ```Bash
    conda activate YourEnvName
    ```
3. Install the dependencies.
   ```Bash
   pip install -r requirements.txt
   ```
4. Clone the repository and change the directory.
   ```Bash
   git clone https://github.com/AiEson/SCANet.git
   cd SCANet
   ```
### Training
#### Dataset Preparation
1. WHU Dataset: http://gpcv.whu.edu.cn/data/building_dataset.html
2. Massachusetts Buildings Dataset: 
   - Original: https://www.cs.toronto.edu/~vmnih/data/
   - Kaggle: https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset
---
#### Run the following command to train the model.
- For WHU Dataset:
  ```Bash
  bash scripts/SCANet_whu.sh 1 12344
  ```
- For Massachusetts Buildings Dataset:
  ```Bash
   bash scripts/SCANet_mass.sh 1 12344
   ```

## The Core Code of SCANet
We utilized the [Segmentation Models](https://segmentation-models-pytorch.readthedocs.io/en/latest/index.html) library as our segmentation model repository and made modifications to its source code to adapt it to our model.

The modified code is placed in the `models/my_smp/` directory, with implementations of various encoders located under the `models/my_smp/encoders/` directory. We implemented SCANet in the [`models/my_smp/encoders/scanet.py`](https://github.com/AiEson/SCANet/blob/main/models/my_smp/encoders/scanet.py) file, enabling easy replacement of the decoder without any modifications to the decoder.

### Call SCANet Model Anywhere
```Python
import models.my_smp as smp
# UNet as the decoder
model = smp.Unet(
   encoder_name='scanet-101e',
   classes=1,
   encoder_weights=None,
   in_channels=3
)
# UNet++ as the decoder
model = smp.UnetPlusPlus(
   encoder_name='scanet-101e',
   classes=1,
   encoder_weights=None,
   in_channels=3
)
```

### Models and Params
| **Encoder Name** | **Params(M)** |
|------------------|---------------|
| scanet-14d       | 34            |
| scanet-26d       | 40.5          |
| scanet-50d       | 50.9          |
| scanet-101e      | 73.2          |
---
