# Efficient Det
This model is the TensorFlow-lite version of the original [Google AutoML](https://github.com/google/automl) model. Currenlty, only the EfficientDet-D0 is available. Soon, I'll add the other variants of EfficientDet also.



## Installation

Clone the repository and move to the `efficientdet` directory in your terminal. It is recommended to create a separate environment, then install the requirements by following command

```shell
pip install -r requirements.txt
```



## Inference

To make the inference of model, first download the model from [EfficientDet Models](https://drive.google.com/drive/folders/14qoeBOQSB6rK_gOR2fj1K1M7NWVQ2gCY?usp=sharing), and place it under the models directory or wherever you think better. Then, run the following command

```shell
python main.py --model-path MODEL_PATH
```

Here, `MODEL_PATH` is the path to downloaded model you on your machine. 

