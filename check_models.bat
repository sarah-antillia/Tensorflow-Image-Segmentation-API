rem 2024/03/25
rem Check all models by calling constructors of each TensorflowUNet class.  
python ./src/TensorflowAttentionUNet.py ./projects/TensorflowAttentionUNet/MultipleMyeloma/train_eval_infer.config
python ./src/TensorflowDeepLabV3Plus.py ./projects/TensorflowDeepLabV3Plus/MultipleMyeloma/train_eval_infer.config

python ./src/TensorflowEfficientUNet.py ./projects/TensorflowEfficientUNet/MultipleMyeloma/train_eval_infer.config
python ./src/TensorflowMultiResUNet.py  ./projects/TensorflowMultiResUNet/MultipleMyeloma/train_eval_infer.config
python ./src/TensorflowSharpUNet.py     ./projects/TensorflowSharpUNet/MultipleMyeloma/train_eval_infer.config
python ./src/TensorflowSwinUNet.py      ./projects/TensorflowSwinUNet/MultipleMyeloma/train_eval_infer.config
python ./src/TensorflowTransUNet.py     ./projects/TensorflowTransUNet/MultipleMyeloma/train_eval_infer.config
python ./src/TensorflowU2Net.py         ./projects/TensorflowU2Net/MultipleMyeloma/train_eval_infer.config
python ./src/TensorflowUNet.py          ./projects/TensorflowSlightlyFlexibleUNet/MultipleMyeloma/train_eval_infer.config
python ./src/TensorflowUNet3Plus.py     ./projects/TensorflowUNet3Plus/MultipleMyeloma/train_eval_infer.config
