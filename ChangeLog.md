<h2>ChangeLog (Updated: 2024/03/20)</h2>
<b>2023/10/07: Updated</b><br>
<li>Added BaseImageMaskDataset.py to src for MultipleMyeloma dataset.</li>
<li>Added datasetclass property to model section in train_eval_infer.config file.</li>
<li>Modified TensorflowUNetTrainer.py to use datasetclass defined in the config file.</li>
<li>Fixed bug infer_files method in TensorflowUNet class.</li>
<br>
<b>2023/11/01: Updated</b><br>
<li>Updated <a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> to set some random seeds and to force deterministic behavior for Tensorflow.</li>
<li>Added an experimental <a href="./src/SeedResetCallback.py">SeedResetCallback.py</a> to src to reset some random seeds on_epoch_begin method of that callback.</li>
<li>Updated <a href="./src/ImageMaskDataset.py">ImageMaskDataset.py</a> to read the interpolation parameter for cv2.resize from a train_eval_infer.config file.</li>
<li>Updated create method of TensorflowUNet class to set random-seed for Dropout Layer.</li>
<li>Updated train method of TensorflowUNet class to split a master dataset into train and validation sets before calling model.fit</li>
<li>Updated callbacks in train method to be able to add SeedResetCallback.</li>
<li>Updated <a href="./projects/TensorflowSlightlyFlexibleUNet/MultipleMyeloma">./projects/TensorflowSlightlyFlexibleUNet/MultipleMyeloma</a></li>
<li>Added <a href="./projects/TensorflowSlightlyFlexibleUNet/MultipleMyeloma/train_eval_infer_image_mask_dataset_512x512.config">
a sample train_eval_infer.config</a> to ./projects/TensorflowSlightlyFlexibleUNet/MultipleMyeloma</li>
<br>
<b>2023/11/07: Updated</b><br>
<li>Moved Development-Environment to WSL2/Ubuntu-22.04 and Tensorflow 2.14.0.</li>
<li>Updated <a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> to be able to choose AdamW optimizer(Tensorflow 2.14.0) through a train_eval_infer.config.</li>
<li>Updated <a href="./projects/TensorflowSlightlyFlexibleUNet/MultipleMyeloma">./projects/TensorflowSlightlyFlexibleUNet/MultipleMyeloma</a></li>
<br>
Please note that you have to install the proper version of libraries of cuDNN and CUDA listed in 
<a href="https://www.tensorflow.org/install/source#gpu">Tensorflow GPU</a> to your WSL2 in order to train a model of Tensorflow 2.14.0 on your GPU.<br>
<br>

<b>2023/11/10: Updated</b><br>
<li>Updated <a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> to be able to save and load a trained model as a saved_model not only a weight_file (.h5).</li>
<li>Updated <a href="./projects/TensorflowSlightlyFlexibleUNet/MultipleMyeloma">./projects/TensorflowSlightlyFlexibleUNet/MultipleMyeloma</a> to use a saved_model.</li>
<br>

<b>2023/11/13: Updated</b><br>
<li>Moved the dataset folder to a new repository <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Dataset">Tensorflow-Image-Segmentation-Dataset</a>.</li>
<li>Updated TensorflowTrainer, Evaluator, Inferencer and TiledInferencer to use the <b>model</b> property of [model] section in a config file.</li>
<li>Modified all bat files to use TensorflowTrainer, Evaluator, Inferencer, TiledInferencer and ModelInspector.</li>
<br>

<b>2023/11/17: Updated</b><br>
<li>Fixed a bug of TensorflowUnet.load_model method.</li>
<br>

<b>2023/11/21: Updated</b><br>
<li>Updated projects/Tensorflow* model folders to support various datasets.</li>
<li>Added ReduceLROnPlateau callback to callbacks parameter of model.fit.</li>
<br>

<b>2023/12/04: Updated</b><br>
<li>Fixed a bug in TensorflowUNetGeneratorTrainer.py.</li>
<li>Added TensorflowTransUNet to ./projects.</li>
<br>

<b>2024/02/25: Updated</b><br>
<li>Updated infer method in TensorflowUNet class.</li>
<li>Updated GrayScaleImageWriter class to colorize inferred segmention regions.</li>
<li>Modified shear method to check self.hflip and self.vflip flags in ImageMaskAugmentor class.</li>
<br>

<b>2024/02/26: Updated</b><br>
<li>Added DatasetStatistics.py to ./src.</li>
<br>

<b>2024/03/02: Updated</b><br>
<li>Fixed a bug in infer_tiles method in TensorflowUNet.py.</li>
<li>Added mish activation function to TensorflowUNet.py</li>
<li>Added ReduceLROnPlateau(learning_rate_reducer) callback to TensorflowUNet.py</li>
<li>Updated save and save_reisze methods in GrayScaleImageWriter.py to call mask_to_image method.</li>
<br>

<b>2024/03/04: Updated</b><br>
 <li>Removed mini_test and 4k_mini_test under projects folder.</li>
<li>Updated TensorflowSwinUNet Tiled-image-segmentation.</li>
<li>Updated TensorflowMultiResUNet Tiled-image-segmentation.</li>
<li>Updated TensorflowAttentionUNet Tiled-image-segmentation.</li>
<li>Updated TensorflowUNet3Plus Tiled-image-segmentation.</li>
<br>

<b>2024/03/05: Updated</b><br>
<li>2024/03/05: Modified TensorflowUNet.py to support bitwise_blending in infer_tiles method.</li>
<br>

<b>2024/03/07: Updated</b><br>
<li>2024/03/07: Updated 1.train_by_augmentor.bat and train_eval_infer_augmentor.config.</li>

<b>2024/03/08: Updated</b><br>
<li>2024/03/08: Updated <a href="./src/TensorflowTransUNet.py">TensorflowTransUnet.py to 
use <a href="https://github.com/yingkaisha/keras-unet-collection/tree/main/keras_unet_collection">keras-unet-collection</a>.</li>

<b>2024/03/10: Updated</b><br>
<li>2024/03/10: Fixed a bug in infer_tiles method of TensorflowUNet.py.</li>

<b>2024/03/20: Updated</b><br>
<li>2024/03/20: Added TensorflowSharpUNet.py.</li>
<li>2024/03/20: Added TensorflowU2Net.py.</li>

<b>2024/03/23: Updated</b><br>
<li>2024/03/23 Modified 'create' method to use for loops to create the encoders and decoders.</li>


