<h2>ChangeLog (Updated: 2024/06/09)</h2>
<b>2023/10/07: Updated</b><br>
<li>Added BaseImageMaskDataset.py to src for MultipleMyeloma dataset.</li>
<li>Added datasetclass property to model section in train_eval_infer.config file.</li>
<li>Modified TensorflowUNetTrainer.py to use datasetclass defined in the config file.</li>
<li>Fixed bug infer_files method in TensorflowUNet class.</li>

<br><b>2023/11/01: Updated</b><br>
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
<li>Modified TensorflowUNet.py to support bitwise_blending in infer_tiles method.</li>
<br>

<b>2024/03/07: Updated</b><br>
<li>Updated 1.train_by_augmentor.bat and train_eval_infer_augmentor.config.</li>
<br>
<b>2024/03/08: Updated</b><br>
<li>Updated <a href="./src/TensorflowTransUNet.py">TensorflowTransUnet.py to 
use <a href="https://github.com/yingkaisha/keras-unet-collection/tree/main/keras_unet_collection">keras-unet-collection</a>.</li>

<br>
<b>2024/03/10: Updated</b><br>
<li>Fixed a bug in infer_tiles method of TensorflowUNet.py.</li>

<br>
<b>2024/03/20: Updated</b><br>
<li>Added TensorflowSharpUNet.py.</li>
<li>Added TensorflowU2Net.py.</li>

<br>
<b>2024/03/23: Updated</b><br>
<li>Modified 'create' method of TensorflowSharpUNet class to use for loops to create the encoders and decoders in 
<a href="./src/TensorflowSharpUNet.py">TensorflowSharpUNet.py.</li>

<br>
<b>2024/03/25: Updated</b><br>
<li>Refactored the constructors of subclasses of TensorFlowUNet classes to achieve a more simplified codebase.</li>
<li>Added check_models.bat.</li>
<li>Added <a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a>.</li>

<br>
<b>2024/03/29</b><br>
<li>Added <a href="./src/LineGraphPlotter.py">LineGraphPlotter.py</a> to plot line_graphs fo train_eval.csv and train_losses.</li>
<li>Added 'plot_line_graphs' method to <a href="./src/TensorflowUNet.py">TensorflowUNet</a> class 
to plot line_graphs for <i>train_eval.csv</i> and <i>train_losses.csv</i> generated through the training-process.</li>

<br>
<b>2024/03/31</b><br>
<li>Added <a href="./src/TensorflowEfficientNetB7UNet.py">TensorflowEfficientNetB7UNet.py</a>.</li>
<li>Updated create method of <a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a>.</li>

<br>
<b>2024/04/02</b><br>
<li>Added <a href="./src/LineGraph.py">LineGraph.py</a> to draw two line-graphs (train_metrics and train_losses).</li>
<li>Modified <a href="./src/EpochChangeCallback.py">EpochChangeCallback</a> class to draw the line-graphs in 
the 'on_epoch_end' method by using LineGraph class.</li>
<li>Retrained <a href="./projects/TensorflowSlightlyFlexibleUNet/MultipleMyeloma">TensorflowSlightlyFlexibleUNet/MultipleMyeloma</a> Model
using a modified train_eval_infer.config file</li>
<br>
<b>2024/04/13</b><br>
<li>Moved 'train', 'evaluate', 'infer' and some other methods in the original TensorflowUNet.py to a new 
  <a href="./src/TensorflowModel.py">TensorflowModel.py</a> </li>
<li>Modified the new 'train' method in <a href="./src/TensorflowModel.py">TensorflowModel.py</a> to be able to 
use valid dataset specified in [eval] section if available.</li>

<li>Modified <a href="./src/TensorflowUNetTrainer.py">TensorflowUNetTrainer.py</a> to call the new 'train' method 
in <a href="./src/TensorflowModel.py">TensorflowModel.py</a> wihtout a parameter.</li>

<li>Moved the section names used in config files to <a href="./src/ConfigParser.py">ConfigParser.py</a> </li>
<li>Modified the section name 'transoformer' to be 'deformation', and added a new section [distortion] to a config file.</li>

<li>Updated <a href="./src/ImageMaskAugmentor.py">ImageMaskAugmentor.py</a> to support distortion-augmentation.</li>
<li> Modified <a href="./src/TensorflowUNetEvaluator.py">TensorflowUNetEvaluator.py</a> to 
be able to specify a section name wich includes the parameters for image/mask datapaths.</li>
<br>
<b>2024/04/22</b><br>

<li>Moved train method in TensorflowModel to <a href="./src/TensorflowUNetTrainer.py">TensorflowUNetTrainer.py</a>.</li>
<li>Moved evaluate method in TensorflowModel to <a href="./src/TensorflowUNetEvaluator.py">TensorflowUNetEvaluator.py</a>.</li>
<li>Moved infer method in TensorflowModel to <a href="./src/TensorflowUNetInferencer.py">TensorflowUNetInferencer.py</a>.</li>
<li>Moved infer_tiles method in TensorflowModel to <a href="./src/TensorflowUNetTiledInferencer.py">TensorflowUNetTiledInferencer.py</a>.</li>
<li>Added <a href="./src/TensorflowModelLoader.py">TensorflowModelLoader.py</a>.</li>
<li>Added <a href="./src/RGB2GrayscaleImageMaskDataset.py">RGB2GrayscaleImageMaskDataset.py</a>.</li>

<br>
<b>2024/04/24</b><br>
<li>Fixed a bug in evaluate method of <a href="./src/TensorflowUNetEvaluator.py">TensorflowUNetEvaluator.py</a>.<br>
</li>
<b>2024/05/04</b><br>

<li>Fixed a bug in infer method to colorize a mask in <a href="./src/TensorflowUNetInferencer.py">TensorflowUNetInferencer.py</a>.</li>
<li>Added <a href="./src/MaskColorizedWriter.py">MaskColorizedWriter.py</a>.</li> to src folder.</li>

<br>
<b>2024/05/08</b><br>
<li>Fixed some bugs to save the augmented masks to files in <a href="./src/ImageMaskAugmentor.py">ImageMaskAugmentor.py</a>.</li>
<li>Removed  TensorflowUNetGeneratorTrainer.py</li>
<li>Fixed a bug to merge an image and a mask in infer method of <a href="./src/TensorflowUNetInferencer.py">TensorflowUNetInferencer.py</a>.</li>

<br>
<b>2024/06/08</b><br>
<li>Added <a href="./src/EpochChangeInferencer.py">EpochChangeInferencer.py</a> callback to infer the segmentation regions 
for an image on epoch_changed.</li>
<li>Added <a href="./src/EpochChangeTiledInferencer.py">EpochChangeTiledInferencer.py</a> callback to 
tiled-infer the segmentation regions for an image on epoch_changed.</li>
<li>Modifed <a href="./src/TensorflowUNetTrainer.py">TensorFlowUNetTrainer</a> to be able to add
EpochChangeInferencer and EpochChangeTiledInferencer callbacks. </li>

<li>Modifed <a href="./src/TensorflowUNetInferencer.py">TensorflowUNetInferencer</a> to use
<a href="./src/Inferencer.py">Inferencer.py</a>. </li>

<li>Modifed <a href="./src/TensorflowUNetTiledInferencer.py">TensorflowUNetTiledInferencer</a> to use
<a href="./src/TiledInferencer.py">TiledInferencer.py</a>. </li>

<br>
<b>2024/06/09</b><br>
<li>Modifed <a href="./src/TensorflowUNetTrainer.py">TensorflowUNetTrainer.py</a> to add EarlyStopping. </li>

