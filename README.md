# Tensorflow-Image-Segmentation-API (Updated: 2023/09/05)
<h2>
1 Image Segmentaion API 
</h2>
We have the following Tensorflow Image Segmentation models<br>

<li><a href="">TensorflowSlightlyFlexibleUNet</a></li>
<li><a href="">TensorflowSwinUNet</a></li>
<li><a href="">TensorflowMultiResUNet</a></li>
<li><a href="">TensorflowAttentionUNet</a></li>
<li><a href="">TensorflowUNet3Plus</a></li>

<h2>
2 Dataset
</h2>
We have the following dataset.<br>
<li><a href="./dataset/ALL">ALL</a></li>
<li><a href="./dataset/BrainTumor">BrainTumor</a></li>
<li><a href="./dataset/Cervial-Cancer">Cervial-Cancer</a></li>
<li><a href="./dataset/GastrointestinalPolyp">GastrointestinalPolyp</a></li>
<li><a href="./Mammogram">Mammogram</a></li>
<li><a href="./MultipleMyeloma">MultipleMyeloma</a></li>
<li><a href="./Nerve">Nerve</a></li>
<li><a href="./Retinal-Vessel">Retinal-Vessel</a></li>

Please note that each dataset file has been archived in 7z format.

<h2>
3 Train
</h2>
For example, please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/MultipleMyeloma</b> folder,<br>
and run the following bat file to train TensorflowUNet model for MultipleMyeloma.<br>
<pre>
./1.train.bat
</pre>



<h2>
4 Evaluate
</h2>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/MultipleMyeloma</b> folder,<br>
and run the following bat file to evalute TensorflowUNet model for MultipleMyeloma.<br>
<pre>
./2.evaluate.bat
</pre>



<h2>
5 Infer 
</h2>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/MultipleMyeloma</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for MultipleMyeloma.<br>
<pre>
./3.infer.bat
</pre>








