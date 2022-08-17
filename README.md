# Additive-Manufacturing-Contrastive-Learners
This repo hosts the codes that were used in journal work "In Situ Quality Monitoring in Direct Energy Deposition Process using Co-axial Process Zone Imaging and Deep Contrastive Learning".

# Journal link
https://doi.org/10.1016/j.jmapro.2022.07.033

![DED Process](https://user-images.githubusercontent.com/39007209/185093514-34cce1b6-674a-4d39-b451-abe5450e9cce.gif)

# Overview

Contrastive Learners help learn mappings from input space to a compact Euclidean space where distances correspond to similarity measures. In the article recently published in "Journal of Manufacturing Processes [impact factor 5.6]" SME, we propose a strategy both in a supervised and semi-supervised manner to monitor the quality of the part built across the possible process spaces that could be simulated on Ti6Al4V grade 1 in a commercial L-DED machine from BeAM Machines. The optical emissions from the process zone, which are imaged co-axially, were distinguished using two deep learning-based contrastive learning frameworks and a methodology to track workpiece manufacturing quality in real-time is demonstrated. Considering the complicated melt pool morphology across process space in the L-DED process, similarity scores from the contrastive learners could also be used in other downstream tasks such as process control apart from process monitoring.

# Contrastive Learners

Contrastive learning is a part of the ML paradigm that enables neural networks to learn without labelling information based on similarities and dissimilarities in data from predefined dataset categories. The core methodology of contrastive learning is that instead of training a network on an image and corresponding ground truth, pairs of images are passed into the network. The network's convolution layers generate a lower-dimensional representation of the images that can be compared using a loss function. The network weights are updated to reduce the distance metric if the images are alike and increased if they are distinct. The trained contrastive model gives a refined lower-dimensional representation which can be further used for classification, segmentation and verification. Two losses are commonly used in the contrastive learning paradigm: contrastive and triplet loss. However, the application of the losses depends primarily on the way the network is trained. The idea behind contrastive loss is the assignment of a Boolean value to a pair of images, i.e., 1 in case they belong to the same category (x, x^+),  0 in case if they are from different categories (x, x^-). During the training, the lower-dimensional representations of images are computed (f(x),f(x^+ ))or (f(x),f(x^- )) and are mutually compared using the aforementioned contrastive loss function, as shown in equation below

![Firstequation](https://latex.codecogs.com/gif.latex?L%3D%281-Y%291/2%20%28D_%7B-%7D%7B%7D%28f%28x%29%2Cf%28x%5E-%29%29%5E2%20&plus;%20%28Y%29%201/2%20%7Bmax%280%2Cm-%28D_%7B&plus;%7D%20%28f%28x%29%2Cf%28x%5E&plus;%29%29%7D%5E2)

where Y is the Boolean label, D_+  and D_- are the distance metrics, and m is the constant margin. The contrastive loss penalizes the high distance metric in samples with Boolean value 1. It also penalizes if the distance metric is lower in samples with Boolean value 0 based on a margin. In other words, the losses are to be minimized if the images are similar and maximized if they are not. The CNN network training with contrastive loss involves taking two instances of the same model with the same architecture and weights, as shown in Figure below. For each iteration, the model is passed with pairs of images with Boolean values 1 or 0. The loss is calculated by comparing the output layer, and the network weights are adjusted accordingly to reduce the loss.

![Fig 2](https://user-images.githubusercontent.com/39007209/185093766-932a1559-0da6-485d-8b38-34266a7e06ad.jpg)

In case of triplet loss, the image triplets are passed into CNN, namely anchor (x), positive (x^+) and negative (x^-). The anchor image serves as a reference, while positive and negative images are correspondingly taken from the same and different categories. The triplet loss minimize the distance between the low-dimensional representation in the anchor f(x) and positive f(x^+), at the same time, maximizing the distance between the anchor f(x) and negative f(x^-). The triplet loss is defined as shown in equation below, 

![Second equation](https://latex.codecogs.com/gif.latex?L%3D%20max%20%28D_&plus;%20%28f%28x%29%2Cf%28x%5E&plus;%29%29%20-D_-%20%28f%28x%29%2Cf%28x%5E-%29%29%20&plus;%20m%2C0%29)

where D_+  is the distance between the positive image and the anchor, D_- is the distance between the negative image and the anchor, and m is a constant margin to differentiate the positive and negative regions. For getting good predictions, the distances D_+ (f(x),f(x^+)) and D_- (f(x),f(x^-)) has to be lower and higher, respectively. The CNN training with triplet loss involves three instances of the same model that share the same architecture and weights, as shown in Figure below. Each model instance is fed with the anchor, positive and negative images. The triplet loss is calculated at each iteration by comparing the lower-dimensional representation in the output layers, and the network weights are adjusted accordingly to reduce the loss.
 

![Fig 3](https://user-images.githubusercontent.com/39007209/185093778-65019378-d13a-4f41-b4ff-9da998fbd15f.jpg)

# Code
```bash
git clone https://github.com/vigneashpandiyan/Additive-Manufacturing-Contrastive-Learners
cd Additive-Manufacturing-Contrastive-Learners
python Main_Siamese.py
python Main_Triplet.py
```

# Citation
```
@article{pandiyan2022situ,
  title={In situ quality monitoring in direct energy deposition process using co-axial process zone imaging and deep contrastive learning},
  author={Pandiyan, Vigneashwara and Cui, Di and Le-Quang, Tri and Deshpande, Pushkar and Wasmer, Kilian and Shevchik, Sergey},
  journal={Journal of Manufacturing Processes},
  volume={81},
  pages={1064--1075},
  year={2022},
  publisher={Elsevier}
}
```
