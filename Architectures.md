

### Architectures
![classification](https://cloud.githubusercontent.com/assets/5226447/8451949/327b9566-2022-11e5-8b34-53b4a64c13ad.PNG)
(from Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, ImageNet Classification with Deep Convolutional Neural Networks, NIPS, 2012.)
* RESNETS [[Paper](http://arxiv.org/pdf/1512.03385v1.pdf)][[Slides](http://kaiminghe.com/icml16tutorial/icml2016_tutorial_deep_residual_networks_kaiminghe.pdf)][[Author github](https://github.com/KaimingHe/deep-residual-networks)][[Keras](https://github.com/fchollet/deep-learning-models)][[TF-Slim](https://github.com/tensorflow/models/tree/master/slim#Pretrained)][[Torch](https://github.com/facebook/fb.resnet.torch)][[PyTorch](https://github.com/pytorch/vision)][[Caffe](https://github.com/BVLC/caffe/wiki/Model-Zoo)]
  * Residual networks add identity skip connnection to layers which makes the layers learn residual transformations. This helps in efficient gradient flow and is currently used in all major network implemetations.
* Inception [[paper1](http://arxiv.org/abs/1409.4842v1)][[paper2](http://arxiv.org/abs/1502.03167)][[paper3](http://arxiv.org/abs/1512.00567)][[paper4](http://arxiv.org/abs/1602.07261)][[paper with ResNet combined](http://arxiv.org/abs/1602.07261)][[Tensorflow](https://github.com/tensorflow/models/tree/master/inception)][[TF-slim](https://github.com/tensorflow/models/edit/master/slim/README.md)][[Keras v3](https://github.com/fchollet/deep-learning-models)][[Torch](https://github.com/Moodstocks/inception-v3.torch)][PyTorch implementation]
  * Googles imagenet models using inception architecture that have networks in networks (hence the name). The latest implementation combines inception along with resnets.
* VGG-Net [[Web]](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) [[Paper]](http://arxiv.org/pdf/1409.1556)
* AlexNet [[Paper]](http://papers.nips.cc/book/advances-in-neural-information-processing-systems-25-2012)
