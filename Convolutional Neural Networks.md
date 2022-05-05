[TOC]

# 1. Introduction

Convolutional neural networks sounds like a weird combination of biology and math with a little CS sprinkled in, but these networks have been some of the most influential innovations in the field of computer vision. 2012 was the first year that neural nets grew to prominence as Alex Krizhevsky used them to win that year’s ImageNet competition (basically, the annual Olympics of computer vision), dropping the classification error record from 26% to 15%, an astounding improvement at the time.Ever since then, a host of companies have been using deep learning at the core of their services. Facebook uses neural nets for their automatic tagging algorithms, Google for their photo search, Amazon for their product recommendations, Pinterest for their home feed personalization, and Instagram for their search infrastructure

![Companies](E:\CS\Computer Vision\Notes\convnet\images\Companies.png)

However, the classic, and arguably most popular, use case of these networks is for image processing. Within image processing, here we look at how to use these CNNs for **image classification**.

**Image classification** is the task of taking an input image and outputting a class (a cat, dog, etc) or a probability of classes that best describes the image. For humans, this task of recognition is one of the first skills we learn from the moment we are born and is one that comes naturally and effortlessly as adults. Without even thinking twice, we’re able to quickly and seamlessly identify the environment we are in as well as the objects that surround us. When we see an image or just when we look at the world around us, most of the time we are able to immediately characterize the scene and give each object a label, all without even consciously noticing. These skills of being able to quickly recognize patterns, generalize from prior knowledge, and adapt to different image environments are ones that we do not share with our fellow machines.

# 2. Why ConvNets?

Convolutional Neural Networks are very similar to ordinary Neural Networks: they are made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole network still expresses a single differentiable score function: from the raw image pixels on one end to class scores at the other. And they still have a loss function (e.g. SVM/Softmax) on the last (fully-connected) layer and all the tips/tricks we developed for learning regular Neural Networks still apply.

So what changes? ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network. Let’s first understand what is an image.

In computers an image is represented as a grid of pixel values — i.e. a grid of positive whole numbers.

![img_pix](E:\CS\Computer Vision\Notes\convnet\images\img_pix.PNG)

​														What We See                                                                         What Computer See

Essentially, every image can be represented as a matrix of pixel values.

<img src="E:\CS\Computer Vision\Notes\convnet\images\8.gif" alt="8" style="zoom:50%;" />

​																										Every image is a matrix of pixel values

In practice, color images are represented using three grids (channel) of numbers stacked on top of each other: one grid for red, one grid for green, and one grid for blue- you can imagine those as three 2d-matrices stacked over each other (one for each color). 

<img src="E:\CS\Computer Vision\Notes\convnet\images\rgb_image.png" alt="rgb_image" style="zoom: 80%;" />

A grayscale image, on the other hand, has just one channel- a single 2d matrix representing an image. Let's say we have a color image in JPG form and its size is 480 x 480. The representative array will be 480 x 480 x 3. Each of these numbers is given a value from 0 (black) to 255 (white) which describes the pixel intensity at that point.

**Regular Neural Networks**

Regular neural networks consisted of layers which compute a linear function followed by a nonlinearity. 
$$
\textbf{a} = f(\textbf{Wx})
$$
They're called fully connected layers, because every one of the input units is connected to every one of the output units. While fully connected layers are useful, they're not always what we want. Here are some reasons:

- They don’t scale well to full images and require a lot of connections: In CIFAR-10, images are only of size 32x32x3, so a single fully-connected neuron in a first hidden layer of a regular Neural Network would have 32 x 32 x 3 = 3072 weights. This amount still seems manageable, but clearly this fully-connected structure does not scale to larger images. For example, an image of more respectable size, e.g. 200x200x3, would lead to neurons that have 200 x 200 x 3 = 120,000 weights. Moreover, we would almost certainly want to have several such neurons, so the parameters would add up quickly! If the input layer has M units and the output layer has N units, then we need M x N connections. This can be quite a lot; for instance, suppose the input layer is an image consisting of M = 256 x 256 = 65563 grayscale pixels, and the output layer consists of N = 1000 units (modest by today's standards). A fully connected layer would require 65 million connections. This causes two problems:
  - Computing the hidden activations requires one add-multiply operation per connection in the network, so large numbers of connections can be expensive.
  - Each connection has a separate weight parameter, so we would need a huge number of training examples in order to avoid overfitting.
- If we're trying to classify an image, there's certain structure we'd like to make use of. For instance:
  - We would like to share **structure** between different parts of the network e.g, features (such as edges) which are useful at one image location are likely to be useful at other locations as well. 
  - Another property we'd like to make use of is **invariance**: if the image or waveform is transformed slightly (e.g. by shifting it a few pixels), the classification shouldn't change. Both of these properties should be encoded into the network's architecture if possible.

<img src="E:\CS\Computer Vision\Notes\convnet\images\neural_net2.jpeg" alt="neural_net2" style="zoom:50%;"/>

​																			            Figure 1: A regular 3-layer fully connected Neural Network



The Convolutional Neural Network deals with all these issues.  Like the name suggests, the architecture is inspired by a mathematical operator called convolution (which we'll explain shortly).



# 3. History of ConvNets

**Simple and Complex Cells**

In 1959, David Hubel and Torsten Wiesel described “simple cells” and “complex cells” in the human visual cortex. They proposed that both kinds of cells are used in pattern recognition. A “simple cell” responds to edges and bars of particular orientations, such as this image:

<img src="E:\CS\Computer Vision\Notes\convnet\images\gabor_filter.png" alt="gabor_filter" style="zoom: 80%;" />

​																									Figure 3: 

A “complex cell” also responds to edges and bars of particular orientations, but it is different from a simple cell in that these edges and bars can be shifted around the scene and the cell will still respond. For instance, a simple cell may respond only to a horizontal bar at the bottom of an image, while a complex cell might respond to horizontal bars at the bottom, middle, or top of an image. This property of complex cells is termed “spatial invariance.”

Figure 1 in [this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1890437/) diagrams the difference between simple and complex cells.

![tjp0577-0463-f1](E:\CS\Computer Vision\Notes\convnet\images\tjp0577-0463-f1.jpg)

​																				Figure 4: 

Hubel and Wiesel proposed in 1962 that complex cells achieve spatial invariance by “summing” the output of several simple cells that all prefer the same orientation (e.g. horizontal bars) but different receptive fields (e.g. bottom, middle, or top of an image). By collecting information from a bunch of simple cell minions, the complex cells can respond to horizontal bars that occur anywhere.

<video src="E:\CS\Computer Vision\Notes\convnet\images\Hubel and Wiesel Cat Experiment.mp4"></video>

​																										Hubel and Wiesel Cat Experiment

This concept – that simple detectors can be “summed” to create more complex detectors – is found throughout the human visual system, and is also the fundamental basis of convolution neural network models.



**The Neocognitron**

In the 1980s, Dr. Kunihiko Fukushima was inspired by Hubel and Wiesel’s work on simple and complex cells, and proposed the “[neocognitron](https://en.wikipedia.org/wiki/Neocognitron)” model (original paper: [“Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position”](https://www.rctn.org/bruno/public/papers/Fukushima1980.pdf)). The neocognitron model includes components termed “S-cells” and “C-cells.” These are not biological cells, but rather mathematical operations. The “S-cells” sit in the first layer of the model, and are connected to “C-cells” which sit in the second layer of the model. The overall idea is to capture the “simple-to-complex” concept and turn it into a computational model for visual pattern recognition.

![neocognition](E:\CS\Computer Vision\Notes\convnet\images\neocognition.PNG)

**ConvNets for Handwriting Recognition**

The first work on modern convolutional neural networks (CNNs) occurred in the 1990s, inspired by the neocognitron. Yann LeCun et al., in their paper [“Gradient-Based Learning Applied to Document Recognition”](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (now cited 17,588 times) demonstrated that a CNN model which aggregates simpler features into progressively more complicated features can be successfully used for handwritten character recognition.

Specifically, LeCun et al. trained a CNN using the MNIST database of handwritten digits (MNIST pronounced “EM-nisst”). MNIST is a now-famous data set that includes images of handwritten digits paired with their true label of 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9. A CNN model is trained on MNIST by giving it an example image, asking it to predict what digit is shown in the image, and then updating the model’s settings based on whether it predicted the digit identity correctly or not. State-of-the-art CNN models can today achieve near-perfect accuracy on MNIST digit classification.

<video src="E:\CS\Computer Vision\Notes\convnet\images\cnn_Yann_LeCun.mp4"></video>

​																		                    



**ConvNets for Imagenet**

Throughout the 1990s and early 2000s, researchers carried out further work on the CNN model. Around 2012 CNNs enjoyed a huge surge in popularity (which continues today) after a CNN called AlexNet achieved state-of-the-art performance labeling pictures in the ImageNet challenge. Alex Krizhevsky et al. published the paper “[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)” describing the winning AlexNet model; this paper has since been cited 38,007 times.

**ConvNets and Human Vision**

The popular press often talks about how neural network models are “directly inspired by the human brain.” In some sense, this is true, as both CNNs and the human visual system follow a “simple-to-complex” hierarchical structure. However, the actual implementation is totally different; brains are built using cells, and neural networks are built using mathematical operations.



# 4. Overview of ConvNets

A simple ConvNet is a sequence of layers, and every layer of a ConvNet transforms one volume of activations to another through a differentiable function. We use three main types of layers to build ConvNet architectures: 

- **Convolutional Layer**, 
- **Pooling Layer,**
- **Fully-Connected Layer** (exactly as seen in regular Neural Networks). 

We will stack these layers to form a full ConvNet architecture.

![Cover](E:\CS\Computer Vision\Notes\convnet\images\Cover.png)

Let’s break down a CNN into its basic building blocks.

1. A tensor can be thought of as an n-dimensional matrix. In the CNN above, tensors will be 3-dimensional with the exception of the output layer.
2. A neuron can be thought of as a function that takes in multiple inputs and yields a single output.
3. A layer is simply a collection of neurons with the same operation, including the same hyperparameters.
4. Kernel weights and biases, while unique to each neuron, are tuned during the training phase, and allow the classifier to adapt to the problem and dataset provided. 
5. A CNN conveys a differentiable score function, which is represented as class scores in the visualization on the output layer.

*We now describe the individual layers and the details of their hyperparameters and their connectivities.*

**Input and Output Layer**

The input layer (leftmost layer) represents the input image into the CNN. The input to a CNN for a computer vision application is an image or a video. (CNNs can also be used on text).  These pixel values of an image, while meaningless to us when we perform image classification, are the only inputs available to the computer. The idea is that you give the computer this array of numbers and it will output numbers that describe the probability of the image being a certain class (.80 for cat, .15 for dog, .05 for bird, etc). The output of a CNN depends on the task. Here are some example CNN inputs and outputs for a variety of classification tasks:

<img src="E:\CS\Computer Vision\Notes\convnet\images\slide1.png" alt="slide1" style="zoom: 67%;" />



Now that we know the problem as well as the inputs and outputs, let’s think about how to approach this. What we want the computer to do is to be able to differentiate between all the images it’s given and figure out the unique features that make a dog a dog or that make a cat a cat. This is the process that goes on in our minds subconsciously as well. When we look at a picture of a dog, we can classify it as such if the picture has identifiable features such as paws or 4 legs. In a similar way, the computer is able perform image classification by looking for low level features such as edges and curves, and then building up to more abstract concepts through a series of convolutional layers. This is a general overview of what a CNN does. 

A more detailed overview of what CNNs do would be that you take the image, pass it through a series of convolutional, nonlinear, pooling (downsampling), and fully connected layers, and get an output. As we said earlier, the output can be a single class or a probability of classes that best describes the image. 

<img src="E:\CS\Computer Vision\Notes\convnet\images\cnn1.png" alt="cnn1" style="zoom:50%;" />

Now, the hard part is understanding what each of these layers do

# 5. Convolution
Suppose we have two **signals** $x$ and $w$, which you can think of as arrays, with elements denoted as $x[t]$ and so on. As you can guess based on the letters, you can think of $x$ as an input signal (such as an image) and $w$ a set of weights, which we'll refer to as a **filter** or **kernel**. Normally the signals we work with are finite in extent, but it is sometimes convenient to treat them as in infinitely large by treating the values as zero everywhere else; this is known as **zero padding**. Let's start with the one-dimensional case. 

## 5.1 One dimensional convolution

The **convolution** of $x$ and $w$, denoted  $x * w$, is a signal with entries given by
$$
(x*w)[t] = \sum x[t-\tau]w[\tau]
$$
There are two ways to think about this equation. 

1. The first is **translate- and-scale**: the signal $x*w$ is composed of multiple copies of $x$, translated and scaled by various amounts according to the entries of $w$. An example of this is shown in Figure 1.

![Translate-and-scale](E:\CS\Computer Vision\Notes\convnet\images\Translate-and-scale.PNG)

​											Figure 1: Translate-and-scale interpretation of convolution of one-dimensional signals.

2. A second way to think about it is  **flip-and-filter**. Here we generate each of the entries of $x*w$ by  flipping $w$, shifting it, and taking the dot
   product with $x$. An example is shown in Figure 2.

![flip_and_filter](E:\CS\Computer Vision\Notes\convnet\images\flip_and_filter.PNG)

​													Figure 2: Flip-and-filter interpretation of convolution of one-dimensional signals.

Convolution can also be viewed as matrix multiplication:

<img src="E:\CS\Computer Vision\Notes\convnet\images\convolution.PNG" alt="convolution" style="zoom:50%;" />

*Aside: This is how convolution is typically implemented. (More efficient than the fast Fourier transform (FFT) for modern conv nets on GPUs!)*

## 5.2 Two-dimensional Convolution

The two-dimensional case is exactly analogous to the one-dimensional case; we apply the same definition, but with more indices:
$$
(x*w)[s,t] = \sum_{\sigma, \tau} x[s-\sigma, t-\tau]w[\sigma,\tau]
$$
This is shown graphically in Figures 3 and 4.

![Translate-and-scale_2d](E:\CS\Computer Vision\Notes\convnet\images\Translate-and-scale_2d.PNG)

​											Figure : Translate-and-scale interpretation of convolution of two-dimensional signals.

![flip_and_filter_2d](E:\CS\Computer Vision\Notes\convnet\images\flip_and_filter_2d.PNG)

​											Figure : Flip-and-filter interpretation of convolution of two-dimensional signals.

The thing we convolve by is called a **kernel**, or **filter** or **feature detector** and the matrix formed by sliding the filter over the image and computing the dot product is called the **Convolved Feature** or **Activation Map** or the **Feature Map**. It is important to note that filters acts as feature detectors from the original input image. . Now, question arises that what does this convolution kernel do? Despite the simplicity of the operation, convolution can do some pretty interesting things. For instance, 

- blur an image

  ![blur](E:\CS\Computer Vision\Notes\convnet\images\blur.PNG)

- sharpen 

![sharpen](E:\CS\Computer Vision\Notes\convnet\images\sharpen.PNG)

- detect edge

![edge1](E:\CS\Computer Vision\Notes\convnet\images\edge1.PNG)

![edge2](E:\CS\Computer Vision\Notes\convnet\images\edge2.PNG)

Convolution operation is simple, but this is the foundation of ConvNet. 




# 6. Components Use to Build ConvNets

A simple ConvNet is a sequence of layers, and every layer of a ConvNet transforms one volume of activations to another through a differentiable function that may or may not have parameters. We use three main types of layers to build ConvNet architectures: **Convolutional Layer**, **Pooling Layer**, and **Fully-Connected Layer** (exactly as seen in regular Neural Networks). We will stack these layers to form a full ConvNet **architecture**.

## 6.1 Convolutional Layers

The Conv layer is the core building block of a Convolutional Network that does most of the computational heavy lifting.  Confusingly, the way they're standardly defined, convolution layers don't actually compute convolutions, but a closely related operation called **filtering**:
$$
(x*w)[t] = \sum_{\tau}x[t+\tau]w[\tau]
$$
Like the name suggests, filtering is essentially like flip-and-filter, but without the flipping. (I.e., $x*w=x*flip(w)$. The two operations are basically equivalent- the difference is just a matter of how the filter (or kernel) is represented.

In the above example, we computed a single feature map, but just as we normally use more than one hidden unit in fully connected layers, convolution layers normally compute multiple feature maps $z_1, \dots , z_M$. The input layers also consist of multiple feature maps $x_1, \dots , x_D$; these could be different color channels of an RGB image, or feature maps computed by another convolution layer. There is a separate  filter $w_{ij}$ associated with each pair of an input and output feature map. The convolution operations are computed as follows:
$$
z_i = \sum_j x_j * w_{ij}
$$
For example, consider a 5 x 5 image whose pixel values are only 0 and 1 (note that for a grayscale image, pixel values range from 0 to 255, the green matrix below is a special case where pixel values are only 0 and 1):



<img src="E:\CS\Computer Vision\Notes\convnet\images\screen-shot-2016-07-24-at-11-25-13-pm.png" alt="screen-shot-2016-07-24-at-11-25-13-pm" style="zoom:120%;" />

Also, consider another 3 x 3 filter as shown below:

<img src="E:\CS\Computer Vision\Notes\convnet\images\screen-shot-2016-07-24-at-11-25-24-pm.png" alt="screen-shot-2016-07-24-at-11-25-24-pm" style="zoom:120%;" />

Then, the Convolution of the 5 x 5 image and the 3 x 3 matrix can be computed as shown in the animation below:

<img src="E:\CS\Computer Vision\Notes\convnet\images\convolution_schematic.gif" alt="convolution_schematic" style="zoom: 80%;" />

Take a moment to understand how the computation above is being done. We slide the orange matrix over our original image (green) by 1 pixel (also called **stride**) and for every position, we compute element wise multiplication (between the two matrices) and add the multiplication outputs to get the final integer which forms a single element of the output matrix (pink). Note that the kernel “sees” only a part of the input image in each stride.

The size of these kernels and stride are hyper-parameter specified by the designers of the network architecture. **The primary purpose of Convolution in case of a ConvNet is to extract features from the input image**. To better understand, look at the animation in Figure 

![giphy](E:\CS\Computer Vision\Notes\convnet\images\giphy.gif)

​																			                        Figure: The Convolution Operation

A filter (with red outline) slides over the input image (convolution operation) to produce a feature map. The convolution of another filter (with the green outline), over the same image gives a different feature map as shown. It is important to note that the Convolution operation captures the local dependencies in the original image. Also notice how these two different filters generate different feature maps from the same original image. Remember that the image and the two filters above are just numeric matrices as we have discussed above.

In practice, a CNN *learns* the values of these filters on its own during the training process (although we still need to specify parameters such as **number of filters**, **filter size**, **architecture of the network** etc. before the training process). The more number of filters we have, the more image features get extracted and the better our network becomes at recognizing patterns in unseen images.

The size of the Feature Map (Convolved Feature) is controlled by four parameters that we need to decide before the convolution step is performed:

1. **Depth:** Depth corresponds to the *number of filters* we use for the convolution operation. In the network shown in Figure , we are performing convolution of the original boat image using three distinct filters, thus producing three different feature maps as shown. You can think of these three feature maps as stacked 2d matrices, so, the ‘depth’ of the feature map would be three.

<img src="E:\CS\Computer Vision\Notes\convnet\images\screen-shot-2016-08-10-at-3-42-35-am.png" alt="screen-shot-2016-08-10-at-3-42-35-am" style="zoom: 50%;" />

2. **Filter Size:** Filter size, often also referred to as kernel size, refers to the dimensions of the sliding window over the input. Choosing this hyperparameter has a massive impact. For example, small kernel sizes are able to extract a much larger amount of information containing highly local features from the input. As you can see on the visualization above, a smaller kernel size also leads to a smaller reduction in layer dimensions, which allows for a deeper architecture. Conversely, a large kernel size extracts less information, which leads to a faster reduction in layer dimensions, often leading to worse performance. Large kernels are better suited to extract features that are larger. At the end of the day, choosing an appropriate kernel size will be dependent on your task and dataset, but generally, smaller kernel sizes lead to better performance for the image classification task because an architecture designer is able to stack more and more layers together to learn more and more complex features!
3. **Stride:** Stride is the number of pixels by which we slide our filter matrix over the input matrix. When the stride is 1 then we move the filters one pixel at a time. When the stride is 2 (or uncommonly 3 or more, though this is rare in practice), then the filters jump 2 pixels at a time as we slide them around. Having a larger stride will produce smaller output volumes spatially.



<img src="E:\CS\Computer Vision\Notes\convnet\images\stride1.gif" alt="stride1" style="zoom:50%;" />

<img src="E:\CS\Computer Vision\Notes\convnet\images\stride2.gif" alt="stride2" style="zoom:50%;" />



4. **Padding**: Padding is often necessary when the kernel extends beyond the activation map. The main benefits of padding are the following:
   - It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers. An important special case is the "same" convolution, in which the height/width is exactly preserved after one layer.
   - It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels as the edges of an image.

There exist many padding techniques, but the most commonly used approach is **zero-padding** because of its performance, simplicity, and computational efficiency. The technique involves adding zeros symmetrically around the edges of an input.  The nice feature of zero padding is that it will allow us to control the spatial size of the output volumes (most commonly as we’ll see soon we will use it to exactly preserve the spatial size of the input volume so the input and output width and height are the same). Adding zero-padding is also called *wide convolution***,** and not using zero-padding would be a *narrow convolution*.

<img src="E:\CS\Computer Vision\Notes\convnet\images\PAD.png" alt="PAD" style="zoom:67%;" />



There are two types of results to the operation — one in which the convolved feature is reduced in dimensionality as compared to the input, this is done by applying **Valid Padding**. The other in which the dimensionality is either increased or remains the same, this is done by applying **Same Padding** . The following animations would help us to get a better understanding of how stride and padding length work together to achieve results of our need. 



<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="E:/CS/Computer Vision/Notes/convnet/images/no_padding_no_strides.gif"></td>
    <td><img width="150px" src="E:/CS/Computer Vision/Notes/convnet/images/arbitrary_padding_no_strides.gif"></td>
    <td><img width="150px" src="E:/CS/Computer Vision/Notes/convnet/images/same_padding_no_strides.gif"></td>
    <td><img width="150px" src="E:/CS/Computer Vision/Notes/convnet/images/full_padding_no_strides.gif"></td>
  </tr>
  <tr>
    <td>No padding, unit strides (Convolving a 3 × 3 kernel over a 4 × 4 input using unit strides)</td>
    <td>Arbitrary padding, unit strides (Convolving a 4 × 4 kernel over a 5 × 5 input padded with a 2 × 2 border of zeros using unit strides)</td>
    <td>Half padding, unit strides (Convolving a 3 × 3 kernel over a 5 × 5 input using half padding and unit strides)</td>
    <td>Full padding, unit strides (Convolving a 3 × 3 kernel over a 5 × 5 input using full padding and unit strides)</td>
  </tr>
  <tr>
    <td><img width="150px" src="E:/CS/Computer Vision/Notes/convnet/images/no_padding_strides.gif"></td>
    <td><img width="150px" src="E:/CS/Computer Vision/Notes/convnet/images/padding_strides.gif"></td>
    <td><img width="150px" src="E:/CS/Computer Vision/Notes/convnet/images/padding_strides_odd.gif"></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, strides (Convolving a 3 × 3 kernel over a 5 × 5 input using 2 × 2 strides)</td>
    <td>Padding, strides (Convolving a 3 × 3 kernel over a 5 × 5 input padded with a 1 × 1 border of zeros using 2 × 2 strides)</td>
    <td>Padding, strides (Convolving a 3 × 3 kernel over a 6 × 6 input padded with a 1 × 1 border of zeros using 2 × 2 strides)</td>
    <td></td>
  </tr>
</table>


The convolutional layers are the foundation of CNN, as they contain the learned kernels (weights), which extract features that distinguish different images from one another—this is what we want for classification! For example, let’s look at the animation in Figure and notice the links between the previous layers and the convolutional layers. Each link represents a unique kernel, which is used for the convolution operation to produce the current convolutional neuron’s output or **activation map**. The convolutional neuron performs an elementwise dot product with a unique kernel and the output of the previous layer’s corresponding neuron. This will yield as many intermediate results as there are unique kernels. The convolutional neuron is the result of all of the intermediate results summed together with the learned bias.

Notice that there are 10 neurons in this layer, but only 3 neurons in the previous layer. In the Tiny CNN architecture, convolutional layers are fully-connected, meaning each neuron is connected to every other neuron in the previous layer. Focusing on the output of the topmost convolutional neuron from the first convolutional layer, we see that there are 3 unique kernels when we hover over the activation map.

![convlayer_overview_demo](E:\CS\Computer Vision\Notes\convnet\images\convlayer_overview_demo.gif)

Figure: As you hover over the activation map of the topmost node from the first convolutional layer, you can see that 3 kernels were applied to yield this activation map. After clicking this activation map, you can see the convolution operation occurring with each unique kernel.

In the case of images with multiple channels (e.g. RGB), the Kernel has the same depth as that of the input image. Below is a running demo of a CONV layer. Since 3D volumes are hard to visualize, all the volumes and the output volume are visualized with each depth slice stacked in rows. The visualization below shows that each output element is computed by elementwise multiplying the highlighted input with the filter, summing it up, and then offsetting the result by the bias.

![1_ciDgQEjViWLnCbmX-EeSrA](E:\CS\Computer Vision\Notes\convnet\images\1_ciDgQEjViWLnCbmX-EeSrA.gif)

​														Figure: Convolution operation on a MxNx3 image matrix with a 3x3x3 Kernel.

Another good way to understand the Convolution operation is by looking at the animation in Figure below:

![convlayer_detailedview_demo](E:\CS\Computer Vision\Notes\convnet\images\convlayer_detailedview_demo.gif)

​                              Figure: The kernel being applied to yield the topmost intermediate result for the discussed activation map.

With some simple math, we are able to deduce that there are 3 x 10 = 30 unique kernels, each of size 3x3, applied in the first convolutional layer. The connectivity between the convolutional layer and the previous layer is a design decision when building a network architecture, which will affect the number of kernels per convolutional layer. 

**Receptive Field**

The convolution operation is carried out by convolving the filter to only a local region of the input volume. In other words, each neuron of the output volume is connected to only a local region of the input volume. The regions of the input volume which influence their activations called **receptive field** of the neuron (equivalently this is the filter size).  The receptive field’s spatial dimensions (width and height) are local, but the depth dimension always full along the entire depth of the input volume.

For instance, consider the example shown in Figure. Each neuron in the convolutional layer is connected only to a local region in the input volume spatially, but to the full depth (i.e. all color channels). Note, there are multiple neurons (5 in this example) along the depth, all looking at the same region in the input: the lines that connect this column of 5 neurons do not represent the weights (i.e. these 5 neurons do not share the same weights, but they are associated with 5 different filters), they just indicate that these neurons are connected to or looking at the same receptive field or region of the input volume, i.e. they share the same receptive field but not the same weights.

<img src="E:\CS\Computer Vision\Notes\convnet\images\depthcol.jpeg" alt="depthcol" style="zoom: 67%;" />

​                                                                                   Figure: 

- *Example 1*. For example, suppose that the input volume has size [32x32x3], (e.g. an RGB CIFAR-10 image). If the receptive field (or the filter size) is 5x5, then each neuron in the Conv Layer will have weights to a [5x5x3] region in the input volume, for a total of 5 x 5 x 3 = 75 weights (and +1 bias parameter). Notice that the extent of the connectivity along the depth axis must be 3, since this is the depth of the input volume.

- *Example 2*. Suppose an input volume had size [16x16x20]. Then using an example receptive field size of 3x3, every neuron in the Conv Layer would now have a total of 3 x 3 x 20 = 180 connections to the input volume. Notice that, again, the connectivity is local in 2D space (e.g. 3x3), but full along the input depth (20).

To recap, the CONV layer’s parameters consist of a set of learnable filters. Every filter is small spatially (along width and height), but extends through the full depth of the input volume. For example, a typical filter on a first layer of a ConvNet might have size 3x3x3 (i.e. 3 pixels width and height, and 3 because images have depth 3, the color channels). During the forward pass, we slide (more precisely, convolve) each filter across the width and height of the input volume and compute dot products between the entries of the filter and the input at any position. As we slide the filter over the width and height of the input volume we will produce a 2-dimensional activation map that gives the responses of that filter at every spatial position. Intuitively, the network will learn filters that activate when they see some type of visual feature such as an edge of some orientation or a blotch of some color on the first layer, or eventually entire honeycomb or wheel-like patterns on higher layers of the network. Now, we will have an entire set of filters in each CONV layer (e.g. 12 filters), and each of them will produce a separate 2-dimensional activation map. We will stack these activation maps along the depth dimension and produce the output volume.

<video src="E:\CS\Computer Vision\Notes\convnet\images\conv_kiank.mp4"></video>



More generally, the convolution layer:

- Accepts a volume of size $W_1×H_1×D_1$
- Requires four hyperparameters:
  - number of filters, $K$
  - their spatial extent, $F$
  - the stride $S$
  - the amout of zero padding, $P$
- Produces a volume of size $W_2 × H_2 × D_2$ where
  - $W_2 = (W_1 - F + 2P)/S + 1$
  - $H_2 = (H_1 - F + 2P)/S + 1$
  - $D_2 = K$
- With parameter sharing, it introduces $F*F*D$ weights per filter, for a total of $(F.F.D_1).K$ weights and $K$ biases
- In the output volume, the $d$-th depth slice (of size $W_2$  x $H_2$) is the result of performing a valid convolution of the $d$-th filter over the input volume with a stride of $S$, and then offset by $d$-th bias.

**The brain view**

If you’re a fan of the brain/neuron analogies, every entry in the 3D output volume can also be interpreted as an output of a neuron that looks at only a small region in the input, called **receptive field** and shares parameters with all neurons to the left and right spatially (since these numbers all result from applying the same filter).



### 6.1.1 Implementation of Convolution as Matrix Multiplication

The convolution operation essentially performs dot products between the filters and local regions of the input. A common implementation pattern of the CONV layer is to take advantage of this fact and formulate the forward pass of a convolutional layer as one big matrix multiply as follows:

1. The local regions in the input image are stretched out into columns in an operation commonly called **im2col**. For example, if the input is [227x227x3] and it is to be convolved with 11x11x3 filters at stride 4, then we would take [11x11x3] blocks of pixels in the input and stretch each block into a column vector of size 11*11*3 = 363. Iterating this process in the input at stride of 4 gives (227-11)/4+1 = 55 locations along both width and height, leading to an output matrix `X_col` of *im2col* of size [363 x 3025], where every column is a stretched out receptive field and there are 55*55 = 3025 of them in total. Note that since the receptive fields overlap, every number in the input volume may be duplicated in multiple distinct columns.
2. The weights of the CONV layer are similarly stretched out into rows. For example, if there are 96 filters of size [11x11x3] this would give a matrix `W_row` of size [96 x 363].
3. The result of a convolution is now equivalent to performing one large matrix multiply `np.dot(W_row, X_col)`, which evaluates the dot product between every filter and every receptive field location. In our example, the output of this operation would be [96 x 3025], giving the output of the dot product of each filter at each location.
4. The result must finally be reshaped back to its proper output dimension [55x55x96].

This approach has the downside that it can use a lot of memory, since some values in the input volume are replicated multiple times in `X_col`. However, the benefit is that there are many very efficient implementations of Matrix Multiplication that we can take advantage of (for example, in the commonly used [BLAS](http://www.netlib.org/blas/) API). Moreover, the same *im2col* idea can be reused to perform the pooling operation, which we discuss next.

### 6.1.2 Convolution followed by A Nonlinearity

Convolution is a linear operation. Therefore, a sequence of convolutions can only compute a linear function i.e, a sequence of convolution would be no more powerful that one, but a sequence of convolutions alternated with nonlinearities can do fancier things. E.g., consider the following sequence of operations: 1) Convolve the image with a horizontal edge filter, 2) Apply ReLu activation function 3) Blur the result. This sequence of steps, shown in Figure,  gives a map of horizontalness in various parts of an image; the same can be done for verticalness. You can hopefully imagine this being a useful feature for further processing. Because the resulting output can be thought of as a map of the feature strength over parts of an image, we refer to it as a feature map.

![conv_with_relu](E:\CS\Computer Vision\Notes\convnet\images\conv_with_relu.PNG)

​																		             Figure 5: Detecting horizontal and vertical edge features.

Neural networks are extremely prevalent in modern technology—because they are so accurate! The highest performing CNNs today consist of an absurd amount of layers, which are able to learn more and more features. Part of the reason these groundbreaking CNNs are able to achieve such tremendous accuracies is because of their non-linearity. Non-linearity is necessary to produce non-linear decision boundaries, so that the output cannot be written as a linear combination of the inputs. If a non-linear activation function was not present, deep CNN architectures would devolve into a single, equivalent convolutional layer, which would not perform nearly as well. The activations are computed as follows:
$$
z_i = \sum_j x_j * w_{ij}\\
a_i = f(z_i)
$$
The activation function $f$ is applied elementwise. 

**ReLU**

The Rectified Linear Activation function (ReLU) applies much-needed non-linearity into the model. The ReLU activation function is specifically used as a non-linear activation function, as opposed to other non-linear functions such as *Sigmoid* because it has been [empirically observed](https://arxiv.org/pdf/1906.01975.pdf) that CNNs using ReLU are faster to train than their counterparts. ReLU is an element wise operation (applied per pixel) and replaces all negative pixel values in the feature map by zero. The ReLU operation can be understood clearly from Figure  below. It shows the ReLU operation applied to one of the feature maps obtained in Figure above. 

![relu_feature_map](E:\CS\Computer Vision\Notes\convnet\images\relu_feature_map.png)



### 6.1.3 Advantages of Convolution Layer over Fully Connecter Layer

We can think about filtering as a layer of a neural network by thinking of the elements of $x$ and $x * w$ as units, and the elements of $w$ as connection weights. Such an interpretation is visualized in Figure  for a one-dimensional example. Each of the units in this network computes its activations in the standard way, i.e. by summing up each of the incoming units multiplied by their connection weights.  This shows that a convolution layer is like a fully connected layer, except with two additional features:

- **Sparse connectivity**: not every input unit is connected to every output unit.
- **Weight sharing**: the network's weights are each shared between multiple connections.

<img src="E:\CS\Computer Vision\Notes\convnet\images\1d_conv.PNG" alt="1d_conv" style="zoom:80%;" />

​																			Figure: A convolution layer, shown in terms of units and connections.

Missing connections can be thought of as connections with weight 0. This highlights an important fact: **any function computed by a convolution layer can be computed by a fully connected layer.** This means convolution layers don't increase the representational capacity, relative to a fully connected layer with the same number of input and output units. But they can reduce the numbers of weights and connections.

For instance, suppose we have 32 input feature maps and 16 output feature maps, all of size 50 x 50, and the  filter are of size 5 x 5. (These are all plausible sizes for a conv net.) The number of weights for the convolution layer is 5 x 5 x 16 x 32 = 12800. The number of connections is approximately
50 x 50 x 5 x 5 x 16 x 32 = 32 million. By contrast, the number of connections (and hence also the number of weights) required for a fully connected layer with the same set of units would be (32 x 50 x 50) x (16 x 50 x 50) = 3.2 billion. Hence, using the convolutional structure reduces the number of connections by a factor of 100 and the number of weights by almost a factor of a million!																								

## 6.2 Pooling Layers

In section 2,  we observed that a neural network's classification ought to be **invariant** to small transformations of an image, such as shifting it by a few pixels. In order to achieve invariance, it is common to periodically insert a Pooling layer (also called subsampling or downsampling) in-between successive Conv layers in a ConvNet architecture. Pooling layers summarize (or compress) the feature maps of the previous layer by computing a simple function over small regions of the image. There are many types of pooling layers in different CNN architectures, but they all have the purpose of gradually decreasing the spatial extent of the network  to reduce the amount of parameters and computation, but retains the most important information and hence to also control overfitting. The Pooling Layer operates independently on every depth slice of the input and resizes it spatially. 

Suppose we have input feature maps $x_1,\dots,x_N$. Each unit of the output map computes the maximum over some region (called a pooling group) of the input map. (Typically, the region could be 3 x 3.) In order to shrink the representation, we don't consider all offsets, but instead we space them by a stride S along each dimension. This results in the representation being shrunk by a factor of approximately S along each dimension. (A typical value for the stride is 2.) Most commonly, this function is taken to be the maximum, so the operation is known as **max-pooling**.

Spatial Pooling can be of different types: Max, Average, Sum etc. In case of Max Pooling, we select a kernel size (for example, a 2×2 window) and a stride length during architecture design. Once selected, the operation slides the kernel with the specified stride over the input while only selecting the largest value at each kernel slice from the input to yield a value for the output. Instead of taking the largest element we could also take the average (Average Pooling) or sum (Sum Pooling) of all elements in that window. Here is a visualization.

<img src="E:\CS\Computer Vision\Notes\convnet\images\pooling.gif" alt="pooling" style="zoom: 80%;" />

In practice, Max Pooling has been shown to work better. Average pooling was often used historically but has recently fallen out of favor compared to the max pooling operation. The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2 down-samples every depth slice in the input by 2 along both width and height, discarding 75% of the activations. Every MAX operation would in this case be taking a max over 4 numbers (little 2x2 region in some depth slice).  For instance, Figure shows an example of input volume of size [224x224x64] is pooled with filter size 2, stride 2 into output volume of size [112x112x64]. Notice that the volume depth is preserved.

<img src="E:\CS\Computer Vision\Notes\convnet\images\pool.jpeg" alt="pool" style="zoom: 80%;" />

It is worth noting that there are only two commonly seen variations of the max pooling layer found in practice: A pooling layer with $F=3,S=2F=3,S=2$ (also called overlapping pooling), and more commonly $F=2,S=2F=2,S=2$. Pooling sizes with larger receptive fields are too destructive.

Figure  shows an example of how pooling can provide partial invariance to translations of the input. Observe that the first output does not change,
since the maximum value remains within its pooling group.

<img src="E:\CS\Computer Vision\Notes\convnet\images\pooling_invariance.PNG" alt="pooling_invariance" style="zoom:80%;" />

​															Figure: An example of how pooling can provide partial invariance to translations of the input.

Pooling also has the effect of increasing the size of units' receptive fields, or the regions of the input image which influence their activations. For instance, consider the network architecture in Figure , which alternates between convolution and pooling layers. Suppose all the filters are 5 x 5 and the pooling layer uses a stride of 2. Then each unit in the first convolution layer has a receptive field of size 5 x 5. But each unit in the second convolution layer has a receptive field of size approximately 10 x 10, since it does 5 x 5 filtering over a representation which was shrunken by a factor of 2 along each dimension. A third convolution layer would have 20 x 20 receptive fields. Hence, pooling allows small filters to account for information over large regions of an image.

![pooling4](E:\CS\Computer Vision\Notes\convnet\images\pooling4.PNG)

Figure: Schematic of a conv net with convolution and pooling layers. Pooling layers expand the receptive fields of units in subsequent convolution layers.

More generally, the pooling layer:

- Accepts a volume of size $W_1×H_1×D_1$
- Requires two hyperparameters:
  - their spatial extent $F$
  - the stride $S$
- Produces a volume of size $W_2×H_2×D_2$ where
  - $W_2 = (W_1 - F)/S + 1$
  - $H_2 = (H_1 - F)/S + 1$
  - $D_2 = D_1$
- Introduces zero parameters since it computes a fixed function of the input.
- For Pooling layers, it is not common to pad the input using zero-padding.

**Getting rid of pooling**. Many people dislike the pooling operation and think that we can get away without it. For example, [Striving for Simplicity: The All Convolutional Net](http://arxiv.org/abs/1412.6806) proposes to discard the pooling layer in favor of architecture that only consists of repeated CONV layers. To reduce the size of the representation they suggest using larger stride in CONV layer once in a while. Discarding pooling layers has also been found to be important in training good generative models, such as variational autoencoders (VAEs) or generative adversarial networks (GANs). It seems likely that future architectures will feature very few to no pooling layers.

**Summary**. To summarize, the conv layer:

- makes the input representations (feature dimension) smaller and more manageable
- reduces the number of parameters and computations in the network, therefore, controlling overfitting
- makes the network invariant to small transformations, distortions and translations in the input image (a small distortion in input will not change the output of Pooling – since we take the maximum / average value in a local neighborhood).
- helps us arrive at an almost scale invariant representation of our image (the exact term is “equivariant”). This is very powerful since we can detect objects in an image no matter where they are located



## 6.3 Fully Connected Layers 

The Fully Connected (**FC**) layer is a regular neural network layer. Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset.  In convNets, the first FC layer is  called the **flatten layer**, which converts a three-dimensional layer in the network into a one-dimensional vector to fit the input of a fully-connected layer for classification. For example, a 5x5x2 tensor would be converted into a vector of size 50.

The output from a sequence of convolutional and pooling layers represent high-level features of the input image. The purpose of the Fully Connected layer is to use these features for classifying the input image into various classes based on the training dataset using the **Softmax Classification** technique.

<img src="E:\CS\Computer Vision\Notes\convnet\images\fc.PNG" alt="fc" style="zoom: 67%;" />

Apart from classification, adding a fully-connected layer is also a (usually) cheap way of learning non-linear combinations of these features. Most of the features from convolutional and pooling layers may be good for the classification task, but combinations of those features might be even better.





# 7. Training ConvNets by Backpropagation



## Overall Training Process of ConvNets







# Visualizing and Interpreting What ConvNets Learn



# Modern ConvNets Architecture Patterns



# ConvNet Architectures



# Transfer Learning



# ConvNets in Practice

## A Recipe for Training ConvNets



## Structuring ConvNets Projects



## Computational Considerations

The largest bottleneck to be aware of when constructing ConvNet architectures is the memory bottleneck. Many modern GPUs have a limit of 3/4/6GB memory, with the best GPUs having about 12GB of memory. There are three major sources of memory to keep track of:

- From the intermediate volume sizes: These are the raw number of **activations** at every layer of the ConvNet, and also their gradients (of equal size). Usually, most of the activations are on the earlier layers of a ConvNet (i.e. first Conv Layers). These are kept around because they are needed for backpropagation, but a clever implementation that runs a ConvNet only at test time could in principle reduce this by a huge amount, by only storing the current activations at any layer and discarding the previous activations on layers below.
- From the parameter sizes: These are the numbers that hold the network **parameters**, their gradients during backpropagation, and commonly also a step cache if the optimization is using momentum, Adagrad, or RMSProp. Therefore, the memory to store the parameter vector alone must usually be multiplied by a factor of at least 3 or so.
- Every ConvNet implementation has to maintain **miscellaneous** memory, such as the image data batches, perhaps their augmented versions, etc.

Once you have a rough estimate of the total number of values (for activations, gradients, and misc), the number should be converted to size in GB. Take the number of values, multiply by 4 to get the raw number of bytes (since every floating point is 4 bytes, or maybe by 8 for double precision), and then divide by 1024 multiple times to get the amount of memory in KB, MB, and finally GB. If your network doesn’t fit, a common heuristic to “make it fit” is to decrease the batch size, since most of the memory is usually consumed by the activations.

# ConvNets for NLP



# Conclusion
