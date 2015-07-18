# AdaptiveLearning

The aim of this project is to explore various adaptive learning techniques
as an alternative to the standard gradient descent approach
for convex optimization problems. 

We establish the merits of adaptive learning rates over fixed learning rates by implementing several techniques such as Momentun, Nestorov’s Accelerated Gradient, AdaGrad, and AdaDelta on different datasets including

  - Digits (digit image data for OCR task),
  - 20newsgroup (documentcollection) and 
  - Labeled Faces in the Wild (LFW) (face image data). 

Comparing the accuracies with the baseline Stochastic gradient descent, we conclude that all adaptive learning techniques perform considerably better than SGD for all the datasets experimented on. Details in the report!

Running Instructions
--------------------
python AdaptiveLearningDNN.py

The code executes SGD, Momentum, NAG, AdaGrad and AdaDelta for all Digits, LFW and 20newsgroup datasets.
