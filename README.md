# Region-captioning-on-images-using-dl
# Implementation based on the research paper by Andrej Karpathy and Li Fei-Fei (CVPR 2015)**
Overview
This project implements the architecture proposed in “Deep Visual-Semantic Alignments for Generating Image Descriptions”, a pioneering work that connects computer vision and natural language processing.
The main objective is to automatically generate meaningful captions for images by learning how to align visual regions in an image with words or phrases in a sentence and then use these alignments to generate fluent natural language descriptions.
The system is composed of two main parts:

1.Alignment Model — learns to associate image regions with words or sentence fragments in a shared embedding space.

2.Generative Model — uses the learned visual-semantic features to generate descriptive sentences using a recurrent neural network.
