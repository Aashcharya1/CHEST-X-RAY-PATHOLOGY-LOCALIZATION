Chest X-Ray Pathology Localization using Multimodal Autoencoder
This project implements a deep learning model to automatically localize and segment pathologies in chest X-ray images based on textual findings from radiology reports. It utilizes a Multimodal U-Net Autoencoder architecture that fuses visual features from the X-ray with semantic features from the text report.

Project Description
The core of this project is a two-phase training approach designed to handle limited labeled medical data:

Phase 1: Self-Supervised Autoencoder Pre-training

Goal: Teach the model to understand the structure and features of Chest X-rays without needing manually drawn masks.

Method: A standard U-Net Autoencoder (with a ResNet34 backbone) is trained on a large set of ~7,400 unlabeled X-ray images. The objective is simple image reconstruction (Input Image → Output Image), forcing the encoder to learn robust visual representations of the lungs and thorax.

Phase 2: Multimodal Supervised Fine-Tuning

Goal: Train the model to generate a binary mask highlighting the disease described in the text report.

Method: The pre-trained image encoder from Phase 1 is combined with a frozen CLIP (Contrastive Language-Image Pre-Training) text encoder.

Architecture:

Image Encoder: ResNet34 (initialized with Phase 1 weights).

Text Encoder: CLIP (ViT-B/32) processes the "findings" text.

Fusion: Text and Image embeddings are concatenated at the bottleneck.

Decoder: A U-Net decoder upsamples the fused features to produce the final segmentation mask.

Dataset
The project uses the Indiana University Chest X-Ray Collection (OpenI).

Source: Kaggle / OpenI

Structure:

images/images_normalized/: Contains ~7,400 unlabeled X-ray images (used for Phase 1).

masked/: Contains ~240 binary masks corresponding to specific pathologies (used for Phase 2 targets).

indiana_reports.csv: Contains the text reports (findings/impressions) linked by UID.

indiana_projections.csv: Maps UIDs to filenames.

Requirements
Python 3.x

PyTorch (Deep Learning Framework)

Segmentation Models PyTorch (smp) (For U-Net implementation)

Transformers (HuggingFace) (For CLIP model)

Pandas (Data manipulation)

Pillow (PIL) (Image loading)

Scikit-learn (Data splitting)

Matplotlib (Visualization)

Installation:

Bash

pip install torch torchvision transformers segmentation-models-pytorch pandas matplotlib scikit-learn
Model Architecture
The Autoencoder (U-Net)
The model follows a U-Shape architecture with skip connections:

Encoder (Downsampling): A ResNet34 backbone extracts hierarchical features from the input X-ray image (256x256).

Bottleneck (Fusion): * The deepest image features (512 channels) are extracted.

The textual report is tokenized and passed through CLIP to get a 512-dimensional embedding.

The text embedding is expanded and concatenated with the image features.

A convolution layer fuses them back to the original channel size.

Decoder (Upsampling): Transpose convolutions upsample the features back to the original resolution, using skip connections from the encoder to preserve spatial detail.

Segmentation Head: A final 1x1 convolution produces the binary mask prediction.

How to Use
1. Data Setup
Ensure your dataset is uploaded to Google Drive as a zip file (Dataset.zip) or available locally. The code will automatically copy and unzip it to the Colab/local environment for speed.

2. Phase 1: Pre-training (Unsupervised)
Run the Phase 1 training cell. This trains the U-Net to act as a standard autoencoder, reconstructing the input images.

Input: Unlabeled X-ray images.

Loss: Mean Squared Error (MSE).

Output: Saves my_pretrained_encoder.pth.

3. Phase 2: Fine-tuning (Multimodal)
Run the Phase 2 training cell. This loads the encoder weights from Phase 1 and trains the full multimodal system.

Input: X-ray Image + Text Report.

Target: Ground Truth Mask.

Loss: Combined Dice Loss + Binary Cross Entropy (BCE) with class weighting (to handle small mask areas).

Output: Saves best_model.pth.

4. Visualization
Run the inference cell to visualize results. It displays:

Original Image: The input X-ray.

Ground Truth: The manual annotation.

Predicted Mask: The model's generated heatmap/mask.

Acknowledgments
Segmentation Models PyTorch: For the efficient U-Net implementation.

OpenAI: For the pre-trained CLIP model.

Indiana University/OpenI: For the public medical dataset.