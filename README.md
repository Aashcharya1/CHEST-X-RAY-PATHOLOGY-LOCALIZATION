Chest X-Ray Pathology Localization
Multimodal autoencoder framework that conditions image segmentation on textual radiology findings to deliver precise, interpretable localization of thoracic pathologies.

## Executive Summary
Radiology reports often describe abnormalities that are visually subtle or spatially diffuse on chest X-rays. This project links the narrative report to the corresponding image, enabling the model to surface regions that match the described pathology. A two-stage training curriculum—self-supervised visual pre-training followed by multimodal fine-tuning—maximizes the value of limited pixel-level annotations while maintaining clinical interpretability.

- **Problem**: Localize pathologies in chest radiographs given the accompanying free-text report.
- **Solution**: ResNet34 U-Net encoder fused with CLIP ViT-B/32 text embeddings at the bottleneck.
- **Outcome**: Produces binary masks and overlay visualizations aligned with radiologist findings.
- **Artifacts**: `my_pretrained_encoder.pth` (Phase 1) and `best_model.pth` (Phase 2).

## Key Features
- **Multimodal fusion** that leverages frozen CLIP text embeddings for semantic grounding.
- **Curriculum learning** approach improves generalization when labeled masks are scarce.
- **Single-notebook workflow** (`Main.ipynb`) for data setup, training, evaluation, and visualization.
- **Interpretable outputs** through side-by-side image, ground-truth mask, and predicted overlay.
- **Hardware-aware design** supporting CPU, single GPU, or Colab runtimes with AMP training.

## Repository Structure
| Path | Description |
| --- | --- |
| `Main.ipynb` | Primary notebook covering setup, pre-training, fine-tuning, inference, and visualization. |
| `README.md` | Project documentation, setup instructions, and roadmap (this file). |

## Dataset Expectations
- **Source**: Indiana University Chest X-Ray Collection (OpenI, mirrored on Kaggle).
- **Folders & Files**  
  - `images/images_normalized/` – ~7,400 unlabeled PA views for Phase 1 reconstruction.  
  - `masked/` – ~240 binary masks paired with specific UIDs for Phase 2 supervision.  
  - `indiana_reports.csv` – Findings and impression text linked via UID.  
  - `indiana_projections.csv` – UID-to-image filename mapping.  
- **Preprocessing Guidelines**  
  - Resize or pad to 256×256, normalize pixel intensities (z-score or min-max).  
  - Remove PHI, handle missing reports, and clean boilerplate phrases.  
  - Tokenize text with the CLIP BPE tokenizer (handled in-notebook).  
  - Create patient-level splits to prevent leakage across train/val/test sets.  

## Environment & Tooling
1. **Python**: 3.9 or newer.  
2. **Core Dependencies**:
   ```
   pip install torch torchvision transformers segmentation-models-pytorch \
               pandas scikit-learn pillow matplotlib
   ```
3. **Optional GPU acceleration**:  
   - Verify CUDA via `nvidia-smi`.  
   - Install the PyTorch build matching your CUDA toolkit.  
4. **Recommended utilities**: Weights & Biases or TensorBoard for experiment tracking.  

## Architectural Overview
- **Image Encoder**: ResNet34 backbone (ImageNet initialization) within a U-Net encoder path.  
- **Text Encoder**: CLIP ViT-B/32 (frozen). Produces a 512-d embedding per report.  
- **Fusion Mechanism**: Concatenate text embedding with the deepest visual feature map, then project back to the encoder channel width.  
- **Decoder**: U-Net decoder with skip connections restoring spatial resolution.  
- **Head**: 1×1 convolution + sigmoid to yield a binary pathology mask.  

## Training Workflow
### Phase 1 · Self-Supervised Visual Pre-Training
- **Objective**: Learn chest X-ray structure via reconstruction.  
- **Loss**: Mean Squared Error, optionally supplemented with SSIM.  
- **Outputs**: Encoder checkpoint `my_pretrained_encoder.pth`, PSNR/SSIM diagnostics, sample reconstructions.  

### Phase 2 · Multimodal Fine-Tuning
- **Inputs**: Paired (image, report, mask) triplets.  
- **Training strategy**:  
  - Load Phase 1 encoder weights.  
  - Freeze CLIP encoder; train fusion module, decoder, and segmentation head.  
  - Apply Dice + BCE loss with positive-class weighting to handle small lesions.  
  - Monitor Dice/IoU on validation masks; keep the best-performing checkpoint (`best_model.pth`).  

## Notebook Walkthrough (`Main.ipynb`)
1. **Runtime configuration** – seed everything, select device, mount cloud storage if applicable.  
2. **Data module** – define PyTorch datasets/dataloaders for unlabeled and labeled splits.  
3. **Phase 1 loop** – train autoencoder, log reconstructions, and save weights periodically.  
4. **Phase 2 loop** – rehydrate encoder weights, instantiate CLIP text encoder, train multimodal U-Net.  
5. **Inference & visualization** – plot input images, ground truth, predictions, and blended overlays; export figures or GIFs.  

## Evaluation & Reporting
- **Quantitative metrics**: Dice coefficient, IoU, pixel accuracy per pathology, calibration plots.  
- **Qualitative review**: Present tri-panel figures (image / ground truth / prediction) and overlay masks for stakeholder review.  
- **Suggested analyses**:  
  - Grad-CAM or attention rollout on the encoder for interpretability.  
  - Sensitivity testing by perturbing report text to ensure the model reacts appropriately.  
  - Cross-dataset evaluation (e.g., CheXlocalize) to gauge robustness.  

## Troubleshooting & FAQ
- **Class imbalance**: Increase BCE `pos_weight`, experiment with focal or Tversky loss, or oversample minority pathologies.  
- **VRAM limitations**: Downscale to 224×224, use mixed-precision, or accumulate gradients.  
- **Noisy text**: Strip templated “normal” statements and consolidate duplicated findings to sharpen the conditioning signal.  
- **Checkpoint hygiene**: Save both encoder-only and full-model weights for flexible re-use.  
- **Training stability**: Use cosine LR schedules and early stopping on validation Dice to avoid overfitting.  

## Roadmap
- Add lightweight transformer encoders (e.g., MobileViT) for edge deployment.  
- Introduce section-aware text attention (Findings vs Impression weighting).  
- Package environment definitions (Dockerfile, `environment.yml`).  
- Extend evaluation suite to include Grad-CAM dashboards and automated report generation.  
- Integrate hyperparameter sweeps via Weights & Biases or Ray Tune.  

## Citations & Acknowledgements
- Indiana University / OpenI for the chest X-ray corpus.  
- OpenAI CLIP for robust language-vision representations.  
- `segmentation-models-pytorch` for reliable U-Net implementations.  
- PyTorch community for the training ecosystem and tooling.  

## License
Specify the license that governs redistribution and derivative work (MIT, Apache-2.0, or institution-specific). Update this section before releasing the repository publicly.  