# Semantic Segmentation with PyTorch on Aerial Drone Dataset

This is a quick exploration of semantic segmentation using PyTorch on the Aerial Semantic Segmentation Drone Dataset. The project aims to provide a basic understanding of training segmentation models and experimenting with different architectures.

**Note:** Code is terrible; require refactoring.


<video controls>
  <source src="plots/video_bad_augs.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Dataset
- **Name:** Aerial Semantic Segmentation Drone Dataset
- **URL:** [Dataset Link](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset)

## Training
- Utilized the simplest out-of-the-box training recipe found online.
- Trained for 10 epochs using a simple UNET-like architecture with a MobileNetV2 pretrained encoder.
- Batch size: 3 (limited by personal PC memory)
- Training time: 24 minutes on an RTX 3080 (2.4 minutes per epoch)

## Model and Augmentations
- Explored basic model architectures like UNET.
- Augmentations including RandomCropResize, ColorJitter, etc., were applied to enhance training.
- Experimented with different segmentation models and encoders from segmentation_models.pytorch and timm libraries.

## Metrics and Evaluation
- Focused on improving mean Intersection over Union (mIoU) metric.
- Saved models whenever mIoU improved or at every epoch.
- Tested multiple models from different training stages to evaluate performance.

## Post-processing
- Merged certain classes in the predicted masks based on safe landing labels (e.g., grass, gravel, dirt).
- Applied strategies like blurring to remove noise from the segmentation mask.

## Future Improvements
- Train for longer durations to observe potential performance improvements (only trained for 10 epochs +- 30min.).
- Experiment with different model architectures, augmentations, and datasets for validation.
- Refactor code for better readability and reusability.

## Results
- Segmentation results are visualized in the provided images:
  - ![seg1.png](/plots/seg1.png)
  - ![seg2.png](/plots/seg2.png)
  - ![seg3.png](/plots/seg3.png)


