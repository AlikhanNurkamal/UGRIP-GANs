# Tumor mask generation with a brain mask

## Dataset

## Generator output post-processing: my approaches
I apply the Softmax activation function to an output of the very last layer of the Generator with respect to the channels dimension (`dim=1`). Since the Generator outputs a 4-channel 3D image of shape 128x128x128, then each channel now contains a probability value.

First thing that I did was that I took an argmax with respect to the channels dimension so that the result of this operation (0 or 1 or 2 or 3) represented one of the output classes - `0: brain mask`, `1: tumor core`, `2: whole tumor`, `3: enhancing tumor`. However, there are 2 main issues with this approach:
1. Argmaxing results in a squeezed 3-dimensional tensor (of shape 128x128x128) with only 0, 1, 2, and 3 values. All these values stand for different output classes, so the background class is the problem here. That is, the output image is either a brain mask or one of the tumor masks...
2. Multi-label problem??

Second idea that I came up with was to add the background label, 0, before argmaxing...

The next idea was to first threshold the brain mask channel from the output of the Generator. The thresholding parameter was chosen to be 0.5. So, if the 0th channel of the output of the Generator contained values greater than the threshold, then they were set to 1, otherwise to 0. This allowed me to separate the foreground (brain segmentation mask) and the background.
