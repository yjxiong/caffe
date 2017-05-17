## Dec 25, 2016

  - Supported dilated convolution
  - Memory optimization is introduced to save memory during training and testing. [Wiki](https://github.com/yjxiong/caffe/wiki/Memory-Optimization)
  - `BatchReductionLayer` supports reduction on an arbitrary axis with cuda implementation.
  - Other small fixes.

## Apr 27, 2016

Features:

  - Supported cuDNN v5
  - Use the cuDNN's BatchNormalization implementation as the default engine for BN layer
  - BN layer will now store running variance in its fourth blob.
  - the script `python/bn_convert_style.py` is added to help converting the bn style forth and back.

## Dec 23, 2015

Features:

  - Implemented a planning algorithm to globally optimize the cudnn workspace consumption and speed trade-off.
  - Now `richness` parameter specifies the total memory in MBs available to cudnn for convolution workspaces.
  - Now the framework will try to find the best convolution algorithm combinations under memory limit.
  
## Dec 17, 2015

Features:

  - cuDNN v4 support
  - 20% overall speed gain with faster convolution and batch normalization
  - the native batch normalization is changed to comply with cuDNN. Use the script `python/bn_var_to_inv_std.py` to upgrade your models.
  
## Nov 22, 2015

Features:
  - python layer can expose a prefetch() method, which will be run in parallel with network processing.
  
## Oct 13, 2015

Features:
  - Improved cuDNN wrapper to use less GPU memory. 
  - Now there is a new parameter `richness` which controls the limit of workspace for cuDNN.
  
## Sep 30, 2015

Features:
  - Support for cuDNN v3.
  
## Sep. 7, 2015

Features:
  - New mechanism for parallel comminucation reduced parallel overhead.
  - Batch normalization, courtesy of @Cysu.
  
## Jul, 2015

Features:
  - Action recognition tools, scripts, and examples.
  - Basic parallel training support
  - Various extra data augmentations
