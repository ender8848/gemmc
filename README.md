# gemmc

gemm API for sound calculation in VNN

# Prerequisite

|required|version|
|--------|-------|
|Linux kernel|>=5.15 (tested on Ubuntu 22.04)|
|GPU hardware|nvidia GPU|
|Python|>=3.9|
|CUDA toolkit|compatible with GPU architecture|
|CUDA driver|compatible with CUDA toolkit|

The compatibility matters and below is a brief summary of cudatoolkit and PyTorch installation. 

1. check GPU opcalss and compute compatibility. 

Taking GTX 1060 as an example: 

It can be seen from the this [link](https://www.nvidia.com/en-us/geforce/graphics-cards/ompare/) that GTX 1060 belongs to Pascal architecture. 

Then we search from the web to know that **opcode class** of Pascal architecture is **SIMT**. 

After that, we can see from this [link](https://developer.nvidia.com/cuda-gpus#compute) that the **compute compatibility** of GeForce GTX 1060 is **6.1**. 

2. check compatible cudatoolkit version

use the above opcode class and compute compatibility info, we can find in [cutlass documentation](https://github.com/NVIDIA/cutlass/blob/master/media/docs/functionality.md#device-level-gemm) that the minimum compatible cudatoolkit version is **9.2+**.

3. check CUDA driver version on device using the following command

```
nvidia-smi
```

if it is not installed on device, you can jump to step 4 as cudatoolkit installation includes cuda driver. If the version is not compatible with cudatoolkit, simply uninstall the current driver. 

4. install cudatoolkit

This [link](https://developer.nvidia.com/cuda-toolkit-archive) contains cudatoolkit installation file. 

If CUDA driver version matches with cudatoolkit version, then you can exclude the cuda driver installation when installing the cudatoolkit. Otherwise simply follow the guide and install everything.

Use this command to see if cudatoolkit is installed successfully.

```
nvcc -V
```
