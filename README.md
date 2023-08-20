# A Deep Learning Approach for Meibomian Gland Atrophy Evaluation in Meibography Images

This repository implements [A Deep Learning Approach for Meibomian Gland Atrophy Evaluation in Meibography Images](https://doi.org/10.1167/tvst.8.6.37). More specifically, the gland segmentation of meibography images.

## Usage 

To follow the training routine in train.py you need a DataLoader that yields the tuples of the following format:

(Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y\_cls) where

x - batch of input images,

y - batch of groung truth seg maps,

y\_cls - batch of 1D tensors of dimensionality N: N total number of classes, 

y\_cls[i, T] = 1 if class T is present in image i, 0 otherwise

## License and Citation
The use of this software is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
```
@article{wang2019deep,
  title={A deep learning approach for meibomian gland atrophy evaluation in meibography images},
  author={Wang, Jiayun and Yeh, Thao N and Chakraborty, Rudrasis and Stella, X Yu and Lin, Meng C},
  journal={Translational vision science \& technology},
  volume={8},
  number={6},
  pages={37--37},
  year={2019},
  publisher={The Association for Research in Vision and Ophthalmology}
}
```