# CodeFormer
Towards Robust Blind Face Restoration with Codebook Lookup TransFormer, based on https://github.com/sczhou/CodeFormer.


## Dependencies
- [NumPy](https://numpy.org/install)
- [OpenCV-Python](https://github.com/opencv/opencv-python)
- [PyTorch](https://pytorch.org/get-started) 1.13.1
- [VapourSynth](http://www.vapoursynth.com/) R55+


## Installation
```
pip install -U vscodeformer
python -m vscodeformer
```


## Usage
```python
from vscodeformer import codeformer

ret = codeformer(clip)
```

See `__init__.py` for the description of the parameters.
