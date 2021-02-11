# Partially Observed Exchangeable Modeling

This is the official implementation of POEx.

## requirements
```
open3d == 0.9.0
pillow == 8.0.1
pytorch == 1.4.0
tensorflow == 1.15.0
tensorflow-probability == 0.5.0
tensorpack == 0.9.8
```

## usage
``` python
python scripts/run.py --cfg_file=/path/to/config --mode=train
```

Please refer to `exp` folder for exempler config files.

See `scripts` folder for evaluation code of each application.

## acknowledgements
Code for computing point cloud metrics (CD, EMD) are taken from public implementation. Please cite their work if you use this repo.
