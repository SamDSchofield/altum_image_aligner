# MicaSense Altum image aligner
Script for aligning images taken with the MicaSense Altum camera.

## Usage
```
image_aligner.py /path/to/image /path/to/output/directory
```

The script expects all the images to be saved in a single directory and follow the following naming convention:
```
/path/to/image_directory/IMG_m_n.tif
```
where `m` is the sequence number and `n` is the spectral id.