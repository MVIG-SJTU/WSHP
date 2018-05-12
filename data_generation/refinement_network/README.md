# Refinement Network

This is the code for the Refinement Network. We use Refinement Network to generate parsing label for single person.

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch 0.3.0 and dependencies from http://pytorch.org
- Install Torch vision from the source.
```bash
git clone https://github.com/pytorch/vision
cd vision
python3 setup.py install
```

- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip3 install visdom
pip3 install dominate
```

- Clone this repo:
```bash
git clone https://github.com/MVIG-SJTU/WSHP
cd WSHP/data_generation/refinement_network
```

### Train/Test
#### Train
- Prepare a training dataset, which should have the following directories:
```
/dataroot
    /img
        img1.ext
        img2.ext
        ...
    /parsing
        img1.ext
        img2.ext
        ...
    /prior
        /img1
            0.png
            1.png
            ...
        /img2
            0.png
            1.png
            ...
        ...
```
In our project, we use `Pascal` dataset to train our Refinement Network. But to train refinement network, besides origin image and parsing label, we also need prior images. In `datasets` directory, we give some code to show how to do this. You can use any dataset which has both two kinds of annotations. [Here](https://drive.google.com/open?id=1Ck8_1m74aLGDhawIbsdYsCee9_sFiBcE) is some data you can use.

- Train a model:
```bash
#!./scripts/train.sh
python3 train.py --dataroot /path/to/dataset --dataset_mode aligned --model pix2pix --no_gan --shuffle --n 5 --k 3 --output_nc 1 --name exp1
```

- To view training results and loss plots, run `python3 -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out `./checkpoints/exp1/web/index.html`

#### Test
- Prepare a testing dataset, which should have the following directories:
```
/dataroot
    /img
        img1.ext
        img2.ext
        ...
    /prior
        img1.ext
        img2.ext
        ...
```
For more details, please refer to the [parent module](https://github.com/Fang-Haoshu/WSHP/data_generation) where we discuss how to generate prior for dataset which has only keypoints information.

- Test the model:
```bash
#!./scripts/test.sh
python3 test.py --dataroot /path/to/dataset --dataset_mode single --model test --output_nc 1 --name exp1
```
The test results will be saved to a html file here: `./results/exp1/test_latest/index.html`.

## Training/Test Details
- Flags: see `options/train_options.py` and `options/base_options.py` for all the training flags; see `options/test_options.py` and `options/base_options.py` for all the test flags.
- CPU/GPU (default `--gpu_ids 0`): set`--gpu_ids -1` to use CPU mode; set `--gpu_ids 0,1,2` for multi-GPU mode. You need a large batch size (e.g. `--batchSize 32`) to benefit from multiple GPUs.
- Visualization: during training, the current results can be viewed using two methods. First, if you set `--display_id` > 0, the results and loss plot will appear on a local graphics web server launched by [visdom](https://github.com/facebookresearch/visdom). To do this, you should have `visdom` installed and a server running by the command `python3 -m visdom.server`. The default server URL is `http://localhost:8097`. `display_id` corresponds to the window ID that is displayed on the `visdom` server. The `visdom` display functionality is turned on by default. To avoid the extra overhead of communicating with `visdom` set `--display_id 0`. Second, the intermediate results are saved to `[opt.checkpoints_dir]/[opt.name]/web/` as an HTML file. To avoid this, set `--no_html`.
- Preprocessing: images can be resized and cropped in different ways using `--resize_or_crop` option. The default option `'resize_and_crop'` resizes the image to be of size `(opt.loadSize, opt.loadSize)` and does a random crop of size `(opt.fineSize, opt.fineSize)`. `'crop'` skips the resizing step and only performs random cropping. `'scale_width'` resizes the image to have width `opt.fineSize` while keeping the aspect ratio. `'scale_width_and_crop'` first resizes the image to have width `opt.loadSize` and then does random cropping of size `(opt.fineSize, opt.fineSize)`.

## Acknowledgments
Code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
