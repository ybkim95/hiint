# HIINT: Historical, Intra- and Inter- personal Dynamics Modeling with Cross-person Memory Transformer


<p align="center">
  <img width="1000" src="https://github.com/ybkim95/hiint/assets/45308022/f6a04c0e-9bed-406e-9d07-40252ab497ee">
</p>

Accepted at ICMI 2023!

Repository contains:

* the code to conduct all experiments reported in the paper
* fine-tuned model weights
* data access link

<br>

## Get Started

1. Create an environment:

```
conda create python=3.9 -y -n multi_person_joint_eng
conda activate multi_person_joint_eng
pip3 install -r requirements.txt
```

2. Download dataset:

If needed, submit a agreement to download the dataset used in the paper. 

3. If needed, create pretrained_models folder and download model weights [here](https://drive.google.com/drive/folders/1ltV9r7PEQE2KOsW9geDQbibN8_4mEAUq?usp=sharing).

<br>

## Datasets

It is recommended to symlink the dataset root to `$ROOT/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
multi_person_joint_engagement
├── mmaction
├── tools
├── configs
├── data
│   ├── triadic
│   │   ├── rawframes_train
│   │   ├── rawframes_val
│   │   ├── triadic_train_list.txt
│   │   ├── triadic_val_list.txt
│   ├── augtriadic
│   │   ├── rawframes_train
│   │   ├── rawframes_val
│   │   ├── augtriadic_train_list.txt
│   │   ├── augtriadic_val_list.txt
│   ├── ...
```

For more information on data preparation, please see [data_preparation.md](data_preparation.md)


<br>


## Train & Evaluate


### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--validate` (**strongly recommended**): Perform evaluation at every k (default value is 5, which can be modified by changing the `interval` value in `evaluation` dict in each config file) epochs during the training.
- `--test-last`: Test the final checkpoint when training is over, save the prediction to `${WORK_DIR}/last_pred.pkl`.
- `--test-best`: Test the best checkpoint when training is over, save the prediction to `${WORK_DIR}/best_pred.pkl`.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--gpus ${GPU_NUM}`: Number of gpus to use, which is only applicable to non-distributed training.
- `--gpu-ids ${GPU_IDS}`: IDs of gpus to use, which is only applicable to non-distributed training.
- `--seed ${SEED}`: Seed id for random state in python, numpy and pytorch to generate random numbers.
- `--deterministic`: If specified, it will set deterministic options for CUDNN backend.
- `JOB_LAUNCHER`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `LOCAL_RANK`: ID for local rank. If not specified, it will be set to 0.

Difference between `resume-from` and `load-from`:
`resume-from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load-from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

<br>


## Visualization (Grad-CAM from videos)

We provide a demo script to visualize GradCAM results using a single video.


```shell
python demo/demo_gradcam.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} [--use-frames] \
    [--device ${DEVICE_TYPE}] [--target-layer-name ${TARGET_LAYER_NAME}] [--fps {FPS}] \
    [--target-resolution ${TARGET_RESOLUTION}] [--resize-algorithm {RESIZE_ALGORITHM}] [--out-filename {OUT_FILE}]
```

- `--use-frames`: If specified, the demo will take rawframes as input. Otherwise, it will take a video as input.
- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `FPS`: FPS value of the output video when using rawframes as input. If not specified, it will be set to 30.
- `OUT_FILE`: Path to the output file which can be a video format or gif format. If not specified, it will be set to `None` and does not generate the output file.
- `TARGET_LAYER_NAME`: Layer name to generate GradCAM localization map.
- `TARGET_RESOLUTION`: Resolution(desired_width, desired_height) for resizing the frames before output when using a video as input. If not specified, it will be None and the frames are resized by keeping the existing aspect ratio.
- `RESIZE_ALGORITHM`: Resize algorithm used for resizing. If not specified, it will be set to `bilinear`.

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from `configs/` to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cache/torch/checkpoints`.

Get GradCAM results of a I3D model, using a video file as input and then generate an gif file with 10 fps.

   ```shell
   python demo/demo_gradcam.py configs/recognition/i3d/i3d_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
       checkpoints/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth demo/demo.mp4 \
       --target-layer-name backbone/layer4/1/relu --fps 10 \
       --out-filename demo/demo_gradcam.gif
   ```


### Result

<p align="center">
  <img width="700" height="500" src="https://user-images.githubusercontent.com/45308022/221434885-615ebe87-107d-4720-9e6c-71408deed2eb.png">
</p>



<br>





## Contact

If you have any problems with the code or have a question, please open an issue or send an email to ybkim95@media.mit.edu. I'll try to answer as soon as possible.


## Acknowledgments and Licenses

The main structure of the code is based on [mmaction2](https://github.com/open-mmlab/mmaction2), [pyskl](https://github.com/kennymckormick/pyskl) and [SimSwap](https://github.com/neuralchen/SimSwap). Thanks for sharing wonderful works!
