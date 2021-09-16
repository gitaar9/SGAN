# π-GAN Prerelease Code

**This code has not yet been publicly released. PLEASE DO NOT SHARE**

### Training a Model

The main training script can be found in train.py, and may be called via cli.

##### Relevant Flags:

Set the output directory:
`--output_dir=[output directory]`
--Default: `output/spatialsirenbaseline`
--Controls where the training script will write images, models, and logging information to.

Set the loading directory:
`--load_dir=[load directory]`
--Default: `None`
--If set, controls where the training script will look for models to continue runs from. This directory should be `--output_dir` from the prior run. If not set, will begin a new run.

Set the current curriculum:
`--curriculum=[curriculum]`
--Default: `SPATIALSIRENBASELINE`
--Curriculums control model-specific variables such as learning rate, regularization parameters, loss functions, etc...

Set the port:
`--port=[port]`
--Default: `12355`
--Sets the internal port for distributed training. Must each be unique if running multiple tests simultaneously.


##### To start training:

On one GPU:
`CUDA_VISIBLE_DEVICES=0 python3 train.py --flag=value --flag=value ...`

On multiple GPUs, simply list cuda visible devices in a comma-separated list:
`CUDA_VISIBLE_DEVICES=1,3 python3 train.py --flag=value --flag=value ...`

__To continue training from another run:__
Specify the `--load_dir=path/to/directory` flag. See the "Relevant Flags" section for details.

#### Evaluating a Model
`python eval_metrics.py path/to/generator.pth --real_image_dir path/to/real_images/directory --num_images 8000`

**Notes on Evaluation**

We log FID scores during training, but these should be used only to gauge rough performance. Run the separate evaluation script for a more accurate evaluation.

Note that the number of images generated for evaluation has a huge impact on GAN metrics, particularly FID. Ensure that you compare all models with the same number of generated images.

For evaluation, run the model at the *exact* settings the model was trained at (but use the EMA).

#### Visualizing Images
Render images of scenes from different angles.

`python render_multiview_images.py path/to/generator.pth --seeds 0 1 2 3`

Optionally, you can pass the flag `--lock_view_dependence` to remove view dependent effects.

For best visual results, load the EMA parameters, use truncation, increase the resolution (e.g. to 512 x 512) and increase the number of depth samples (e.g. to 24 or 36).

#### Rendering Videos
Render videos of scenes.

`python render_video.py path/to/generator.pth --seeds 0 1 2 3`

Optionally, you can pass the flag `--lock_view_dependence` to remove view dependent effects. This can help mitigate distracting visual artifacts such as shifting eyebrows. However, locking view dependence may lower the visual quality of images (edges may be blurrier etc.)

#### Extracting Shapes

Extract the 3D shape of a scene by running the following:

`python3 shape_extraction.py path/to/generator.pth --seed 0`

This will export the scene as a .mrc file. Visualize the shape by downloading ChimeraX, loading in the mrc, and setting the level to ~10. Alternatively, you can treat the .mrc file as a 256 x 256 x 256 occupancy cube and run marching cubes.

#### Inverse Rendering
Need to clean this up. Let me know if you want the messy code.

#### Changes/additions since original implementation

1. Added experimental pose identity loss. Controlled by pos_lambda in the curriculum, helps ensure generated scenes share the same canonical pose. Empirically, it seems to improve 3D models, but may introduce a minor decrease in image quality scores.

2. Added script for latent code interpolation in W-space.

3. Added options for truncation, following implementation in StyleGAN.

4. Tweaks to hyperparmeters, e.g. learning rate and initialization. Should result in improved evaluation metrics.


### Training Tips

Training has been tested to work well on either two RTX 6000's or one RTX 8000. Training on a smaller GPU might work fine, but it's also possible you'll encounter instability, especially at higher resolutions. Bubbles and artifacts that suddenly appear, or blurring in the tilted angles, are signs that training destabilized. This can usually be mitigated by training with a larger batch size or by reducing the learning rate.



### Files
The discriminators, generators, and siren folders contain files I changed very little from the original implementation.
Jobscripts contains all the jobscripts I used to run the experiments on the peregrine cluster.

**calculate_simm.py:**
A script I used for preliminary evaluation of novel view synthesis quality.

**curriculums.py:**
Contains the settings for the different experiments.

**datasets.py:**
Dataloader etc. for ShapeNet, CARLA and legacy datasets.

**eval_metrics.py:**
The evaluation script used for the final results.

**fid_evaluation.py:**
Outdated evaluation script

**graf_metrics.py:**
Script to perform the eval_metrics.py evaluation on the GRAF model results.

**inverse_render_dataset.py:**
Generate novel views for simm evaluation.

**inverse_render_dataset_creation.py:**
This script generates novel views for dataset that can be used for the data augmentation experiments.

**mirror_train.py:**
My version of train.py with a symmetry-based loss.

**plot_results.py:**
A script to download log files from peregrine and plot the results.

**render_image_carla.py:**
Render an image of a random latent code using a trained generator.

**render_video_carla.py:**
Render a video of a random latent code using a trained generator.

**resnet_instance_recognition.py:**
Script to train the resnet classifier for the data augmentation experiments.

**train.py:**
Original training script without the symmetry-based loss.

**try_3d_plotting.py:**
Script I used for the creation of some plots and getting a grasp of NeRF.

**resnet_instance_classification/simple_dataset.py:**
Dataloaders for the resnet_instance_recognition.py script.

