# tf-dist

# Usage

## Setting Up TFRecords Dataset
Use train_val_split.py to split folder of categorical subfolders into train and validation folders.  this will be used to get data into format for tf.inception's build_image_data.py script for TFRecord creation.

Use build_image_data.py to create TFRecord database with the following syntax:
```
python build_image_data.py --train_directory=<PATH_TO_THE_UNZIPPED_TRAIN_DIRECTORY> --validation_directory=<PATH_TO_THE_VALIDATION_DIRECTORY> --output_directory=<PATH_TO_SAVE_TF_RECORDS> --labels_file=<PATH_TO_THE_CREATED_LABELS_FILE>
```
E.g.
```
python build_image_data.py --train_directory=train/ --validation_directory=val/ --output_directory= --labels_file=labels_file.txt
```
trainer.py uses the training shards created as the input data.

The dataset used with this script can be found here:
* [Distracted Driver TFRecord Dataset](https://drive.google.com/open?id=1FYrVAszEFMNTUdObK8SrKOqM8bwVxSPl)

## Generate Scripts
We'll need shell scripts which can be qsubbed for each node we allocate for the job.  Run buildscripts.sh to generate the ps and worker scripts and the trainer.sh script which will qsub them. trainer.sh is called to allocate nodes and start training.
To generate scripts for 1 ps server and 4 worker nodes, and the model defined at trainer.py (use your defined model here), you would run the command
```
buildscripts.sh 1 4 trainer.py
```
*trainer.py reads from tf records the image and label in batches of BATCH_SIZE and runs them through a 5 layer CNN (3c2d)
	NOTE: currently trainer.py is incompatible with tensorboard.
*mnist_trainer.py runs mnist training using a 3 layer CNN (2c1d) and logs to tensorboard.

Shell scripts are used to qsub nodes and train on cluster. scripts put their ids into a nodes file which is used to call trainer.py with the node information.

## Run Training
After the node scripts have been generated, you can run training by running
```
./trainer.sh
```
Nodes will begin being allocated.  Use qstat to check if nodes are in running state.
To stop training and cleanup nodes run
```
./delnodes.sh
```
To clear the output and error directories run
```
./clearout.sh
```

## Remote Tensorboard Setup
To use tensorboard, qsub tensorboard.sh which will open a tensorboard instance using logdir=logs.  open a local terminal and run
```
ssh -L localhost:16006:c002-n001:6006 colfaxc002
```
where c002-n001 can be replaced with the hostname of the node allocated for the job. open a browser and enter localhost:16006.

## Running Inference and Validation
Inference for TFRecord trainer.py can be run by qsubbing infer.sh.
Validation will be added in a future commit

