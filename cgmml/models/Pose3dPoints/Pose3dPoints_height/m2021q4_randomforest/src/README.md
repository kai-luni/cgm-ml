# Mini-ML-model using pose-3d-points

## Training

Run the jupyter notebook `train_on_pose3dpoints.ipynb` interactively.

## Inference

* Make sure you have a model file from the training. This is a file with the `.joblib` extension.
* Put the model file into the correct directory, e.g., `<REPO_DIR>/data/models/pose3dpointsModel/2021q4-points3d-rf-height-28k-200and201.joblib`
* Run python python `cgmml/models/Pose3dPoints/Pose3dPoints-height/2021q4-randomforest/src/inference.py --obj-file <path/to/your/obj_file.obj>`
