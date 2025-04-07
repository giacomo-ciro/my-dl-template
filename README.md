# my-dl-template
A template repository for developing deep learning pipelines.

- `config.json`     configuration for the entire project (hyper-parameters, paths to files etc.)
- `train.py`        main script handling the training logic
- `model.py`        custom implementation of a lightning model (forwards a batch, returns loss and prediction)
- `datamodule.py`   custom implementation of a lightning datamodule (handles reading data, preparing and returning batches)
- `utils.py`        any additional helper object