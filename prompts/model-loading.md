 ### Prompt

Currently, the detector in @src/litter_detector/detector/  loads the model from mlflow directly. The mlflow storage is local and not saved through vsc, as it would cause too many merge issues. Additionally to the mlflow model, the model is also saved as a .pth file inside @models/ folder. The current is how we can distribute trained models easier within our team and across machines so the detector can use the trained model for inference.
  Your task: Research and recommend potential solutions for distributing trained models, preferably through git vsc if possible. If a suitable solution is found, add this model functionality to the @src/litter_detector/detector/ with args to select the model (mlflow / local and which model).


| Metric                              | Score                  |
|-------------------------------------|------------------------|
| **Tool used**                       | Claude Code (Opus 4.6) |
| **Error Rate (0 - 4)**              | 4                      |
| **Code Quality (0 - 4)**            | 3                      |
| **Discrepancy from Prompt (0 - 4)** | 3                      |
| **Notes**                           | Started in Plan Mode   |