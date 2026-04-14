### Prompt: 
For training the litter detection model, several students will be training models.
Currently, the model training gets stopped after a certain training time has been reached. Training time as a stop metric makes little sense as the training speed is dependent on the hardware of each student, the size of the model and other factors. To fix this issue, the training should run for a specified number of epochs before stopping. In Mlflow the duration should also reflect the number of epochs.

Your Task: Rewrite the train.py to include the number of epochs as a parameter and stop after reaching the specified number of epochs, then log the model to mlflow.
If there is a better approach to compare trained models across different training runs and machines, let me know.

| Metric                      | Score                |
|-----------------------------|----------------------|
| **Tool used**               | Claude Code          |
| **Error Rate**              | 5                    |
| **Code Quality**            | 5                    |
| **Discrepancy from Prompt** | 5                    |
| **Notes**                   | Started in Plan Mode |
