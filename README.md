# First Lab KI-Systeme

Over all task is to build a robot that can detect litter and notify its operator.

This project was build with the autoresaerch idea of Andrew Karpathy: https://github.com/karpathy/autoresearch

The overall idea is to critically look at the experiments and progress the AI made, identify improvements and integrate a further improved version into a robot setup.

Other approaches fine-tune a yolo model: e.g. see for https://github.com/jeremy-rico/litter-detection

## 1 Student Task

- [Task Description](student_task.md)
- [Context to this project](explainer.md)

## Example images not in the dataset

|No litter | Litter |
|---|---|
|![](Image2.jpeg) | ![](Image3.jpeg) |

## Autoresearch Content

> Note: There is already one good model in this repository. Thus you should be able to investigate the performance using the Analysis Notebook.

- [Analysis Notebook](analysis.ipynb)
- [Instructions](program.md)
- [Finding from previous runs](findings.md)

## Setup

Init project:

```bash
uv sync
```

Content:

- There is a [analysis.ipynb](analysis.ipynb) notebook to take a first look on the project and test the existing models.
- The project contains a mlflow project that stores the hole experiment and training history.
  Run the following command to launch the mlflow server and ui
  ```bash
  uv run mlflow ui
  ```





## Additional Content

- [Experiment Tracking](https://mlflow.org/docs/latest/ml/getting-started/deep-learning/)