# Your task

> Your mission, should you choose to accept it...

## Aim

Build a litter detection system based on the proposed training history. This system should use images from the robodog and detect litter in it.

- While the operator controls the robot the dog should make some noise, if it detects litter.
- The system should be better than the proposed baseline and operate in realtime on the robodog hardware.
- The system should offer a possibility to identify and investigate possible wrong litter detections.

Reminder:

- Document the process and usage of AI during the lab task

Assumptions:

- litter can only be on the ground
- litter has a sufficient size (amount of pixel)

### Work Packages

1. Understand the provided repository and steps taken by the system. Understand the approach of automated research 
2. Identify improvement points and improve the solution (everything allowed)
3. Prepare the model for inference on the Jetson hardware using tensorRT
4. Compare the model with the alternative approach fine-tuning a yolo-model.
5. Prepare the inference system by connecting a webcam via eclipse zenoh.
6. Add an  application monitoring stack with suitable dashboards according to the introduced stack in the lecture

Optional:

1. Tune the system by adding additional perception approaches like an open word object detector to improve the systems performance.

## Deliverable

1. AI Usage: How did you use AI during this task? (Prompts, Agents, Pipelines, Tools, ...)
2. Identified improvement points and result, with of these points worked out and which did not.
3. Demo of the litter detection model with reported IoU
4. Webcam Demo

## Starting Questions for finding Improvements

Here are some starting questions to assess the current approach:

1. Is a time cap a good idea while comparing different encoder backbones?
2. Is the efficientNet_B4 too big for the amount of training data?
3. How good is the labeling of the data?

## Camera-Sensor

- Use either the image from the robodog camera or the image from the webcam

## Guardrails

- track the experiments with mlflow
- Compare the U-Net based approach with the yolo based fine tuning approach

## Zenoh Kickstart

We use zenoh as router. To start it as container use:

```bash
docker run --init -p 7447:7447/tcp -p 8000:8000/tcp eclipse/zenoh
```

Use [zenoh-hammer](https://github.com/sanri/zenoh-hammer) to show and debug the messages

Basic tutorial for zenoh:

- Getting started with zenoh: https://zenoh.io/docs/getting-started/first-app/
- Webcam demo: https://github.com/eclipse-zenoh/zenoh-demos/tree/main/computer-vision/zcam/zcam-python
