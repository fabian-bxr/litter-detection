### Prompt: 
For the litter detector, i have now built the system to publish a camera image stream over Zenoh. Now, a system for running inference on the image stream.

For this, implement the following:
 - Set up OTel Instrumentation, check the camera implementation as reference
 - Create the Zenoh Session and subscribe to the image stream
 - Load the pre-trained model and run inference on the image
    - If the inference is slower than the incoming stream, drop the last frames and always use the newest frame
    - I am using MLflow to track my experiments, is it possible to load the model from MLflow directly?
    - If the current way of saving models is not optimal, suggest a better way
 - Publish the results of the inference to the Zenoh Session
   - Publish the image that was used for inference, the mask and the image with the mask applied
 - Add OTel traces, metrics and logs where they are useful (Dropped frames, inference time etc...)

I have already created a rough draft of the detector, check the current state for context.
If any topics need further clarification, please let me know.

**Follow-Up Prompts:**
- Why is the LatestFrameSlot class needed, is it possible to use deque instead of implementing a custom queue class?   

**Tool used:** Claude Code  
**Error Rate (0-5):** 5   
**Code Quality (0-5):** 4.5   
**Discrepancy from Prompt (0-5):** 5    
**Notes:** Started in plan mode  
---