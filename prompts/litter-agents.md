Your task is to help us create a multi agent system for litter detection using the camera from a Unitree Go2 robot. 

## What we currently already have:
- trained .onnx model running inference on a camera feed, outputting tracked segments of detected litter
- Go2 control system that publishes it's odometry (localization/pose topic) and accepts NavigationRequests (straight lines) (You can check it on my drive: ~/PyCharmProjects/robodog-digipro)
- Zenoh Router

## What we have planned:
- User gives a prompt: "Search 10m around me for litter"
- One agent should be started that reads the detections from our detection system in src/litter_detector/detector & src/litter_detector/tracker
  - This agent uses an Ollama Cloud Model (for example Gemma4) to verify a cropped image of the detected trash and classify it's type into categories
  - If the trash has been validated, the position of the robot where it was detected, the image of the trash and the category should be written to a database (img may be stored locally, maybe use sqlite)
  - Not sure what the best setup for this could be, have several tools for a single agent, run several agents with this? Both the detections and the robot position are published on zenoh, no queryables.
- Another agent is started along side the detection agent, responsible for parsing the search area the user requested, What shape, is it offset from my current position etc.
  - To navigate the area properly, we have pre-mapped it and generated a static cost map from it, see as an example the my_lab_grid.png file.
  - Not sure if it's needed to use this map in this step. This map should later be fetched either over Zenoh or a RestAPI, not sure yet, but it wont be a file on this system. Use this file for all the testing for now
- Then the main path planning loop is started. Not sure if agent should be used for the path planning or how we could integrate this, but these are our goals 
  - The main idea of our search algorithm is that we map the area that we have seen already and then calculate the next straight line to a new area that we haven't seen before, optimizing for information gain, finishing when we have seen the before generated area.
  - More specifically, the robot has a camera mounted on the front with a FoV of 70°. So when the robot walks around, we want to fill out our static map using raycasting and based on the current position of what the robot has seen of the room before. It cant see through obstacles, so the raycasting should not go through "black parts" of the static map image basically.
  - Then when the next waypoint has been reached, a new waypoint needs to be generated. The robot can only walk in straight lines, from its current position, so using some kind of raycasting algorithm again and a cost function of how much information can be gained, a new waypoint needs to be selected where the robot should move towards. Optimize this for smooth traversal of the map. See the robodog-digipro project for how to handle the waypoint execution.
  - Important is also that this system needs to detect if parts of the generated area cant be accessed, they might be marked as free on the map but might not be reachable. Plan for this as well. If most of the accessible area of the map has been marked as seen, this process is finished.
- Check the litter-agents-sketch.png for an ugly sketch of our idea

## Additional information
Use Pydantic-AI for all agent related code


Based on these information, research into these ideas and create a plan first before working on the code for our proposed architecture. If parts need to be changed or reworked, let us know. Ask if there are open questions.



| Metric                              | Score                             |
|-------------------------------------|-----------------------------------|
| **Tool used**                       | Claude Code, Fable 5 Extra Effort |
| **Error Rate (0 - 4)**              |                                   |
| **Code Quality (0 - 4)**            |                                   |
| **Discrepancy from Prompt (0 - 4)** |                                   |
| **Notes**                           |                                   |