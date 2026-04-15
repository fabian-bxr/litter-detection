### Prompt

My current litter-detection can use different cameras with different settings. Currently, the camera can't be selected without changing the code inside @src/litter_detector/camera/main.py
There is a system already for configuration (@config.py) but it does not yet configure which camera should be used (Webcam / Go2) and which webcam ID. All settings should be configurable with arguments. If no args are set, use webcam with auto id (id=None) should be used. Update the config and code accordingly.


| Metric                      | Score                         |
|-----------------------------|-------------------------------|
| **Tool used**               | Gemini Code Assist (2.5 Pro)  |
| **Error Rate**              | 5                             |
| **Code Quality**            | 5                             |
| **Discrepancy from Prompt** | 5                             |
| **Notes**                   | Claude was down 🥀            |