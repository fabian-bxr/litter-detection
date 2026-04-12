### Prompt: 
To monitor the full litter detection and processing pipeline, OpenTelemetry should be used to send out metrics, logs and traces.
The OTel Collector is running in a Docker container on the host machine. Connect to the OTel Collector via gRPC and set up the manual instrumentation. Preferably use decorators for spans, collect meaningful metrics such as framerate, number of frames captures, processed, processing time etc. Ask any questions before starting if further clarification is needed.


**Follow-up Prompts:**
- Opentelemetry already has a decorator for a span: @tracer.start_as_current_span("do_work") Can this be used instead of creating a new custom decorator, if yes, implement it.
- We will be adding more components to the litter detector, such as the processing pipeline that runs inference on the sent image over zenoh. The telemetry.py file is closely tailored to the camera, with the file being in the main folder it might seem that it can be used for other services, but it cant be used. Refactor the telemetry.py to be generic with component-specific metrics defined at module-level.
- Add a span to the frame capture 


**Tool used:** Claude Code (Opus 4.6)  
**Error Rate (0-5):** 5  
**Code Quality (0-5):** 4  
**Discrepancy from Prompt (0-5):** 4  
**Notes:** Started in plan mode
---