'''
This shell script helps run the HTTP server app and request by executing the relevant python files for the project.

Author: Dheeraj Singh
Email: dheeraj.singh@rwth-aachen.de, singh97.dheeraj@gmail.com
'''

#!/bin/bash
echo 'Object pose estimation: HTTP server for object pose for input image'


python ./scripts/app.py &

sleep 2

python ./scripts/request.py 

