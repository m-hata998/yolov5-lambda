ARG FUNCTION_DIR="/home/app/"

FROM python:3.8-slim-buster
RUN apt-get update
# autoconf automake libexecinfo-dev make cmake libcurl 
# RUN apt-get install libgl1-mesa-dev
ENV DEBCONF_NOWARNINGS yes
# RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt-get install -y libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6
# RUN apt-get clean


ARG FUNCTION_DIR
RUN mkdir -p ${FUNCTION_DIR}
COPY yolov5/* ${FUNCTION_DIR}
COPY yolov5/models/ ${FUNCTION_DIR}/models/
# COPY yolov5/models/hub/ ${FUNCTION_DIR}/models/hub/
COPY yolov5/utils/ ${FUNCTION_DIR}/utils/
COPY yolov5/utils/google_app_engine/ ${FUNCTION_DIR}/utils/google_app_engine/
COPY yolov5/utils/wandb_logging/ ${FUNCTION_DIR}/utils/wandb_logging/


RUN which python3
RUN pip3 install awslambdaric --target ${FUNCTION_DIR}
RUN pip3 install -r ${FUNCTION_DIR}requirements.txt --target ${FUNCTION_DIR}
ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/bin/aws-lambda-rie
RUN chmod 755 /usr/bin/aws-lambda-rie
WORKDIR ${FUNCTION_DIR}
COPY entry.sh /
RUN chmod 755 /entry.sh
RUN ls -l ${FUNCTION_DIR}
ENTRYPOINT [ "/entry.sh"  ]
CMD [ "app.handler" ]