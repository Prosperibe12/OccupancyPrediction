FROM python:3.10.12

COPY . /app 

WORKDIR  /app 

RUN python -m venv venv 
RUN  source/bin/activate 
RUN  pip install -r requirements.txt 
CMD [ "executable" ]