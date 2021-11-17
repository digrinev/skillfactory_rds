FROM python:3.8

WORKDIR /app

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

RUN pip install git+https://github.com/elliottzheng/face-detection.git@master

COPY modules/ /app/modules
COPY models/ /app/models

COPY ./backend.py /app/backend.py

EXPOSE 8000
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
