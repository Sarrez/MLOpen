FROM python:3.10.1-buster
#VOLUME /appvolume
RUN mkdir /app
WORKDIR /app
#ADD . /app/
#output sent to cmd line
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
COPY ./requirements.txt /app/
#Install requirements
RUN pip install --upgrade pip
RUN pip install opencv-python

#for cv2
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY . /app
#add user 
RUN adduser user
USER root
#RUN python manage.py makemigrations mlopenapp
#RUN python manage.py migrate
CMD ["npm", "run", "dev"]
#CMD ["run_tasks.sh"]
EXPOSE 8000
