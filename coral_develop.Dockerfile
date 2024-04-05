FROM coral_base:latest

RUN useradd -ms /bin/bash cats
USER cats
WORKDIR /home/cats