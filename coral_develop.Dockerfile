FROM coral:latest

RUN useradd -ms /bin/bash cats
USER cats
WORKDIR /home/cats

RUN pip install -U tensorflow keras