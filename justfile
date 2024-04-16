_default:
    just --list

build_cats_base:
    docker build -t cats_base -f dockerfiles/cats_base.Dockerfile .

build_coral_base: build_cats_base
    docker build -t coral_base -f dockerfiles/coral_base.Dockerfile .

build_coral_develop: build_coral_base
    docker build -t coral_develop -f dockerfiles/coral_develop.Dockerfile .

build_all: build_cats_base build_coral_base build_coral_develop

train: build_cats_base
    docker build -t train_model -f dockerfiles/train_model.Dockerfile .
    docker run --gpus all --rm -it -v $PWD/output:/training -v $PWD/scripts:/scripts train_model