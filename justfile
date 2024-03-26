_default:
    just --list

build_base:
    docker build -t coral -f coral_base.Dockerfile .

build_develop: build_base
    docker build -t coral_develop -f coral_develop.Dockerfile .

build_test: build_base
    docker build -t coral_test -f coral_test.Dockerfile .

build_all: build_base build_develop build_test

coral_test img_name: build_test
    cp {{img_name}} model/image.jpg
    docker run --privileged --rm -v ./model:/pycoral/model -v /dev:/dev coral_test