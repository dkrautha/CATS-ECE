build:
    docker build -t coral -f coral_base.Dockerfile .
    docker build -t coral_develop -f coral_develop.Dockerfile .
    docker build -t coral_test -f coral_test.Dockerfile .

coral_test: build
    docker run --privileged --rm -v /dev:/dev coral_test