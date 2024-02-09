build:
    docker build -t coral -f coral_base.Dockerfile .
    docker build -t coral_test -f coral_test.Dockerfile .

coral_test: build
    docker run --privileged --rm --mount type=bind,source=/dev,target=/dev coral_test