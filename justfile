_default:
    just --list

build_cats_base:
    docker build -t cats_base -f cats_base.Dockerfile .

build_coral_base: build_cats_base
    docker build -t coral_base -f coral_base.Dockerfile .

build_coral_develop: build_coral_base
    docker build -t coral_develop -f coral_develop.Dockerfile .

build_all: build_cats_base build_coral_base build_coral_develop