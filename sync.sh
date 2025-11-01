#!/usr/bin/env bash

rsync --exclude env --exclude .git -P -a . rangpur:$(basename "$PWD")/