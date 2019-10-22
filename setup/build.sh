#!/bin/bash
set -ev
echo "hello $LBR_WEBHOOK_PUSHER" > hello.txt
cat hello.txt
echo "foo" 1>&2
pwd
env
python train.py
