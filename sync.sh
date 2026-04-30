#!/bin/bash

fswatch -r . | while read; do
  rsync -az \
    --exclude='.git/' \
    . "$ak:/home/akhlaque.ak@gmail.com/G-ThinkerCG/"
done

git add -A
git commit -am "updating"
git push