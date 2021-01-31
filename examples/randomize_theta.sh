#!/bin/bash

for ((theta=1;theta<180;theta++)); do
  echo "processing theta: " + $theta;
  imagine "seed of hope"\
     --seed=872072 \
     --num_layers=32 \
     --image_width=256 --save_progress=True --save_every=10 --epochs=1 \
     --batch_size=32 --gradient_accumulate_every=1 \
     --iterations=1000
     --overwrite=True \
     --theta_initial=$theta \
     --theta_hidden=$theta \
     --save_date_time=True;
  wait; # Important, otherwise the loop will continue before finishing and you'll run out of memory.
done
