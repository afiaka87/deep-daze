#!/bin/bash

imagine "seed of hope"\
   --seed=872072 \
   --num_layers=32 \
   --image_width=200 --save_progress=True --save_every=10 --epochs=1 \
   --batch_size=64 --gradient_accumulate_every=1 \
   --iterations=50 \
   --theta_initial=15 \
   --theta_hidden=15 \
   --overwrite=True \
   --save_date_time=True;
