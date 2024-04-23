#!/bin/bash

date  #>> /home/gti/logs/scripts_log.txt
echo $$

python experiments_random.py historical -id 1 -e 50 -nn dumb batchnorm dropout -a relu selu leaky_relu sigmoid -ns 0.01 0.001 0.1 -l 2 -lw 1.1 0.9 1.0 -lw 1.0 1.0 1.0 -b 64 -s minmax norm -lr 0.1 0.01 0.001 0.0001 0.00001 -o -t rs_historical_basic -r 20 & 
python experiments_random.py historical -id 1 -e 50 -nn dumb batchnorm dropout -a relu selu leaky_relu sigmoid -ns 0.01 0.001 0.1 -l 3 -lw 1.1 0.9 1.0 -lw 1.0 1.0 1.0 -b 64 -s minmax norm -lr 0.1 0.01 0.001 0.0001 0.00001 -o -t rs_historical_basic -r 20 & 
# python experiments_random.py historical -id 1 -e 50 -nn dumb batchnorm dropout -a relu selu leaky_relu sigmoid -ns 0.01 0.001 0.1 -l 4 -lw 1.1 0.9 1.0 -lw 1.0 1.0 1.0 -b 64 -s minmax norm -lr 0.1 0.01 0.001 0.0001 0.00001 -o -t rs_historical_basic -r 20 & 
# python experiments_random.py historical -id 1 -e 50 -nn dumb batchnorm dropout -a relu selu leaky_relu sigmoid -ns 0.01 0.001 0.1 -l 9 -lw 1.1 0.9 1.0 -lw 1.0 1.0 1.0 -b 64 -s minmax norm -lr 0.1 0.01 0.001 0.0001 0.00001 -o -t rs_historical_basic -r 20 & 

echo 'Finished'
