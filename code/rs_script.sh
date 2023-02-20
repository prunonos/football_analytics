#!/bin/bash
date
echo $$


python experiments_random.py wyscout -id 1 -e 250 -nn dropout -b 4 8 64 -u 120 50 25 -s norm -lr 1. 0.1 .0001 0.00001 -o -t m3_rs2_basic -r 600
#python experiments_random.py wyscout -id 2 -e 250 -nn dumb batchnorm dropout -anova 5 10 15 20 -b 16 32 64 128 -u 120 50 25 -s basic minmax norm std maxabs -lr 0.1 0.01 0.001 0.0001 0.00001 -o -t m3_rs_anova -r 250
#python experiments_random.py wyscout -id 3 -e 250 -nn dumb batchnorm dropout -pca 5 10 15 -b 16 32 64 128 -u 120 50 25 -s basic minmax norm std maxabs -lr 0.1 0.01 0.001 0.0001 0.00001 -o -t m3_rs_pca -r 200
#python experiments_random.py wyscout -id 4 -e 250 -nn dumb batchnorm dropout -varthr 0.1 0.2 0.25 0.3 0.4 -b 16 32 64 128 -u 120 50 25 -s basic minmax norm std maxabs -lr 0.1 0.01 0.001 0.0001 0.00001 -o -t m3_rs_varthr -r 100
#python experiments_random.py wyscout -id 5 -e 250 -nn dumb batchnorm dropout -feat -b 16 32 64 128 -u 120 50 25 -s basic minmax norm std maxabs -lr 0.1 0.01 0.001 0.0001 0.00001 -o -t m3_rs_feat -r 500

echo 'Finished'
