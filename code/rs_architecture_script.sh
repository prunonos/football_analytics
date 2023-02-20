#!/bin/bash
date
echo $$

#BASIC
python experiments_script.py wyscout -id 1 -e 250 -nn batchnorm dropout -p 0.12 -s minmax -lr 0.1 0.001 0.0001 -m 0.9 -d 0.5 1. -l 2 -b1 0. 0.99 -b2 0.8 0.99 -t rs_arch2_basic -r 150
python experiments_script.py wyscout -id 2 -e 250 -nn batchnorm dropout -p 0.12 -s minmax -lr 0.1 0.001 0.0001 -m 0.9 -d 0.5 1. -l 3 -b1 0. 0.99 -b2 0.8 0.99 -t rs_arch3_basic -r 300
python experiments_script.py wyscout -id 3 -e 250 -nn batchnorm dropout -p 0.12 -s minmax -lr 0.1 0.001 0.0001 -m 0.9 -d 0.5 1. -l 4 -b1 0. 0.99 -b2 0.8 0.99 -t rs_arch4_basic -r 150
python experiments_script.py wyscout -id 4 -e 250 -nn batchnorm dropout -p 0.12 -s minmax -lr 0.1 0.001 0.0001 -m 0.9 -d 0.5 1. -l 9 -b1 0. 0.99 -b2 0.8 0.99 -t rs_arch9_basic -r 150

# ANOVA
python experiments_script.py wyscout -id 5 -e 250 -nn batchnorm dropout -p 0.12 -anova 5 -s minmax -lr 0.1 0.001 0.0001 -m 0.9 -d 0.5 1. -l 2 -b1 0. 0.99 -b2 0.8 0.99 -t rs_arch2_anova_5 -r 150
python experiments_script.py wyscout -id 6 -e 250 -nn batchnorm dropout -p 0.12 -anova 5 -s minmax -lr 0.1 0.001 0.0001 -m 0.9 -d 0.5 1. -l 3 -b1 0. 0.99 -b2 0.8 0.99 -t rs_arch3_anova_5 -r 300
python experiments_script.py wyscout -id 7 -e 250 -nn batchnorm dropout -p 0.12 -anova 5 -s minmax -lr 0.1 0.001 0.0001 -m 0.9 -d 0.5 1. -l 4 -b1 0. 0.99 -b2 0.8 0.99 -t rs_arch4_anova_5 -r 150
python experiments_script.py wyscout -id 8 -e 250 -nn batchnorm dropout -p 0.12 -anova 5 -s minmax -lr 0.1 0.001 0.0001 -m 0.9 -d 0.5 1. -l 9 -b1 0. 0.99 -b2 0.8 0.99 -t rs_arch9_anova_5 -r 150

echo 'Finished'
