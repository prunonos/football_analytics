#!/bin/bash
# EJECUTAMOS CADA DATASET
echo >> /home/gti/logs/scripts_log.txt
echo >> /home/gti/logs/scripts_log.txt
# EJECUCION historical_goals_date
echo 'EJECUCION NUEVA: historical_goals_date' >> /home/gti/logs/scripts_log.txt
date  >> /home/gti/logs/scripts_log.txt
echo $$ >> /home/gti/logs/scripts_log.txt
#######################
# EJECUCION CON MLP 1X5
#######################
# 1. EJECUCION historical_goals_date (sin red. dimensionalidad)
python experiments_script.py historical_goals_date 1 3 4 5 6 7 8 9 10 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 5 -s none minmax norm std maxabs -id 0
python experiments_script.py historical_goals_date 1 2 3 5 6 7 8 9 10 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 5 -s none minmax norm std maxabs -id 1
python experiments_script.py historical_goals_date 1 2 3 4 5 6 7 8 9 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 5 -s none minmax norm std maxabs -id 2
python experiments_script.py historical_goals_date -e 5 20 50 100 200 -b 8 16 32 64 128 -u 5 -s none minmax norm std maxabs -id 3
python experiments_script.py historical_goals_date 6 7 8 9 10 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 5 -s none minmax norm std maxabs -id 4
python experiments_script.py historical_goals_date 6 7 8 9 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 5 -s none minmax norm std maxabs -id 5
python experiments_script.py historical_goals_date 1 4 6 7 8 9 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 5 -s none minmax norm std maxabs -id 6
python experiments_script.py historical_goals_date 4 6 7 8 9 10-e 5 20 50 100 200 -anova 5 10  -b 8 16 32 64 128 -u 5 -s none minmax norm std maxabs -id 7
# 2. EJECUCION historical_goals_date (resto)
python experiments_script.py historical_goals_date -e 5 20 50 100 200 -anova 5 10 25 50 -b 8 16 32 64 128 -u 5 -s none minmax norm std maxabs -id 8
python experiments_script.py historical_goals_date -e 5 20 50 100 200 -pca 5 10 25 50 -b 8 16 32 64 128 -u 5 -s none minmax norm std maxabs -id 9
python experiments_script.py historical_goals_date -e 5 20 50 100 200 -varth 0.15 0.2 0.3 0.4 -b 8 16 32 64 128 -u 5 -s none minmax norm std maxabs -id 10
#######################
# EJECUCION CON MLP 1X10
#######################
# 1. EJECUCION historical_goals_date (sin red. dimensionalidad)
python experiments_script.py historical_goals_date 1 3 4 5 6 7 8 9 10 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 10 -s none minmax norm std maxabs -id 11
python experiments_script.py historical_goals_date 1 2 3 5 6 7 8 9 10 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 10 -s none minmax norm std maxabs -id 12
python experiments_script.py historical_goals_date 1 2 3 4 5 6 7 8 9 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 10 -s none minmax norm std maxabs -id 13
python experiments_script.py historical_goals_date -e 5 20 50 100 200 -b 8 16 32 64 128 -u 10 -s none minmax norm std maxabs -id 14
python experiments_script.py historical_goals_date 6 7 8 9 10 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 10 -s none minmax norm std maxabs -id 15
python experiments_script.py historical_goals_date 6 7 8 9 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 10 -s none minmax norm std maxabs -id 16
python experiments_script.py historical_goals_date 1 4 6 7 8 9 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 10 -s none minmax norm std maxabs -id 17
python experiments_script.py historical_goals_date 4 6 7 8 9 10-e 5 20 50 100 200 -anova 5 10  -b 8 16 32 64 128 -u 10 -s none minmax norm std maxabs -id 18
# 2. EJECUCION historical_goals_date (resto)
python experiments_script.py historical_goals_date -e 5 20 50 100 200 -anova 5 10 25 50 -b 8 16 32 64 128 -u 10 -s none minmax norm std maxabs -id 19
python experiments_script.py historical_goals_date -e 5 20 50 100 200 -pca 5 10 25 50 -b 8 16 32 64 128 -u 10 -s none minmax norm std maxabs -id 20
python experiments_script.py historical_goals_date -e 5 20 50 100 200 -varth 0.15 0.2 0.3 0.4 -b 8 16 32 64 128 -u 10 -s none minmax norm std maxabs -id 21
#######################
# EJECUCION CON MLP 1X5
#######################
# 1. EJECUCION historical_goals_date (sin red. dimensionalidad)
python experiments_script.py historical_goals_date 1 3 4 5 6 7 8 9 10 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 10 5 5 -s none minmax norm std maxabs -id 22
python experiments_script.py historical_goals_date 1 2 3 5 6 7 8 9 10 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 10 5 -s none minmax norm std maxabs -id 23
python experiments_script.py historical_goals_date 1 2 3 4 5 6 7 8 9 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 10 5 -s none minmax norm std maxabs -id 24
python experiments_script.py historical_goals_date -e 5 20 50 100 200 -b 8 16 32 64 128 -u 10 5 -s none minmax norm std maxabs -id 25
python experiments_script.py historical_goals_date 6 7 8 9 10 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 10 5 -s none minmax norm std maxabs -id 26
python experiments_script.py historical_goals_date 6 7 8 9 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 10 5 -s none minmax norm std maxabs -id 27
python experiments_script.py historical_goals_date 1 4 6 7 8 9 -e 5 20 50 100 200 -b 8 16 32 64 128 -u 10 5 -s none minmax norm std maxabs -id 28
python experiments_script.py historical_goals_date 4 6 7 8 9 10-e 5 20 50 100 200 -anova 5 10  -b 8 16 32 64 128 -u 10 5 -s none minmax norm std maxabs -id 29
# 2. EJECUCION historical_goals_date (resto)
python experiments_script.py historical_goals_date -e 5 20 50 100 200 -anova 5 10 25 50 -b 8 16 32 64 128 -u 10 5 -s none minmax norm std maxabs -id 30
python experiments_script.py historical_goals_date -e 5 20 50 100 200 -pca 5 10 25 50 -b 8 16 32 64 128 -u 10 5 -s none minmax norm std maxabs -id 31
python experiments_script.py historical_goals_date -e 5 20 50 100 200 -varth 0.15 0.2 0.3 0.4 -b 8 16 32 64 128 -u 10 5 -s none minmax norm std maxabs -id 32