#! /bin/bash

#sample runs:
#sh cleanNrun.sh Parameters/X2_2D_3bin process.dat WD/X2_2D_3bin
#sh cleanNrun.sh Parameters/SumOfDiffPowers_2D_3bin process.dat WD/SumOfDiffPowers_2D_3bin
mkdir -p $3
mkdir -p $3/conf
mkdir -p $3/logs
cp decaf-henson_miniapp.json $3/decaf-henson.json
cp decaf-henson_python $3/.
cp $2 $3/. #process.dat
cp *.py $3/. #maybe give full path in JSON

cp $1/algoparams_bk.json $3/conf/algoparams_bk.json
cp $1/data.json $3/conf/data.json
cp $1/weights $3/conf/weights
#rm -r logs/*
cd $3
mpirun -np 4 ./decaf-henson_python
