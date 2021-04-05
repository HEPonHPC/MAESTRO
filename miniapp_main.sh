#!/bin/bash

rm -r WD;
sh cleanNrun.sh Parameters/Miniapp process.dat WD/Miniapp;
cd WD/Miniapp ||exit ;
python orchestrator.py  -v -a conf/algoparams_bk.json;
status=0;
until [ $status -ne 0 ]
do
  python mc_miniapp.py -e conf/data.json -c process.dat -o initial
  python buildInterpol.py -c process.dat
  mpirun -np 5 python mc_miniapp.py -e conf/data.json -c process.dat -o multi
  python approx.py -e conf/data.json -w conf/weights
  python chi2.py -e conf/data.json -w conf/weights -c process.dat
  python mc_miniapp.py -e conf/data.json -c process.dat -o single
  python newBox.py -w conf/weights -e conf/data.json
  status=$?
  python orchestrator.py  -v -c -a conf/algoparams_bk.json
done
cd ../..

