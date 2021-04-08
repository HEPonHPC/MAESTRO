#!/bin/bash

miniappParam="Parameters/Miniapp"
processfile="process.dat"
WDdir="WD/Miniapp"

bebop=""
nprocs=8
if [ "$1" = "-b" ]; then
  bebop="-b"
  nprocs=4
fi

rm -r WD;
mkdir -p $WDdir
mkdir -p $WDdir/conf
mkdir -p $WDdir/logs

cp decaf-henson-miniapp.json $WDdir/decaf-henson.json
cp decaf-henson_python $WDdir/.
cp $processfile $WDdir/. #process.dat
cp *.py $WDdir/. #maybe give full path in JSON

cp $miniappParam/algoparams_bk.json $WDdir/conf/algoparams_bk.json
cp $miniappParam/data.json $WDdir/conf/data.json
cp $miniappParam/weights $WDdir/conf/weights

#sh cleanNrun_miniapp.sh Parameters/Miniapp process.dat WD/Miniapp;
cd WD/Miniapp ||exit ;
python orchestrator.py   -a conf/algoparams_bk.json;
status=0;
until [ $status -ne 0 ]
do
  python mc_miniapp.py -e conf/data.json -c process.dat -o initial $bebop
  python buildInterpol.py -c process.dat
  mpirun -np $nprocs python mc_miniapp.py -e conf/data.json -c process.dat -o multi $bebop
  python approx.py -e conf/data.json -w conf/weights
  python chi2.py -e conf/data.json -w conf/weights -c process.dat
  python mc_miniapp.py -e conf/data.json -c process.dat -o single $bebop
  python newBox.py -w conf/weights -e conf/data.json
  status=$?
  python orchestrator.py  -c -a conf/algoparams_bk.json
done
cd ../..

