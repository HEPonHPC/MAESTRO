#! /bin/bash

#sample runs:
# WD="../log/workflow/A14/WD_workflow"; rm -r $WD; sh cleanNrun_A14.sh Parameters/A14 Runcards/A14/main30_rivet.z.cmnd Runcards/A14/main30_rivet.ttbar.cmnd Runcards/A14/main30_rivet.qcd.cmnd $WD;
mkdir -p $5
mkdir -p $5/conf
mkdir -p $5/logs
cp decaf-henson-A14.json $5/decaf-henson.json
cp decaf-henson_python $5/.
cp $2 $5/. #process.dat
cp $3 $5/. #process.dat
cp $4 $5/. #process.dat
cp *.py $5/. #maybe give full path in JSON

cp $1/algoparams_bk.json $5/conf/algoparams_bk.json
cp $1/data.json $5/conf/data.json
cp $1/weights $5/conf/weights

cd $5
mpirun -np 5 ./decaf-henson_python
