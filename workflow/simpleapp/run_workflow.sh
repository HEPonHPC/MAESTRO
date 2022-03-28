#! /bin/bash

#sample runs:
# sh run_workflow.sh /tmp/simpleapp/WD_workflow ../../parameter_config_backup/simpleapp
# sh run_workflow.sh ../../../log/workflow/simpleapp/WD_workflow ../../parameter_config_backup/simpleapp
rm -rf $1
mkdir -p $1
mkdir -p $1/conf
mkdir -p $1/logs
cp decaf-henson.json $1/decaf-henson.json
cp ../bin/decaf-henson_python $1/.
#cp $2 $3/. #process.dat
cp ../../mfstrodf/optimizationtask.py $1/.
cp ../../mfstrodf/mc/bin/simpleapp.py $1/.

cp $2/algoparams.json $1/conf/algoparams.json
cp $2/config_workflow.json $1/conf/config.json

#rm -r logs/*
cd $1 || exit
mpirun -np 4 ./decaf-henson_python
