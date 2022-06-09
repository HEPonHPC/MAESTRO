#! /bin/bash

#sample runs:
# sh run_workflow.sh /tmp/deterministicapp/WD_workflow ../../parameter_config_backup/deterministicapp
# sh run_workflow.sh ../../../log/workflow/deterministicapp/WD_workflow ../../parameter_config_backup/deterministicapp
rm -rf $1
mkdir -p $1
mkdir -p $1/conf
mkdir -p $1/logs
cp decaf-henson.json $1/decaf-henson.json
cp /home/oyildiz/mohan/pythia/pythia8-diy-master/install/bin/latest-160522/decaf-henson_python $1/.
#cp $2 $3/. #process.dat
cp ../../maestro/optimization-task $1/optimization-task.py
cp ../../maestro/mc/bin/deterministicapp.py $1/.

cp $2/algoparams.json $1/conf/algoparams.json
cp $2/config_workflow.json $1/conf/config.json

#rm -r logs/*
cd $1 || exit
mpirun -np 4 ./decaf-henson_python
