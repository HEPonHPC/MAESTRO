#! /bin/bash

#sample runs:
# sh run_workflow.sh /tmp/a14app/WD_workflow ../../parameter_config_backup/a14app
# sh run_workflow.sh ../../../log/workflow/a14app/WD_workflow ../../parameter_config_backup/a14app
rm -rf $1
mkdir -p $1
mkdir -p $1/conf
mkdir -p $1/logs
cp decaf-henson.json $1/decaf-henson.json
cp /home/oyildiz/mohan/pythia/pythia8-diy-master/install/bin/latest-160522/decaf-henson_python $1/.
cp $2/*.cmnd $1/conf/.
cp $2/data.json $1/conf/.
cp $2/weights $1/conf/.
cp ../../maestro/optimizationtask $1/optimizationtask.py
#cp ../../maestro/mc/bin/miniapp.py $1/.

cp $2/algoparams.json $1/conf/algoparams.json
cp $2/config_workflow.json $1/conf/config.json

#rm -r logs/*
cd $1 || exit
mpirun -np 4 ./decaf-henson_python
