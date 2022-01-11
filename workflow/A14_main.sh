#!/bin/bash

#USAGE
#ON BEBOP
# ./A14_main.sh WD -b
#NOT ON BEBOP
# ./A14_main.sh WD

WDname=$1
A14Param="Parameters/A14"
processfile1="Runcards/A14/main30_rivet.qcd.cmnd"
processfile2="Runcards/A14/main30_rivet.ttbar.cmnd"
processfile3="Runcards/A14/main30_rivet.z.cmnd"
WDdir="../log/workflow/A14/$WDname"

bebop=""
nprocs=8
if [ "$2" = "-b" ]; then
  bebop="-b"
  nprocs=36
else
  echo "Cannot run A14 main locally"
#  exit;
fi

rm -r $WDdir;
mkdir -p $WDdir
mkdir -p $WDdir/conf
mkdir -p $WDdir/logs

cp decaf-henson-A14.json $WDdir/decaf-henson.json
cp decaf-henson_python $WDdir/.
cp $processfile1 $WDdir/. #process.dat
cp $processfile2 $WDdir/. #process.dat
cp $processfile3 $WDdir/. #process.dat
cp *.py $WDdir/. #maybe give full path in JSON

cp $A14Param/algoparams_bk.json $WDdir/conf/algoparams_bk.json
cp $A14Param/data.json $WDdir/conf/data.json
cp $A14Param/weights $WDdir/conf/weights

currDir=`pwd`
cd $WDdir ||exit ;
python orchestrator.py -o 50 -a conf/algoparams_bk.json;
status=0;
until [ $status -ne 0 ]
do

  python mc_A14.py -e conf/data.json -c main30_rivet.qcd.cmnd main30_rivet.ttbar.cmnd main30_rivet.z.cmnd -o initial $bebop -n $nprocs
  python buildInterpol.py -c main30_rivet.qcd.cmnd main30_rivet.ttbar.cmnd main30_rivet.z.cmnd
  python mc_A14.py -e conf/data.json -c main30_rivet.qcd.cmnd main30_rivet.ttbar.cmnd main30_rivet.z.cmnd -o multi $bebop -n $nprocs
  mpirun -np $nprocs python approx.py -e conf/data.json -w conf/weights
  mpirun -np $nprocs python chi2.py -e conf/data.json -w conf/weights -c main30_rivet.qcd.cmnd main30_rivet.ttbar.cmnd main30_rivet.z.cmnd
  python mc_A14.py -e conf/data.json -c main30_rivet.qcd.cmnd main30_rivet.ttbar.cmnd main30_rivet.z.cmnd -o single $bebop -n $nprocs
  python newBox.py -w conf/weights -e conf/data.json
  python orchestrator.py -o 10 -c -a conf/algoparams_bk.json
  status=$?
done
cd $currDir ||exit ;

