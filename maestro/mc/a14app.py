import sys

from maestro.mc import MCTask
from maestro import DiskUtil
import numpy as np
import pprint,os
from maestro.mpi4py_ import MPI_
from subprocess import Popen, PIPE
import apprentice

class A14App(MCTask):
    """
    MC task for Pythia 8 Monte Carlo event generator’s [https://pythia.org] parton shower and
    multiple parton interaction parameters to a range of data observables from ATLAS Run 1 from 2014 (A14)
    [https://cds.cern.ch/record/1966419/files/ATL-PHYS-PUB-2014-021.pdf]
    """
    rivett_analysis = {
        "qcd":["ATLAS_2011_S8924791", "ATLAS_2011_S8971293","ATLAS_2011_I919017","ATLAS_2011_S9128077","ATLAS_2012_I1125575","ATLAS_2014_I1298811","ATLAS_2012_I1094564"],
        # "qcd":["ATLAS_2011_S8924791"], #shortened
        "z":["ATLAS_2011_S9131140","ATLAS_2014_I1300647"],
        # "z":["ATLAS_2011_S9131140"], #shortened
        "ttbar":["ATLAS_2012_I1094568","ATLAS_2013_I1243871"]
    }
    @staticmethod
    def __chunk_fidelity(run_at_fidelity,min_fidelity=50):
        import warnings
        warnings.warn("RUN MC for A14 depricated")
        sys.stdout.flush()
        return None
        # comm = MPI_.COMM_WORLD
        # size = comm.Get_size()
        #
        # split_fidelity = np.ceil(run_at_fidelity/size)
        # if split_fidelity >min_fidelity:
        #     run_fidelity_arr = [int(split_fidelity)] * size
        # else:
        #     run_fidelity_arr = [0] * size
        #     fidelity_remaining = run_at_fidelity
        #     for rank in range(size):
        #         if fidelity_remaining < min_fidelity:
        #             run_fidelity_arr[rank] = min_fidelity
        #             break
        #         run_fidelity_arr[rank] = min_fidelity
        #         fidelity_remaining -= min_fidelity
        # return run_fidelity_arr

    def __run_mc_command(self,runcard,fidelity,anaylysis_name,seed,output_loc):
        """
        Run the miniapp MC command

        :param pp: parameter values
        :param fidelity: number of events to use
        :param output_loc: output location
        :type pp: list
        :type fidelity: int
        :type loc: str
        :return: return code obtained after running miniapp
        :rtype: int

        """
        import warnings
        warnings.warn("RUN MC for A14 depricated")
        sys.stdout.flush()
        return
        # runcardstr = "{}".format(runcard)
        # fidstr = "{}".format(fidelity)
        # seedstr = "{}".format(str(seed))
        # outstr = "{}".format(output_loc)
        # argarr = [self.mc_parmeters['mc_location'], "-p",runcardstr, "-n",fidstr, "-s",seedstr, "-o",outstr]
        # for ra in A14App.rivett_analysis[anaylysis_name]:
        #     argarr.append("-a")
        #     argarr.append(ra)
        # p = Popen(argarr,stdin=PIPE, stdout=PIPE, stderr=PIPE)
        # p.communicate(b"input data that is passed to subprocess' stdin")
        # return p.returncode

    def run_mc(self):
        """

        This method cannot be used with the A14 MC. See documentation on
        how to use the MC task using the ``script run``or ``workflow`` caller_type
        to run the this task for the A14 MC

        """
        raise Exception("A14 MC cannot be run using a function call")

        # import warnings
        # warnings.warn("RUN MC for A14 depricated")
        # sys.stdout.flush()
        # comm = MPI_.COMM_WORLD
        # rank = comm.Get_rank()
        # dirlist = self.get_param_directory_array(self.mc_run_folder) # from super class
        # for dno,d in enumerate(dirlist):
        #     # param = self.get_param_from_directory(d) # from super class
        #     run_fidelity = self.get_fidelity_from_directory(d) # from super class
        #     rank_run_fidelity = None
        #     if rank==0:
        #         min_f = self.mc_parmeters['min_fidelity'] \
        #             if 'min_fidelity' in self.mc_parmeters else 50
        #         rank_run_fidelity = A14App.__chunk_fidelity(run_fidelity,min_f)
        #     rank_run_fidelity = comm.scatter(rank_run_fidelity,root=0)
        #     if rank_run_fidelity !=0:
        #         for ano, anlysis_name in enumerate(A14App.rivett_analysis.keys()):
        #             runcard = os.path.join(d, "main30_rivet.{}.cmnd".format(anlysis_name))
        #             outfile = os.path.join(d,"out_{}_curr_r{}.yoda".format(anlysis_name,rank))
        #             seed = np.random.randint(1,9999999)
        #             self.__run_mc_command(runcard,rank_run_fidelity,anlysis_name,seed,outfile)
        # comm.barrier()


    def __merge_yoda_files(self,yoda_files,outfile):
        """
        Merge yoda file statistics into a single yoda file

        :param yoda_files: yoda files to merge
        :type yoda_files: list
        :param outfile: output filename
        :type outfile: str

        """
        from subprocess import Popen, PIPE
        if len(yoda_files) == 1:
            DiskUtil.copyanything(yoda_files[0], outfile)
        else:
            argarr = [self.mc_parmeters['rivetmerge_location'],'-o',outfile]
            for file in yoda_files:
                argarr.append(file)
            argarr.append("-e")
            p = Popen(
                argarr,stdin=PIPE, stdout=PIPE, stderr=PIPE)
            p.communicate(b"input data that is passed to subprocess' stdin")

    def merge_statistics_and_get_max_sigma(self):
        """

        Merge MC output statistics and find the maximum standard deviation of the
        MC output.

        :return: maximum standard deviation of the MC output
        :rtype: float

        """
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        # If pythia8-diy USES the filename from the -o option passed to it
        # rivet_filenames = [
        #     "out_rivet_qcd.yoda",
        #     "out_rivet_z.yoda",
        #     "out_rivet_ttbar.yoda"
        # ]
        # If pythia8-diy DOES NOT USE the filename from the -o option passed to it
        rivet_filenames = {
            "qcd":"main30_rivet.qcd.cmnd.yoda",
            "z":"main30_rivet.z.cmnd.yoda",
            "ttbar":"main30_rivet.ttbar.cmnd.yoda"
        }

        parameter_conf_filenames = {
            "qcd":"main30_rivet.qcd.cmnd",
            "z":"main30_rivet.z.cmnd",
            "ttbar":"main30_rivet.ttbar.cmnd"
        }
        dirlist = self.get_param_directory_array(self.mc_run_folder)
        rank_dirs = None
        if rank == 0:
            rank_dirs = MPI_.chunk_it(dirlist)
        rank_dirs = comm.scatter(rank_dirs, root=0)
        rank_max_sigma = 0.
        wtfile = self.mc_parmeters['weights'] if 'weights' in self.mc_parmeters else None
        for dno,d in enumerate(rank_dirs):
            """
            Rivet files from MC: See rivet_filenames array above
            out_{}.yoda (if exists) and corresponding file name from rivet_filenames (if exists) merges into out_{}_curr.yoda
            move out_{}_curr.yoda to out_{}.yoda 
            """
            rank_max_sigma = 0.
            for analysis_name in A14App.rivett_analysis.keys():

                yodafiles = []
                mainfile = os.path.join(d, "out_{}.yoda".format(analysis_name))
                rivet_filepath = os.path.join(d,rivet_filenames[analysis_name])
                if os.path.exists(mainfile):
                    yodafiles.append(mainfile)
                if os.path.exists(rivet_filepath):
                    yodafiles.append(rivet_filepath)
                outfile = os.path.join(d, "out_{}_curr.yoda".format(analysis_name))
                self.__merge_yoda_files(yodafiles,outfile)
                for i in range(len(yodafiles)):
                    if i == 0 and mainfile in yodafiles[i]: continue
                    file = yodafiles[i]
                    os.remove(file)
                DiskUtil.moveanything(outfile,mainfile)
                if os.path.exists(os.path.join(d,parameter_conf_filenames[analysis_name])):
                    os.remove(os.path.join(d,parameter_conf_filenames[analysis_name]))
                (DATA,BNAMES) = apprentice.io.readSingleYODAFile(d, "params.dat", wtfile)
                sigma = [_E[0] for mcnum, (_X, _Y, _E) in enumerate(DATA)]
                rank_max_sigma = max(rank_max_sigma,max(sigma))
        all_sigma = comm.gather(rank_max_sigma,root=0)
        max_sigma = None
        if rank == 0:
            max_sigma = max(all_sigma)
        max_sigma = comm.bcast(max_sigma,root=0)
        return max_sigma

    def check_df_structure_sanity(self,df):
        """

        Check the sanity of the pandas data frame created from the
        MC output

        :param df: pandas data frame created from the MC output
        :type df: pandas.DataFrame
        :return: corrected structure of the data frame
        :rtype: pandas.DataFrame

        """
        rownames = list(df.columns.values)
        columnnames = list(df.index)
        if len(rownames)>1 and ('.P' not in rownames[0] and '.V' not in rownames[1]) and \
                len(columnnames)>1 and ('.P' not in columnnames[0] and '.V' not in columnnames[1]):
            raise Exception('The MC data frame does not have a parameter index that ends in \".P\" '
                            'and value index that ends in \".V\"')
        if len(rownames)>1 and ('.P' in rownames[0] and '.V' in rownames[1]):
            df = df.transpose()
        return df

    def convert_mc_output_to_df(self, all_param_directory):
        """

        Convert CSV MC output to a pandas dataframe.

        :Example:
            Example of the returned pandas dataframe is given below.
            In the example below, there are three parameters and two terms of the objective function.
            Terms that end with ``.P`` are the parameters and those ending with ``.V`` are the values
            associated with either the ``MC``, i.e., MC sample values  or the ``DMC``, i.e., MC standard deviation
            values. You can add more rows, i.e, more sets of parameter and values  for additional terms in the objective function
            or more columns, i.e., more components of the each term of the objective that
            come from the MC simulator::

                >df
                                MC                      DMC
                Term1.P        [[1,2],[3,4],[6,3]]     [[1,2],[3,4],[6,3]]
                Term1.V        [19, 18, 17]            [99, 98, 97]
                Term2.P        [[1,2],[3,4],[6,3]]     [[1,2],[3,4],[6,3]]
                Term2.V        [29, 28, 27]            [89, 88, 87]

        :param all_param_directory: MC outout directory path
        :type all_param_directory: str
        :return: pandas dataframe formatted MC output
        :rtype: pandas.DataFrame

        """
        import pandas as pd
        main_object = {}
        wtfile = self.mc_parmeters['weights'] if 'weights' in self.mc_parmeters else None
        main_object['MC'] = {}
        main_object['DMC'] = {}
        DATA, binids, pnames, xmin, xmax = apprentice.io.read_input_data_YODA_on_all_ranks(
            [all_param_directory], "params.dat",wtfile,storeAsH5=None)
        for num, (X, Y, E) in enumerate(DATA):
            bin = binids[num]
            main_object["MC"]["{}.P".format(bin)] = X
            main_object["MC"]["{}.V".format(bin)] = Y
            main_object["DMC"]["{}.P".format(bin)] = X
            main_object["DMC"]["{}.V".format(bin)] = E
        observables = np.unique([b.split("#")[0]  for b in binids])
        self.check_and_resolve_nan_inf(main_object,all_param_directory,binids,observables)
        df = pd.DataFrame(main_object)
        df = self.check_df_structure_sanity(df)
        additional_data = {"MC":{"xmin":xmin,"xmax":xmax},"DMC":{"xmin":xmin,"xmax":xmax}}
        return (df,additional_data)

    def write_param(self, parameters, parameter_names, at_fidelities, run_fidelities,
                    mc_run_folder, expected_folder_name,
                    fnamep="params.dat", fnamerf="run_fidelity.dat",
                    fnameaf="at_fidelity.dat"):
        """

        Write parameters to parameter directory and generate parameter metadata
        Additionally, also write the pythia parameter configuration files for the three
        categories of A14 observables with all relevant and pertinent information as required by
        phytia8-diy

        :param parameters: list of parameters points
        :type parameters: list of lists
        :param parameter_names: names of the parameter dimensions
        :type parameter_names: list
        :param at_fidelities: current fidelity of the parameters
        :type at_fidelities: list
        :param run_fidelities: expected fidelity of the parameters i.e., these are the
            fidelities at which the MC should be run at the corresponding parameters
        :type run_fidelities: list
        :param mc_run_folder: MC run folder path
        :type mc_run_folder: str
        :param expected_folder_name: expected MC run folder path with the
            type of run (single or sample) and iteration number
        :type expected_folder_name: str
        :param fnamep: name of the parameter file (default: params.dat)
        :type fnamep: str
        :param fnamerf: name of the run fidelity file (default: run_fidelity.dat)
        :type fnamerf: str
        :param fnameaf: name of the at fidelity file (default: at_fidelity.dat)
        :type fnameaf: str
        :return: parameter metadata object
        :rtype: dict

        """
        ds = super().write_param(parameters,parameter_names,at_fidelities,run_fidelities,mc_run_folder,
                            expected_folder_name,fnamep,fnamerf,fnameaf)
        from maestro import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            dirlist = ds['mc param directory']
            run_fidelities = ds['run fidelity']
            parameters = ds['parameters']
            if 'seed' in self.mc_parmeters:
                random_seed =  self.mc_parmeters['seed']
            else:
                random_seed = np.random.randint(1,9999999)
            for dno, d in enumerate(dirlist):
                if run_fidelities[dno] > 0:
                    for rc_path in self.mc_parmeters['run_card_paths']:
                        dst = os.path.join(d,os.path.basename(rc_path))
                        DiskUtil.copyanything(rc_path,dst)
                        fout = open(dst, "a")
                        fout.write("\n")
                        for k, v in zip(parameter_names, parameters[dno]):
                            fout.write("{name} {val:.16e}\n".format(name=k, val=v))
                        fout.write("\n")
                        fout.write("Main:numberOfEvents = {}\n".format(run_fidelities[dno]))
                        fout.write("Random:setSeed = on\n")
                        fout.write("Random:seed = {}\n".format(random_seed))
                        fout.close()

        return ds