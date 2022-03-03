import sys

from mfstrodf.mc import MCTask,DiskUtil
import numpy as np
import pprint,os
from mfstrodf.mpi4py_ import MPI_
from subprocess import Popen, PIPE
import apprentice
import math

class MiniApp(MCTask):
    @staticmethod
    def __chunk_fidelity(run_at_fidelity,min_fidelity=50):
        comm = MPI_.COMM_WORLD
        size = comm.Get_size()

        split_fidelity = np.ceil(run_at_fidelity/size)
        if split_fidelity >min_fidelity:
            run_fidelity_arr = [int(split_fidelity)] * size
        else:
            run_fidelity_arr = [0] * size
            fidelity_remaining = run_at_fidelity
            for rank in range(size):
                if fidelity_remaining < min_fidelity:
                    run_fidelity_arr[rank] = min_fidelity
                    break
                run_fidelity_arr[rank] = min_fidelity
                fidelity_remaining -= min_fidelity
        return run_fidelity_arr

    def __run_mc_command(self,pp,fidelity,output_loc):
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
        p = Popen(
            [self.mc_parmeters['mc_location'], str(pp[0]), str(pp[1]), str(pp[2]),
             str(fidelity), str(np.random.randint(1,9999999)), "0", "1", output_loc],
            stdin=PIPE, stdout=PIPE, stderr=PIPE)
        p.communicate(b"input data that is passed to subprocess' stdin")
        return p.returncode

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
            for filenum,files in enumerate(yoda_files):
                if filenum == len(yoda_files)-1:
                    break
                if filenum==0:
                    file1=yoda_files[filenum]
                    file2=yoda_files[filenum+1]
                else:
                    file1=outfile
                    file2=yoda_files[filenum+1]
                p = Popen(
                    [self.mc_parmeters['yodamerge_location'],'-o',outfile,file1,file2],
                    stdin=PIPE, stdout=PIPE, stderr=PIPE)
                p.communicate(b"input data that is passed to subprocess' stdin")

    def run_mc(self):
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        dirlist = self.get_param_directory_array(self.mc_run_folder) # from super class
        for dno,d in enumerate(dirlist):
            param = self.get_param_from_directory(d) # from super class
            run_fidelity = self.get_fidelity_from_directory(d) # from super class

            rank_run_fidelity = None
            if rank==0:
                min_f = self.mc_parmeters['min_fidelity'] \
                    if 'min_fidelity' in self.mc_parmeters else 50
                rank_run_fidelity = MiniApp.__chunk_fidelity(run_fidelity,min_f)
            rank_run_fidelity = comm.scatter(rank_run_fidelity,root=0)
            if rank_run_fidelity !=0:
                outfile = os.path.join(d,"out_curr_r{}.yoda".format(rank))
                self.__run_mc_command(param,rank_run_fidelity,outfile)
        comm.barrier()

    def merge_statistics_and_get_max_sigma(self):
        comm = MPI_.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        dirlist = self.get_param_directory_array(self.mc_run_folder)
        rank_dirs = None
        if rank == 0:
            rank_dirs = MPI_.chunk_it(dirlist)
        rank_dirs = comm.scatter(rank_dirs, root=0)
        rank_max_sigma = 0.
        wtfile = self.mc_parmeters['weights'] if 'weights' in self.mc_parmeters else None
        for dno,d in enumerate(rank_dirs):
            yodafiles = []
            mainfile = os.path.join(d, "out.yoda")
            if os.path.exists(mainfile):
                yodafiles.append(mainfile)
            for r in range(size):
                curr_file = os.path.join(d,"out_curr_r{}.yoda".format(r))
                if os.path.exists(curr_file):
                    yodafiles.append(curr_file)
            outfile = os.path.join(d, "out_curr.yoda")
            self.__merge_yoda_files(yodafiles,outfile)
            for i in range(len(yodafiles)):
                if i == 0 and "out.yoda" in yodafiles[i]: continue
                file = yodafiles[i]
                os.remove(file)
            DiskUtil.moveanything(outfile,mainfile)
            (DATA,BNAMES) = apprentice.io.readSingleYODAFile(d, "params.dat", wtfile)
            sigma = [_E[0] for mcnum, (_X, _Y, _E) in enumerate(DATA)]
            rank_max_sigma = max(sigma)
        all_sigma = comm.gather(rank_max_sigma,root=0)
        all_sigma = comm.bcast(all_sigma,root=0)
        return max(all_sigma)

    def check_df_structure_sanity(self,df):
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
        import pandas as pd
        main_object = {}
        comm = MPI_.COMM_WORLD
        wtfile = self.mc_parmeters['weights'] if 'weights' in self.mc_parmeters else None
        main_object['MC'] = {}
        main_object['DMC'] = {}
        DATA, binids, pnames, rankIdx, xmin, xmax = apprentice.io.readInputDataYODA(
            [all_param_directory], "params.dat",wtfile,storeAsH5=None, comm=comm)
        for num, (X, Y, E) in enumerate(DATA):
            bin = binids[num]
            main_object["MC"]["{}.P".format(bin)] = X
            main_object["MC"]["{}.V".format(bin)] = Y
            main_object["DMC"]["{}.P".format(bin)] = X
            main_object["DMC"]["{}.V".format(bin)] = E
        df = pd.DataFrame(main_object)
        df = self.check_df_structure_sanity(df)
        additional_data = {"MC":{"xmin":xmin,"xmax":xmax},"DMC":{"xmin":xmin,"xmax":xmax}}
        return (df,additional_data)


if __name__ == "__main__":
    S = MiniApp("MC_RUN",{"weights":"/Users/mkrishnamoorthy/Research/Code/workflow/parameter_config_backup/miniapp/weights"})
    directory = "/Users/mkrishnamoorthy/Research/Code/log/workflow/old/Miniapp/WD_local/logs/pythia_Np_k0"
    # directory = "/Users/mkrishnamoorthy/Research/Code/log/workflow/miniapp/WD/log/MC_RUN"
    S.convert_mc_output_to_df(directory)


