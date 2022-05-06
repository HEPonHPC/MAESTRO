from mfstrodf.mc import MCTask,DiskUtil
import numpy as np
import pprint,os
from mfstrodf.mpi4py_ import MPI_
from subprocess import Popen, PIPE
import apprentice

class A14App(MCTask):
    def run_mc(self):
        raise Exception("A14 MC cannot be run using a function call")

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

    def __check_and_resolve_nan_inf(self,data, binids,all_param_directory):
        def interpolate_nan_inf(data_array):
            resolved = True
            if np.isnan(data_array).any() or np.isinf(data_array).any():
                nan_inf = [i or j for (i,j) in zip(np.isnan(data_array), np.isinf(data_array))]
                # Do (any) middle
                not_ni_left = None
                ni_count = 0
                for ninum, ni in enumerate(nan_inf):
                    if not ni and ni_count == 0:
                        not_ni_left = data_array[ninum]
                    elif not ni and ni_count > 0:
                        if not_ni_left is not None:
                            increment = (data_array[ninum]-not_ni_left)/(ni_count+1)
                            for i in range(ninum-ni_count,ninum):
                                data_array[i] = data_array[i-1]+increment
                            ni_count = 0
                            not_ni_left = data_array[ninum]
                        else:
                            not_ni_left = data_array[ninum]
                            ni_count = 0
                    else: ni_count+=1

                # Do (any) left
                nan_inf = [i or j for (i,j) in zip(np.isnan(data_array), np.isinf(data_array))]
                ni_count = 0
                for ninum, ni in enumerate(nan_inf):
                    if ni: ni_count+=1
                    elif ni_count>0:
                        if ninum < len(nan_inf)-1 and not nan_inf[ninum] and not nan_inf[ninum+1]:
                            difference = data_array[ninum+1] - data_array[ninum]
                            for i in range(ninum-1,-1,-1):
                                data_array[i] = max(data_array[i+1] - difference,0.0)
                        else:resolved = False
                        break

                # Do (any) right
                nan_inf = [i or j for (i,j) in zip(np.isnan(data_array), np.isinf(data_array))]
                ni_count = 0
                for ninum in range(len(nan_inf)-1,-1,-1):
                    ni = nan_inf[ninum]
                    if ni: ni_count+=1
                    elif ni_count>0:
                        if ninum > 0 and not nan_inf[ninum] and not nan_inf[ninum-1]:
                            difference = data_array[ninum] - data_array[ninum-1]
                            for i in range(ninum+1,len(nan_inf)):
                                data_array[i] = data_array[i-1] + difference
                        else:resolved = False
                        break

            return (data_array, resolved)

        def use_model_to_resolve_nan_inf_or_drop():
            for binid in binids:
                X_V = data['MC']["{}.P".format(binid)]
                V = data['MC']["{}.V".format(binid)]
                X_DV = data['DMC']["{}.P".format(binid)]
                DV = data['DMC']["{}.V".format(binid)]

                import json
                from mfstrodf.model import ModelConstruction
                apd_arr = os.path.basename(all_param_directory).split('_')
                for (X,data_array,mc_prefix) in zip([X_V,X_DV],[V,DV],['MC','DMC']):
                    if np.isnan(data_array).any() or np.isinf(data_array).any():
                        nan_inf = [i or j for (i,j) in zip(np.isnan(data_array), np.isnan(data_array))]
                        #   If only one parameter
                        if apd_arr[2] != 'Np':
                            if apd_arr[3] == 'k0':
                                raise Exception("Nan and Inf found at the start point. Algorithm cannot "
                                                "continue. Please select a new start point in \"tr_center\" and "
                                                "try again")
                            # If k > 0: read model of the corresponding bin and evaluate model at x where y is nan/inf
                            else:
                                log_dir = os.path.normpath(self.mc_run_folder + os.sep + os.pardir)
                                # read algorithm_parameters_dump.json
                                with open(os.path.join(log_dir,'algorithm_parameters_dump.json')) as f:
                                    algo_ds = json.load(f)
                                k = algo_ds['current_iteration']
                                with open(os.path.join(log_dir,'config_dump.json')) as f:
                                    config_ds = json.load(f)
                                function_str_dict = config_ds['model']['function_str']
                                with open(os.path.join(log_dir,"{}_model_k{}.json".format(mc_prefix,k))) as f:
                                    model_ds = json.load(f)
                                for ninum,ni in enumerate(nan_inf):
                                    if ni:
                                        x = X[ninum]
                                        binid = bins_in_obs[ninum]
                                        bin_model_ds = model_ds[binid]
                                        model = ModelConstruction.get_model_object(function_str_dict[mc_prefix],bin_model_ds)
                                        data_array[ninum] = model(x)
                #   If more than one parameter:
                if apd_arr[2] == 'Np':
                    if np.isnan(V).any() or np.isinf(V).any():
                        nan_inf = [i or j for (i,j) in zip(np.isnan(V), np.isinf(V))]
                        X_V = np.array(X_V)[np.invert(nan_inf)].tolist()
                        V =  np.array(V)[np.invert(nan_inf)].tolist()
                        X_DV = np.array(X_DV)[np.invert(nan_inf)].tolist()
                        DV = np.array(DV)[np.invert(nan_inf)].tolist()
                    if np.isnan(DV).any() or np.isinf(DV).any():
                        nan_inf = [i or j for (i,j) in zip(np.isnan(DV), np.isinf(DV))]
                        X_V = np.array(X_V)[np.invert(nan_inf)].tolist()
                        V =  np.array(V)[np.invert(nan_inf)].tolist()
                        X_DV = np.array(X_DV)[np.invert(nan_inf)].tolist()
                        DV = np.array(DV)[np.invert(nan_inf)].tolist()
                data['MC']["{}.P".format(binid)] =X_V
                data['MC']["{}.V".format(binid)] =V
                data['DMC']["{}.P".format(binid)] =X_DV
                data['DMC']["{}.V".format(binid)] =DV


        # Try to interpolate
        # If not possible to interpolate:
        #   If only one parameter:
        #       find out the iteration number k of the run
        #       If k == 0: raise exception
        #       If k > 0: read model of the corresponding bin and evaluate model at x where y is nan/inf
        #   If more than one parameter:
        #       delete x and y where y is a nan/inf
        observables = np.unique([b.split("#")[0]  for b in binids])
        binids = np.array(binids)
        resolved = True
        for onum, obs in enumerate(observables):
            # Find bins in the observable
            bins_in_obs = np.sort(binids[np.flatnonzero(np.core.defchararray.find(binids,obs)!=-1)])
            for pnum,param in enumerate(data['MC']["{}.P".format(bins_in_obs[0])]):
                V = [data['MC']["{}.V".format(binid)] for binid in bins_in_obs]
                DV = [data['DMC']["{}.V".format(binid)] for binid in bins_in_obs]
                # Try to interpolate
                (V,V_resolved) = interpolate_nan_inf(V)
                (DV,DV_resolved) = interpolate_nan_inf(DV)
                if V_resolved and DV_resolved:
                    for bno,binid in enumerate(bins_in_obs):
                        data['MC']["{}.V".format(binid)] = V[bno]
                        data['DMC']["{}.V".format(binid)] = DV[bno]
                # If not possible to interpolate
                else:
                    resolved = resolved and False
        if not resolved:
            use_model_to_resolve_nan_inf_or_drop()

    def merge_statistics_and_get_max_sigma(self):
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        rivett_analysis = ["qcd","z","ttbar"]
        dirlist = self.get_param_directory_array(self.mc_run_folder)
        rank_dirs = None
        if rank == 0:
            rank_dirs = MPI_.chunk_it(dirlist)
        rank_dirs = comm.scatter(rank_dirs, root=0)
        rank_max_sigma = 0.
        wtfile = self.mc_parmeters['weights'] if 'weights' in self.mc_parmeters else None
        for dno,d in enumerate(rank_dirs):
            """
            Rivet files from MC: out_rivet_qcd.yoda, out_rivet_z.yoda,out_rivet_ttbar.yoda
            merges into: out_curr_tmp.yoda
            out.yoda (if exists) and out_curr_tmp.yoda (if exists) merges into out_curr.yoda
            move out_curr.yoda to out.yoda 
            """
            outfile_tmp = os.path.join(d, "out_curr_tmp.yoda")
            rivet_file_exists = [os.path.exists(os.path.join(d,"out_rivet_{}.yoda".format(rname)))
                                 for rname in rivett_analysis]
            with open(outfile_tmp, 'w') as outfile_tmp_file_handle:
                for rno,rname in enumerate(rivett_analysis):
                    if np.all(rivet_file_exists):
                        rivet_fpath = os.path.join(d,"out_rivet_{}.yoda".format(rname))
                        with open(rivet_fpath,'r') as rivetfile_file_handle:
                            for line in rivetfile_file_handle:
                                outfile_tmp_file_handle.write(line)
                        os.remove(rivet_fpath)
                        outfile_tmp_file_handle.write("\n")
            yodafiles = []
            mainfile = os.path.join(d, "out.yoda")
            if os.path.exists(mainfile):
                yodafiles.append(mainfile)
            if os.path.exists(outfile_tmp):
                yodafiles.append(outfile_tmp)
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
        max_sigma = None
        if rank == 0:
            max_sigma = max(all_sigma)
        max_sigma = comm.bcast(max_sigma,root=0)
        return max_sigma

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
        self.__check_and_resolve_nan_inf(main_object,binids,all_param_directory)
        df = pd.DataFrame(main_object)
        df = self.check_df_structure_sanity(df)
        additional_data = {"MC":{"xmin":xmin,"xmax":xmax},"DMC":{"xmin":xmin,"xmax":xmax}}
        return (df,additional_data)

    def write_param(self, parameters, parameter_names, at_fidelities, run_fidelities,
                    mc_run_folder, expected_folder_name,
                    fnamep="params.dat", fnamerf="run_fidelity.dat",
                    fnameaf="at_fidelity.dat"):
        ds = super().write_param(parameters,parameter_names,at_fidelities,run_fidelities,mc_run_folder,
                            expected_folder_name,fnamep,fnamerf,fnameaf)
        from mfstrodf import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            dirlist = self.get_param_directory_array(mc_run_folder)
            for d in dirlist:
                for rc_path in self.mc_parmeters['run_card_paths']:
                    dst = os.path.join(d,os.path.basename(rc_path))
                    DiskUtil.copyanything(rc_path,dst)
        return ds