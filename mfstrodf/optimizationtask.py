
import argparse
import json
import numpy as np
import pprint
import os

import sys
from mpi4py import MPI
from mfstrodf import Settings,OutputLevel,DiskUtil,InterpolationSample,ModelConstruction,TRSubproblem,TrAmmendment
class OptimizaitionTask(object):
    def __init__(self, working_dir=None,algorithm_parameters=None,config=None,
                 parameter_file="params.dat",fidelity_file="fidelity.dat"):
        self.state = Settings(working_dir,algorithm_parameters,config)
        self.parameter_file = parameter_file
        self.fidelity_file = fidelity_file
        if self.state.mc_call_on_workflow:
            try:
                import pyhenson as h
                self.henson = h
            except:
                raise Exception("Cannot run workflow without pyhenson")

    def check_whether_to_stop(self):
        if self.state.algorithm_status.status_val != 0:
            self.state.save(to_log=True)
            print("The algorithm stopped with exit code {} because:\n".format(self.state.algorithm_status.status_val))
            print(self.state.algorithm_status.status_def)
            sys.exit(self.state.algorithm_status.status_val)

    def run(self):
        self.check_whether_to_stop()
        if self.state.next_step == "ops_start":
            self.ops_start()
        else: self.mc_caller_interpretor_resolver(meta_data_file=self.state.meta_data_file,
                                                  next_step=self.state.next_step)
        self.check_whether_to_stop()

    def ops_start(self):
        meta_data_file = self.state.working_directory.get_log_path(
            "parameter_metadata_1_k{}.json".format(self.state.k))
        self.initialize(meta_data_file)
        self.mc_caller_interpretor_resolver(meta_data_file = meta_data_file,
                                            next_step="ops_sample")
        self.check_whether_to_stop()

    def ops_sample(self):
        if self.state.k == 0:
            # Move RUN folder to appropriate folder
            oldparamdir  = self.state.working_directory.get_log_path("MC_RUN")
            newparamdir  = self.state.working_directory.get_log_path("MC_RUN_1_k{}".format(self.state.k))
            DiskUtil.copyanything(oldparamdir,newparamdir)

        meta_data_file = self.state.working_directory.get_log_path(
            "parameter_metadata_Np_k{}.json".format(self.state.k))
        sample_obj = InterpolationSample(self.state,self.parameter_file,self.fidelity_file)
        sample_obj.build_interpolation_points(meta_data_file)

        self.mc_caller_interpretor_resolver(meta_data_file = meta_data_file,
                                            next_step="ops_model")
        self.check_whether_to_stop()

    def ops_model(self):
        oldparamdir  = self.state.working_directory.get_log_path("MC_RUN")
        newparamdir  = self.state.working_directory.get_log_path("MC_RUN_Np_k{}".format(self.state.k))
        DiskUtil.copyanything(oldparamdir,newparamdir)
        model = ModelConstruction(self.state,newparamdir)
        model.consturct_models()
        self.check_whether_to_stop()
        subproblem = TRSubproblem(self.state)
        subproblem.check_close_to_optimal_conditions()
        self.check_whether_to_stop()
        if self.state.close_to_min_condition: self.ops_tr()
        tr_subproblem_result_file = self.state.working_directory.get_log_path("tr_subproblem_result_k{}.json".format(self.state.k))
        subproblem.solve_tr_subproblem(tr_subproblem_result_file)
        meta_data_file = self.state.working_directory.get_log_path(
            "parameter_metadata_1_k{}.json".format(self.state.k + 1))

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm.barrier()
        with open(tr_subproblem_result_file,'r') as f:
            ds = json.load(f)
        new_param = ds['x']
        if rank == 0:
            expected_folder_name = self.state.working_directory.get_log_path("MC_RUN_1_k{}".format(self.state.k+1))
            self.state.mc_object.write_param(parameters=[new_param],
                                             parameter_names=self.state.param_names,
                                             at_fidelities=[0],
                                             run_fidelities=[self.state.fidelity],
                                             file=meta_data_file,
                                             mc_run_folder=self.state.mc_run_folder_path,
                                             expected_folder_name=expected_folder_name,
                                             fnamep=self.parameter_file,
                                             fnamef=self.fidelity_file,
                                             **self.state.mc_parameters)
        self.mc_caller_interpretor_resolver(meta_data_file = meta_data_file,
                                            next_step="ops_tr")
        self.check_whether_to_stop()

    def ops_tr(self):
        oldparamdir  = self.state.working_directory.get_log_path("MC_RUN")
        newparamdir_kp1  = self.state.working_directory.get_log_path("MC_RUN_1_k{}".format(self.state.k+1))
        if not self.state.close_to_min_condition:
            DiskUtil.copyanything(oldparamdir,newparamdir_kp1)
        newparamdir_k = self.state.working_directory.get_log_path("MC_RUN_1_k{}".format(self.state.k))
        meta_data_file_kp1 = self.state.working_directory.get_log_path(
            "parameter_metadata_1_k{}.json".format(self.state.k + 1))
        meta_data_file_k = self.state.working_directory.get_log_path(
            "__parameter_metadata_1_k{}.json".format(self.state.k))

        tr_ammend = TrAmmendment(self.state,meta_data_file_k,meta_data_file_kp1,newparamdir_k,newparamdir_kp1)
        tr_ammend.perform_tr_update()
        tr_ammend.check_stopping_conditions()
        if self.state.algorithm_status.status_val == 0:
            self.state.save(to_log=True)
            self.state.increment_k()
            self.ops_sample()
        else:
            self.check_whether_to_stop()

    def initialize(self,meta_data_file):
        debug = OutputLevel.is_debug(self.state.output_level)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm.barrier()
        if rank==0 and debug:
            print("\n#####################################")
            print("Initially")
            print("#####################################")
            print("\Delta_1 \t= {}".format(self.state.tr_radius))
            print("N_p \t\t= {}".format(self.state.N_p))
            print("dim \t\t= {}".format(self.state.dim))
            print("P_1 \t\t= {}".format(self.state.tr_center))
            print("Min Phy bnd \t= {}".format(self.state.min_param_bounds))
            print("Min Phy bnd \t= {}".format(self.state.max_param_bounds))


        if rank == 0:
            expected_folder_name = self.state.working_directory.get_log_path("MC_RUN_1_k{}".format(self.state.k))
            self.state.mc_object.write_param(parameters=[self.state.tr_center],
                                             parameter_names=self.state.param_names,
                                             at_fidelities=[0],
                                             run_fidelities=[self.state.fidelity],
                                             file=meta_data_file,
                                             mc_run_folder=self.state.mc_run_folder_path,
                                             expected_folder_name = expected_folder_name,
                                             fnamep=self.parameter_file,
                                             fnamef=self.fidelity_file,
                                             **self.state.mc_parameters)


    def mc_caller_interpretor_resolver(self, meta_data_file, next_step:str):
        # Run MC
        if self.state.mc_call_on_workflow:
            try:
                self.henson.yield_()
            except:
                raise Exception("Workflow yield failed")
        elif self.state.mc_call_using_script_run:
            if self.state.mc_ran:
                self.state.change_mc_ran(False)
            else:
                self.state.change_mc_ran(True)
                self.state.save(meta_data_file=meta_data_file,next_step=next_step)
                #todo check if this works on script call. Otherwise change to number and read next_step from state
                sys.exit(next_step)
        elif self.state.mc_call_using_function_call:
            self.state.mc_object.run_mc()

        # Merge statistics and get max sigma
        max_sigma = self.state.mc_object.merge_statistics_and_get_max_sigma()

        # Update current fid (run_fid+at_fid) && get current fidelities
        current_fidelities = self.state.mc_object.get_updated_current_fidelities(meta_data_file)
        self.state.update_fidelity(max(current_fidelities))
        self.state.update_simulation_budget_used(sum(self.state.mc_object.get_run_fidelity_from_metadata(meta_data_file)))
        bound = self.state.kappa*(self.state.tr_radius**2)

        new_fidelities = []
        for cf in current_fidelities:
            if max_sigma is None or self.state.usefixedfidelity:
                nf = 0
            elif max_sigma > bound:
                diff_sigma = max_sigma - bound
                nf = int(np.ceil((cf/max_sigma)*diff_sigma))
                nf = max(self.state.min_fidelity,nf)
                if cf + nf > self.state.max_fidelity:
                    nf = self.state.max_fidelity - cf
            else:
                nf = 0
            new_fidelities.append(nf)
        # pprint.pprint(new_fidelities)
        self.state.mc_object.write_run_fidelity_to_metadata_and_directory(meta_data_file,new_fidelities)
        if sum(new_fidelities) >0:
            self.mc_caller_interpretor_resolver(meta_data_file,next_step)
        else:
            method_to_call = getattr(self,next_step)
            method_to_call()

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    """
    Helper class for better formatting of the script usage.
    """
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the MF-STRO-DF algorithm',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
                        help="Algorithm Parameters (JSON) location")
    parser.add_argument("-c", dest="CONFIG", type=str, default=None,
                        help="Config file (JSON) location")
    parser.add_argument("-d", dest="WORKINGDIR", type=str, default=None,
                        help="Working Directory")
    args = parser.parse_args()
    DiskUtil.remove_directory(args.WORKINGDIR)
    os.makedirs(os.path.join(args.WORKINGDIR,"log"),exist_ok=False)
    os.makedirs(os.path.join(args.WORKINGDIR,"conf"),exist_ok=False)

    opt_task = OptimizaitionTask(args.WORKINGDIR,args.ALGOPARAMS,args.CONFIG)

    # pprint.pprint(opt_task.state.algorithm_parameters_dict)
    # pprint.pprint(opt_task.state.config_dict)
    opt_task.run()
