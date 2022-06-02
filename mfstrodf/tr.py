from mfstrodf import OutputLevel,Settings, ParameterPointUtil, Fstructure
import json, sys
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.16f')
import numpy as np
import pprint
class TrAmmendment(object):
    def __init__(self,state, mc_run_folder_k, mc_run_folder_kp1):
        self.state:Settings = state
        # self.meta_data_file_k = meta_data_file_k
        # self.meta_data_file_kp1 = meta_data_file_kp1
        self.mc_run_folder_k =  mc_run_folder_k
        self.mc_run_folder_kp1 = mc_run_folder_kp1

        self.debug = OutputLevel.is_debug(self.state.output_level)
        self.one_line_output = OutputLevel.is_one_line_output(self.state.output_level)
        self.is_param_kp1 = OutputLevel.is_param_kp1(self.state.output_level)
        self.is_norm_of_step = OutputLevel.is_norm_of_step(self.state.output_level)

    def perform_tr_update(self):
        from mfstrodf.mpi4py_ import MPI_
        comm = MPI_.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        if self.state.algorithm_status.status_val == 0:
            metadata_k = self.state.get_paramerter_metadata(self.state.k,"__1")
            p_star_k = metadata_k['parameters'][0]
            f_structure = Fstructure(self.state)
            sp_object = self.state.f_structure_function_handle(f_structure) # calls Fstructure.appr_tuning_objective
            if self.debug: print("inside tr update w gradcond", self.state.close_to_min_condition)
            sys.stdout.flush()

            if not self.state.close_to_min_condition:
                metadata_kp1 = self.state.get_paramerter_metadata(self.state.k+1, "1")
                p_star_kp1 = metadata_kp1['parameters'][0]

                approx_obj_val_k = sp_object.objective(p_star_k)
                approx_obj_val_kp1 = sp_object.objective(p_star_kp1)
                (mc_data_df_k, additional_data_k) = \
                    self.state.mc_object.convert_mc_output_to_df(self.mc_run_folder_k)
                (mc_data_df_kp1, additional_data_kp1) = \
                    self.state.mc_object.convert_mc_output_to_df(self.mc_run_folder_kp1)
                mc_obj_val_k = sp_object.objective_without_surrograte_values(mc_data_df_k)
                mc_obj_val_kp1 = sp_object.objective_without_surrograte_values(mc_data_df_kp1)

                if self.debug:
                    print("chi2/ra k\t= %.4E" % (approx_obj_val_k))
                    print("chi2/ra k+1\t= %.4E" % (approx_obj_val_kp1))
                    print("chi2/mc k\t= %.4E" % (mc_obj_val_k))
                    print("chi2/mc k+1\t= %.4E" % (mc_obj_val_kp1))
                rho = (mc_obj_val_k - mc_obj_val_kp1) / (approx_obj_val_k - approx_obj_val_kp1)
                if self.debug: print("rho k\t\t= %.4E" % (rho))
                norm_of_step = ParameterPointUtil.get_infinity_norm(np.array(p_star_kp1)-np.array(self.state.tr_center))
                if norm_of_step == 0 or np.isnan(rho) or np.isinf(rho):
                    self.state.algorithm_status.update_status(10)
                if rho < self.state.tr_eta :
                    if self.debug: print("rho < eta New point rejected")
                    if ParameterPointUtil.order_of_magnitude(self.state.max_fidelity) ==
                        ParameterPointUtil.order_of_magnitude(self.state.fidelity):
                        tr_radius = min(self.state.tr_radius,norm_of_step)/2
                    else:
                        tr_radius = self.state.tr_radius/2
                    curr_p = p_star_k
                    self.state.algorithm_status.update_tr_status(tr_radius_messaage="TR radius halved",
                                                                 tr_center_messaage="TR center remains the same",
                                                                 tr_update_code="R")
                    # DiskUtil.copyanything(self.meta_data_file_k,self.meta_data_file_kp1)
                    # DiskUtil.copyanything(self.mc_run_folder_k,self.mc_run_folder_kp1)
                    metadata_kp1_new = self.state.mc_object.set_current_iterate_as_next_iterate(
                                                                current_iterate_parameter_data=metadata_k,
                                                                next_iterate_parameter_data=metadata_kp1
                                            )
                    self.state.update_parameter_metadata(self.state.k + 1, "1", metadata_kp1_new)
                    if rank==0 and self.one_line_output:
                        str = ""
                        if self.state.k %10 == 0:
                            str = "iter\tCMC  PGNorm     \Delta_k" \
                                  "     NormOfStep  S   C_RA(P_k)  C_RA(P_{k+1}) C_MC(P_k)  C_MC(P_{k+1}) N_e(apprx)    \\rho\n"
                        str += "%d\tF %.6E %.6E %.6E %s %.6E %.6E %.6E %.6E %.4E %.6E" \
                               %(self.state.k+1,self.state.proj_grad_norm,self.state.tr_radius,norm_of_step,
                                 self.state.algorithm_status.tr_update_code,
                                 approx_obj_val_k,approx_obj_val_kp1,mc_obj_val_k,mc_obj_val_kp1,self.state.fidelity,rho)
                        print(str)
                        sys.stdout.flush()
                else:
                    if self.debug: print("rho >= eta. New point accepted")
                    if ParameterPointUtil.is_close(
                            ParameterPointUtil.get_infinity_norm(
                                np.array(p_star_kp1)-np.array(self.state.tr_center)),
                            self.state.tr_radius,rel_tol=1e-03):
                        tr_radius = min(self.state.tr_radius*2,self.state.tr_max_radius)
                        trradmsg = "TR radius doubled"
                        trupdatecode = "A"
                    else:
                        trradmsg = "TR radius stays the same"
                        trupdatecode = "M"
                        tr_radius = self.state.tr_radius
                    curr_p = p_star_kp1
                    trcentermsg = "TR center moved to the SP amin"
                    self.state.algorithm_status.update_tr_status(tr_radius_messaage=trradmsg,
                                                                 tr_center_messaage=trcentermsg,
                                                                 tr_update_code=trupdatecode)
                    if rank==0 and self.one_line_output:
                        str = ""
                        if self.state.k %10 == 0:
                            str = "iter\tCMC  PGNorm     \Delta_k" \
                                  "     NormOfStep  S   C_RA(P_k)  C_RA(P_{k+1}) C_MC(P_k)  C_MC(P_{k+1}) N_e(apprx)    \\rho\n"
                        str += "%d\tF %.6E %.6E %.6E %s %.6E %.6E %.6E %.6E %.4E %.6E" \
                               %(self.state.k+1,self.state.proj_grad_norm,self.state.tr_radius,norm_of_step,
                                 self.state.algorithm_status.tr_update_code,
                                 approx_obj_val_k,approx_obj_val_kp1,mc_obj_val_k,mc_obj_val_kp1,self.state.fidelity,rho)
                        print(str)
                        sys.stdout.flush()
            else:
                if self.debug: print("gradient condition failed")
                tr_radius = self.state.tr_radius/2
                curr_p = p_star_k
                self.state.algorithm_status.update_tr_status(tr_radius_messaage="TR radius halved",
                                                             tr_center_messaage="TR center remains the same",
                                                             tr_update_code="CM/R")
                # DiskUtil.copyanything(self.meta_data_file_k,self.meta_data_file_kp1)
                # DiskUtil.copyanything(self.mc_run_folder_k,self.mc_run_folder_kp1)
                metadata_kp1_new = self.state.mc_object.set_current_iterate_as_next_iterate(current_iterate_parameter_data=metadata_k,
                                                                         next_iterate_parameter_data=None,
                                                                         next_iterate_mc_directory=self.mc_run_folder_kp1
                                                                         )
                self.state.update_parameter_metadata(self.state.k+1,"1",metadata_kp1_new)
                if rank==0 and self.one_line_output:
                    str = ""
                    if self.state.k %10 == 0:
                        str = "iter\tCMC  PGNorm     \Delta_k" \
                              "     NormOfStep  S   C_RA(P_k)  C_RA(P_{k+1}) C_MC(P_k)  C_MC(P_{k+1}) N_e(apprx)    \\rho\n"
                    norm_of_step = 0.
                    str += "%d\tT %.6E %.6E %.6E %s" \
                            %(self.state.k+1,self.state.proj_grad_norm,self.state.tr_radius,norm_of_step,
                                self.state.algorithm_status.tr_update_code)
                    print(str)
                    sys.stdout.flush()
            if self.debug: print("\Delta k+1 \t= %.4E (%s)"%(tr_radius,self.state.algorithm_status.tr_radius_messaage))
            if self.is_param_kp1:
                print("P k+1 \t\t= {} ({})".format(["%.4f"%(c) for c in curr_p],self.state.algorithm_status.tr_center_messaage))
            if self.is_norm_of_step:
                norm_of_step = ParameterPointUtil.get_infinity_norm(np.array(curr_p)-np.array(self.state.tr_center))
                print("Norm of Step \t= %.8E (%s)"%(norm_of_step,self.state.algorithm_status.tr_center_messaage))

            self.state.update_tr_center(curr_p)
            self.state.update_tr_radius(tr_radius)

    def check_stopping_conditions(self):
        if self.state.algorithm_status.status_val == 0:
            if not self.state.close_to_min_condition:
                if self.state.fidelity >= self.state.max_fidelity:
                    self.state.increment_no_iters_at_max_fidelity()
                    self.state.set_radius_at_which_max_fidelity_reached(self.state.previous_tr_radius)

            if self.state.k >= self.state.max_iterations -1:
                self.state.algorithm_status.update_status(2)
            elif self.state.simulation_budget_used >= self.state.max_simulation_budget:
                self.state.algorithm_status.update_status(3)
            elif self.state.radius_at_which_max_fidelity_reached is not None and \
                    ParameterPointUtil.order_of_magnitude(self.state.previous_tr_radius) <\
                        ParameterPointUtil.order_of_magnitude(self.state.radius_at_which_max_fidelity_reached):
                self.state.algorithm_status.update_status(5)
            elif self.state.no_iters_at_max_fidelity >= self.state.max_fidelity_iteration:
                self.state.algorithm_status.update_status(6)
            elif self.state.previous_tr_radius <= self.state.tr_min_radius:
                self.state.algorithm_status.update_status(9)
            else: self.state.algorithm_status.update_status(0)

            if self.debug: print("Status\t\t= {} : {}".format(self.state.algorithm_status.status_val,self.state.algorithm_status.status_def))