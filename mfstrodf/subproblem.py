from mfstrodf import Settings, OutputLevel, DiskUtil
import numpy as np
import sys
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.16f')
import pprint


class TRSubproblem(object):
    def __init__(self, state):
        self.state: Settings = state
        self.debug = OutputLevel.is_debug(self.state.output_level)

    @staticmethod
    def projection(X, MIN, MAX):
        """
        Project gradient onto a box
    
        :param X: graient to be projected
        :param MIN: minimum bounds of the box
        :param MAX: maximum bounds of the box
        :type X: list
        :type MIN: list
        :type MAX: list
        :return: projected gradient
        :rtype: list
    
        """
        return np.array([
            min(max(x, mi), ma) for x, mi, ma in zip(X, MIN, MAX)
        ])

    def check_close_to_optimal_conditions(self):
        try:
            sp_object = self.state.subproblem_function_handle(self,
                                                              parameter=self.state.tr_center_scaled,
                                                              use_scaled=True)  # calls self.appr_tuning_objective
            grad = sp_object.gradient(self.state.tr_center_scaled)
            min_param_bounds = self.state.min_parameter_bounds_scaled
            max_param_bounds = self.state.max_parameter_bounds_scaled
            proj_grad = TRSubproblem.projection(self.state.tr_center_scaled - grad, min_param_bounds,
                                                max_param_bounds) - self.state.tr_center_scaled
            proj_grad_norm = np.linalg.norm(proj_grad)
            if proj_grad_norm <= self.state.min_gradient_norm and self.state.fidelity >= self.state.max_fidelity:
                self.state.algorithm_status.update_status(1)
            if proj_grad_norm <= self.state.tr_mu * self.state.tr_radius:
                self.state.update_close_to_min_condition(True)
            else:
                self.state.update_close_to_min_condition(False)
            if self.debug: print(
                "||pgrad|| \t= %.3f <=> %.3f" % (proj_grad_norm, self.state.tr_mu * self.state.tr_radius))
            self.state.update_proj_grad_norm(proj_grad_norm)
        except:
            self.state.update_close_to_min_condition(False)
            self.state.update_proj_grad_norm(1.0)
            pass
        sys.stdout.flush()

    def solve_tr_subproblem(self, tr_subproblem_result_file):
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            sp_object = self.state.subproblem_function_handle(self,use_scaled=False)  # calls self.appr_tuning_objective
            optimization_parameters = self.state.optimization_parameters
            optimization_parameters['comm'] = comm
            # optimization_parameters
            # "nstart":5,"nrestart":10,"comm"=comm,"saddle_point_check"=false,"minimize"=true,"use_mpi"=1
            result = sp_object.minimize(**optimization_parameters)
            outputdata = {
                "x": result['x'].tolist(),
                "fun": result['fun'],
            }
            if rank == 0:
                with open(tr_subproblem_result_file, 'w') as f:
                    json.dump(outputdata, f, indent=4)
                if self.debug:
                    sys.stdout.flush()
                    print("\n\\SP amin \t= {}".format(["%.3f" % (c) for c in outputdata['x']]))
                    print("\\SP min \t= {}".format(outputdata['fun']))
            sys.stdout.flush()
        except:
            self.state.algorithm_status.update_status(8, "Something went wrong when trying to solve the TR subproblem. "
                                                     "Quitting now")
            pass

    def appr_tuning_objective(self, parameter=None, use_scaled=False):
        # Make data compatible to work with apprentice
        expdatafile = self.state.subproblem_parameters['data']
        wtkeys = []
        with open(expdatafile, 'r') as f:
            exp_ds = json.load(f)
        new_ds = {}
        for key in exp_ds:
            if '#' not in key:
                new_ds["{}#1".format(key)] = exp_ds[key]
                wtkeys.append(key)
            else:
                new_ds[key] = exp_ds[key]
                prefix = key.split('#')[0]
                if prefix not in wtkeys:
                    wtkeys.append(prefix)

        my_expdatafile = self.state.working_directory.get_conf_path("data.json")
        with open(my_expdatafile, 'w') as f:
            json.dump(new_ds, f, indent=4)
        self.state.update_subproblem_parameters('data', my_expdatafile)

        my_wtfile = self.state.working_directory.get_conf_path("weights")
        if 'weights' in self.state.subproblem_parameters:
            wtfile = self.state.subproblem_parameters['weights']
            if wtfile != my_wtfile:
                DiskUtil.copyanything(wtfile, my_wtfile)
        else:
            wtstr = ""
            for key in wtkeys:
                wtstr += "{}\t1\n".format(key)
            with open(my_wtfile, "w") as ff:
                ff.write(wtstr)
        self.state.update_subproblem_parameters('weights', my_wtfile)

        m_type = 'model_scaled' if use_scaled else 'model'
        valscaledoutfile = self.state.subproblem_parameters[m_type][self.state.data_names[0]]
        new_ds = {}
        with open(valscaledoutfile, 'r') as f:
            ds = json.load(f)
        for key in ds:
            if '#' not in key:
                new_ds["{}#1".format(key)] = ds[key]
            else:
                new_ds[key] = ds[key]
        with open(valscaledoutfile, 'w') as f:
            json.dump(new_ds, f, indent=4)

        if len(self.state.data_names)>1 and \
            self.state.subproblem_parameters[m_type][self.state.data_names[1]] is not None:
            errscaledoutfile = self.state.subproblem_parameters[m_type][self.state.data_names[1]]
            with open(errscaledoutfile, 'r') as f:
                ds = json.load(f)
            for key in ds:
                if '#' not in key:
                    new_ds["{}#1".format(key)] = ds[key]
                else:
                    new_ds[key] = ds[key]
            with open(errscaledoutfile, 'w') as f:
                json.dump(new_ds, f, indent=4)
        else:
            errscaledoutfile = None

        import apprentice
        IO = apprentice.appset.TuningObjective2(my_wtfile,
                                                my_expdatafile,
                                                valscaledoutfile,
                                                errscaledoutfile,
                                                debug=self.debug)
        if parameter is not None:
            IO._AS.setRecurrence(parameter)
            if errscaledoutfile is not None: IO._EAS.setRecurrence(parameter)
        return IO

    def appr_tuning_objective_without_error_vals(self,parameter=None,use_scaled=False):
        if len(self.state.data_names) > 1:
            m_type = 'model_scaled' if use_scaled else 'model'
            self.state.subproblem_parameters[m_type][self.state.data_names[1]] = None
        return self.appr_tuning_objective(parameter,use_scaled)


class MCSubproblem(object):
    def __init__(self, state, meta_data_file, mc_run_folder):
        self.state: Settings = state
        with open(meta_data_file, 'r') as f:
            ds = json.load(f)
        self.parameter = ds['parameters'][0]
        self.mc_run_folder = mc_run_folder
        (self.mc_data_df, self.additional_data) = self.state.mc_object.convert_mc_output_to_df(self.mc_run_folder)
        rownames = list(self.state.data_names)
        columnnames = list(self.mc_data_df.index)
        if len(rownames)>1 and ('.P' not in rownames[0] and '.V' not in rownames[1]) and \
                len(columnnames)>1 and ('.P' not in columnnames[0] and '.V' not in columnnames[1]):
            raise Exception('The MC data frame does not have a parameter index that ends in \".P\" '
                            'and value index that ends in \".V\"')
        if len(rownames)>1 and ('.P' in rownames[0] and '.V' in rownames[1]):
            self.mc_data_df = self.mc_data_df.transpose()

        """
        Code works for 
        >inp = {'MC':{'bin1.P':[1,2,3],'bin1.V':[19,18,17],'bin2.P':[1,2,3],'bin2.V':[29,28,27]},
                'DMC':{'bin1.P':[4,5,6],'bin1.V':[99,98,97],'bin2.P':[4,5,6],'bin2.V':[89,88,87]}}
        >df = pd.DataFrame(inp)
        >df
                        MC           DMC    
        bin1.P        [1, 2, 3]     [4, 5, 6]    
        bin1.V        [19, 18, 17]  [99, 98, 97]
        bin2.P        [1, 2, 3]     [4, 5, 6]    
        bin2.V        [29, 28, 27]  [89, 88, 87]    
        
        or its transpose
        MC,DMC are data_names. bin1.P,bin1,V,bin2.P,bin2.V are column names 
        
        additional_data can be None and even if it exists may not contain the required keys. always check
        1. whether additional_data is none 
        2. if not, then whether the required key exists in additional data 
        { MC:{xmin:[],xmax:[]}, DMC:{xmin:[],xmax:[]}}
        
        """

    def objective(self):
        return self.state.mc_objective_function_handle(self)  # calls self.appr_tuning_objective

    def appr_tuning_objective(self):
        columnnames = list(self.mc_data_df.index)
        tr_subproblem = TRSubproblem(self.state)
        tr_sp_object = self.state.subproblem_function_handle(tr_subproblem,
                                                             use_scaled=False)  # calls TRSubproblem.appr_tuning_objective
        obj_val = 0.
        for cnum in range(0, len(columnnames),2):
            _Y = self.mc_data_df[self.state.data_names[0]]['{}'.format(columnnames[cnum+1])]
            _E = self.mc_data_df[self.state.data_names[1]]['{}'.format(columnnames[cnum+1])]
            term_name = columnnames[cnum].split('.')[0]
            if '#' not in term_name:
                term_name += "#1"
            if term_name in tr_sp_object._binids and len(_Y) > 0:
                ionum = tr_sp_object._binids.index(term_name)
                obj_val += tr_sp_object._W2[ionum] * (
                            (_Y[0] - tr_sp_object._Y[ionum]) ** 2 / (_E[0] ** 2 + tr_sp_object._E[ionum] ** 2))
            else:
                continue
        return obj_val

    def appr_tuning_objective_without_error_vals(self):
        columnnames = list(self.mc_data_df.index)
        tr_subproblem = TRSubproblem(self.state)
        tr_sp_object = self.state.subproblem_function_handle(tr_subproblem,
                                                             use_scaled=False)  # calls TRSubproblem.appr_tuning_objective
        obj_val = 0.
        for cnum in range(0, len(columnnames),2):
            _Y = self.mc_data_df[self.state.data_names[0]]['{}'.format(columnnames[cnum+1])]
            term_name = columnnames[cnum].split('.')[0]
            if '#' not in term_name:
                term_name += "#1"
            if term_name in tr_sp_object._binids and len(_Y) > 0:
                ionum = tr_sp_object._binids.index(term_name)
                obj_val += tr_sp_object._W2[ionum] * (
                        (_Y[0] - tr_sp_object._Y[ionum]) ** 2 / (tr_sp_object._E[ionum] ** 2))
            else:
                continue
        return obj_val
