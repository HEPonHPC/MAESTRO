import os.path

from maestro import Settings, OutputLevel, DiskUtil
import numpy as np
import sys
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.16f')
import pprint


class Fstructure(object):
    """
    Construct and interface with all possible objective function structures
    """
    def __init__(self, state):
        """

        Initialize the function structure object

        :param state: algortithm state containing algorithm, configuration and
            other parameters and state information
        :type state: maestro.Settings

        """
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
        """

        Helper function to check whether the current trust region center
        is close to optimal. If projected gradient is small enough at maximum fidelity
        or fixed fidelity is in used, then we claim that we are close to minmum without the
        ability to improve.  Otherwise, if projected gradient is small enough, then the
        trust region radius is halved to try to certify the minimum since we are dealing
        with a noisy function

        """
        try:
            sp_object = self.state.f_structure_function_handle(self,
                                                              parameter=self.state.tr_center_scaled,
                                                              use_scaled=True)  # calls self.appr_tuning_objective
            grad = sp_object.gradient(self.state.tr_center_scaled)
            min_param_bounds = self.state.min_parameter_bounds_scaled
            max_param_bounds = self.state.max_parameter_bounds_scaled
            proj_grad = Fstructure.projection(self.state.tr_center_scaled - grad, min_param_bounds,
                                                max_param_bounds) - self.state.tr_center_scaled
            proj_grad_norm = np.linalg.norm(proj_grad)
            if proj_grad_norm <= self.state.min_gradient_norm and \
                    (self.state.usefixedfidelity or self.state.fidelity >= self.state.max_fidelity):
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

    def solve_f_structure_subproblem(self, f_structure_subproblem_result_file):
        """

        Solve the subproblem within the trust region by calling minimize on the function structure function handle
        referrenced in the configuration input:f_structure:function_str

        :param f_structure_subproblem_result_file: subproblem result file path in
            which an object with minimum and argmin of the subproblem optimization
            solution is stored
        :type f_structure_subproblem_result_file: str

        """
        try:
            from maestro.mpi4py_ import MPI_
            comm = MPI_.COMM_WORLD
            rank = comm.Get_rank()
            sp_object = self.state.f_structure_function_handle(self,use_scaled=False)  # calls self.appr_tuning_objective
            optimization_parameters = self.state.optimization_parameters
            optimization_parameters['comm'] = comm
            optimization_parameters['x00'] = self.state.tr_center
            # optimization_parameters
            # "nstart":5,"nrestart":10,"comm"=comm,"saddle_point_check"=false,"minimize"=true,"use_mpi"=1
            result = sp_object.minimize(**optimization_parameters)
            outputdata = {
                "x": result['x'].tolist(),
                "fun": result['fun'],
            }
            if rank == 0:
                with open(f_structure_subproblem_result_file, 'w') as f:
                    json.dump(outputdata, f, indent=4)
                if self.debug:
                    sys.stdout.flush()
                    print("\n\\SP amin \t= {}".format(["%.16f" % (c) for c in outputdata['x']]))
                    print("\\SP min \t= {}".format(outputdata['fun']))
            sys.stdout.flush()
        except:
            self.state.algorithm_status.update_status(8, "Something went wrong when trying to solve "
                                                     "the function structure subproblem. Quitting now")
            raise

    def appr_tuning_objective(self, parameter=None, use_scaled=False, approx_iteration_minus_no=0):
        """

        Construct apprentice.appset.TuningObjective2 objective function with MC standard deviation values

        :param parameter: (optional) if parameter is provided, then the tuning objective recurrence
            is set to that parameter (default: None)
        :type parameter: list
        :param use_scaled: true is scaled parameters are to be used, false otherwise (default: False)
        :type use_scaled: bool
        :param approx_iteration_minus_no: number of previous iterations to deduct from the current
            iteration from which to use the model files
        :type approx_iteration_minus_no: int
        :return: TuningObjective2 objective function object
        :rtype: apprentice.appset.TuningObjective2

        """
        from maestro import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.rank
        (my_wtfile,my_expdatafile,valscaledoutfile,errscaledoutfile) = \
                        (None,None,None,None)

        if rank == 0:
            terms = []
            wtkeys = []
            # Make val approximation compatible to work with apprentice
            m_type = 'model_scaled' if use_scaled else 'model'
            valscaledoutfile = self.state.f_structure_parameters[m_type][self.state.data_names[0]]
            if approx_iteration_minus_no > 0: valscaledoutfile.replace('_k{}'.format(self.state.k),
                                                                       '_k{}'.format(self.state.k-approx_iteration_minus_no))
            new_ds = {}
            with open(valscaledoutfile, 'r') as f:
                ds = json.load(f)
            for key in ds:
                if '#' not in key:
                    new_ds["{}#1".format(key)] = ds[key]
                    terms.append("{}#1".format(key))
                    wtkeys.append(key)
                else:
                    new_ds[key] = ds[key]
                    terms.append(key)
                    prefix = key.split('#')[0]
                    if prefix not in wtkeys:
                        wtkeys.append(prefix)
            with open(valscaledoutfile, 'w') as f:
                json.dump(new_ds, f, indent=4)

            # If err approximation exists then make it compatible to work with apprentice
            if len(self.state.data_names)>1 and \
                    self.state.f_structure_parameters[m_type][self.state.data_names[1]] is not None:
                errscaledoutfile = self.state.f_structure_parameters[m_type][self.state.data_names[1]]
                if approx_iteration_minus_no > 0: errscaledoutfile.replace('_k{}'.format(self.state.k),
                                                                           '_k{}'.format(self.state.k-approx_iteration_minus_no))
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

            # Make data compatible to work with apprentice
            new_ds = {}
            if 'data' in self.state.f_structure_parameters:
                expdatafile = self.state.f_structure_parameters['data']
                with open(expdatafile, 'r') as f:
                    exp_ds = json.load(f)
                for key in exp_ds:
                    if '#' not in key:
                        new_ds["{}#1".format(key)] = exp_ds[key]
                    else:
                        new_ds[key] = exp_ds[key]
            else:
                for t in terms:
                    new_ds[t] = [0.,1.]

            my_expdatafile = self.state.working_directory.get_conf_path("data.json")
            with open(my_expdatafile, 'w') as f:
                json.dump(new_ds, f, indent=4)
            self.state.update_f_structure_parameters('data', my_expdatafile)

            my_wtfile = self.state.working_directory.get_conf_path("weights")
            if 'weights' in self.state.f_structure_parameters:
                wtfile = self.state.f_structure_parameters['weights']
                if not os.path.samefile(wtfile,my_wtfile):
                    DiskUtil.copyanything(wtfile, my_wtfile)
            else:
                wtstr = ""
                for key in wtkeys:
                    wtstr += "{}\t1\n".format(key)
                with open(my_wtfile, "w") as ff:
                    ff.write(wtstr)
            self.state.update_f_structure_parameters('weights', my_wtfile)

        my_wtfile = comm.bcast(my_wtfile, root=0)
        my_expdatafile = comm.bcast(my_expdatafile, root=0)
        valscaledoutfile = comm.bcast(valscaledoutfile, root=0)
        errscaledoutfile = comm.bcast(errscaledoutfile, root=0)

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

    def appr_tuning_objective_without_error_vals(self,parameter=None,use_scaled=False, approx_iteration_minus_no=0):
        """

        Construct apprentice.appset.TuningObjective2 objective function without MC standard deviation values

        :param parameter: (optional) if parameter is provided, then the tuning objective recurrence
            is set to that parameter (default: None)
        :type parameter: list
        :param use_scaled: true is scaled parameters are to be used, false otherwise (default: False)
        :type use_scaled: bool
        :param approx_iteration_minus_no: number of previous iterations to deduct from the current
            iteration from which to use the model files
        :type approx_iteration_minus_no: int
        :return: TuningObjective2 objective function object
        :rtype: apprentice.appset.TuningObjective2

        """
        if len(self.state.data_names) > 1:
            m_type = 'model_scaled' if use_scaled else 'model'
            self.state.f_structure_parameters[m_type][self.state.data_names[1]] = None
        return self.appr_tuning_objective(parameter,use_scaled,approx_iteration_minus_no)
