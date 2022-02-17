from mfstrodf import OutputLevel, Settings, ParameterPointUtil, DiskUtil
import numpy as np
import json, os
from mpi4py import MPI
import pprint


class InterpolationSample(object):
    def __init__(self, state,parameter_file="params.dat",fidelity_file="fidelity.dat"):
        self.state: Settings = state
        self.debug = OutputLevel.is_debug(self.state.output_level)
        self.n_to_get = 2 * self.state.N_p
        self.parameter_file = parameter_file
        self.fidelity_file = fidelity_file
        ############################################################
        # Data Structures
        ############################################################

        self.min_bound = [max(self.state.tr_center[d] - self.state.tr_radius,
                              self.state.min_param_bounds[d]) for d in range(self.state.dim)]
        self.max_bound = [min(self.state.tr_center[d] + self.state.tr_radius,
                              self.state.max_param_bounds[d]) for d in range(self.state.dim)]
        self.min_dist = self.state.theta * self.state.tr_radius
        self.equivalence_dist = self.state.thetaprime * self.state.tr_radius

        self.p_init = np.array([])
        self.p_pool = None
        self.p_pool_at_fidelity = []
        self.p_pool_metadata = {}
        self.i_init = []
        self.i_pool = []
        self.p_sel = None
        self.p_sel_metadata = None
        self.p_sel_files = []
        self.p_sel_indices = []

    def add_params_to_pool(self, file, iterno, paramtype):
        """
        Add all acceptable (in current trust region and not the same parameter) to the pool.

        :param file: parameter file location
        :param iterno: iteration number
        :param paramtype: type of parameter. Use ``1`` when it comes from the ``initial``/``single`` MC type run and ``Np`` when
            the parameter comes from ``multi`` type MC run. See doc for `mc_*.py`
        :type file: str
        :type iterno: int
        :type paramtype: str
        :return: parameter pool, fidelity of parameters in the pool
        :rtype: list, list

        """
        params = self.state.mc_object.get_param_from_metadata(file)
        at_fidelity = self.state.mc_object.get_current_fidelities(file)
        for pno, p in enumerate(params):
            if ParameterPointUtil.check_if_point_in_TR(p, self.state.tr_center, self.state.tr_radius):
                if not ParameterPointUtil.check_if_same_point(p, self.state.tr_center):
                    if self.p_pool is None:
                        self.p_pool = np.array([p])
                        self.p_pool_metadata["0"] = {"file": file,
                                                     "fidelity to reuse": at_fidelity[pno],
                                                     "index": pno, "k": iterno, "ptype": paramtype}
                    else:
                        self.p_pool = np.concatenate((self.p_pool, np.array([p])))
                        self.p_pool_metadata[str(len(self.p_pool) - 1)] = {"file": file,
                                                                           "fidelity to reuse": at_fidelity[pno],
                                                                           "index": pno, "k": iterno,
                                                                           "ptype": paramtype}
                    self.p_pool_at_fidelity.append(at_fidelity[pno])

    def add_tr_center_to_selected_params(self, file):
        at_fidelity = self.state.mc_object.get_current_fidelities(file)
        self.add_param_to_selected_params(self.state.tr_center, at_fidelity[0],file=file,index=0)
        self.p_sel_files.append(file)
        self.p_sel_indices.append(0)

    def add_prev_param_to_selected_params(self, pool_index):
        param = self.p_pool[pool_index]
        at_fidelity = self.p_pool_at_fidelity[pool_index]
        file = self.p_pool_metadata[str(pool_index)]['file']
        file_index = self.p_pool_metadata[str(pool_index)]['index']
        self.add_param_to_selected_params(param, at_fidelity,file=file,index=file_index)
        self.p_sel_files.append(file)
        self.p_sel_indices.append(file_index)

    def add_new_param_to_selected_params(self):
        for pino, pi in enumerate(self.p_init):
            if not self.i_init[pino]: continue
            result = True
            if self.p_sel is not None:
                for psel in self.p_sel:
                    result = ParameterPointUtil.check_if_not_in_min_seperation_dist(pi, psel, self.min_dist)
                    if not result:
                        break

            if result:
                self.add_param_to_selected_params(pi, 0, None)
                self.i_init[pino] = False

            if self.p_sel is not None:
                if len(self.p_sel) == self.n_to_get:
                    break

    def add_param_to_selected_params(self, param, at_fidelity,file=None,index=None):
        if self.p_sel is None:
            self.p_sel = np.array([param])
        else:
            self.p_sel = np.concatenate((self.p_sel, np.array([param])))
        if self.p_sel_metadata is None:
            self.p_sel_metadata = {"at fidelity": [], "run fidelity": [],"param dir":[]}
        self.p_sel_metadata["at fidelity"].append(at_fidelity)
        if file is None:
            param_dir=None
        else:
            with open(file, 'r') as f:
                ds = json.load(f)
            value = ds['param directory']
            dir_to_remove = value[index]
            fname = os.path.basename(dir_to_remove)
            dname = os.path.dirname(dir_to_remove)
            param_dir = os.path.join(dname, "__" + fname)
        self.p_sel_metadata["param dir"].append(param_dir)
        diff = self.state.fidelity - at_fidelity
        self.p_sel_metadata["run fidelity"].append(max(self.state.min_fidelity, diff) if diff > 0 else 0)

    def remove_selected_params_from_prev_metadata_and_directory(self):
        comm = MPI.COMM_WORLD
        rank = comm.rank
        u_files, u_argindices, u_count = np.unique(self.p_sel_files, return_inverse=True,return_counts=True)
        for fno,file in enumerate(u_files):
            p_sel_argindices = np.argwhere(u_argindices == fno).ravel()
            f_indicies = np.array(self.p_sel_indices)[p_sel_argindices]
            f_rev_argindices =  np.argsort(f_indicies)
            f_rev_argindices = f_rev_argindices[::-1]

            # Always save metadata file for start/iterate parameters and not for sample set parameters
            if rank == 0:
                if '_1_' in file and '_Np_' not in file:
                    fname = os.path.basename(file)
                    dname = os.path.dirname(file)
                    new_file = os.path.join(dname, "__" + fname)
                    DiskUtil.copyanything(file, new_file)
                with open(file, 'r') as f:
                    ds = json.load(f)
                for f_a_index in f_rev_argindices:
                    index = f_indicies[f_a_index]
                    for key in ['parameters', 'at fidelity', 'run fidelity', 'mc param directory']:
                        value = ds[key]
                        del value[index]
                        ds[key] = value
                    value = ds['param directory']
                    dir_to_remove = value[index]
                    fname = os.path.basename(dir_to_remove)
                    dname = os.path.dirname(dir_to_remove)
                    new_dir = os.path.join(dname, "__" + fname)
                    DiskUtil.copyanything(dir_to_remove, new_dir)
                    DiskUtil.remove_directory(dir_to_remove)
                    del value[index]
                    ds["param directory"] = value

                with open(file, 'w') as f:
                    json.dump(ds, f, indent=4)

    @staticmethod
    def get_lhs_samples(dim, npoints, criterion, minarr, maxarr, seed=87236):
        # TODO Move to scipy.stats.qmc
        from pyDOE import lhs
        import apprentice
        np.random.seed(seed)
        X = lhs(dim, samples=npoints, criterion=criterion)
        s = apprentice.Scaler(np.array(X, dtype=np.float64), a=minarr, b=maxarr)
        return s.scaledPoints

    def build_interpolation_points(self, meta_data_file):
        comm = MPI.COMM_WORLD
        rank = comm.rank
        factor = 1
        while len(self.p_init[self.i_init]) < self.n_to_get:
            ############################################################
            # Inititalize p_init and I_init
            ############################################################
            self.p_init = InterpolationSample.get_lhs_samples(dim=self.state.dim,
                                                              npoints=factor * self.n_to_get,
                                                              criterion="maximin",
                                                              minarr=self.min_bound,
                                                              maxarr=self.max_bound)
            self.i_init = [True for i in self.p_init]

            ############################################################
            # Discard points from p_init that are in minimum seperation distance
            ############################################################
            for pcurrindex in range(1, len(self.p_init)):
                pcurr = self.p_init[pcurrindex]
                for potherindex in range(pcurrindex):
                    if not self.i_init[potherindex]: continue
                    pother = self.p_init[potherindex]
                    result = ParameterPointUtil.check_if_not_in_min_seperation_dist(pcurr, pother, self.min_dist)
                    if not result:
                        self.i_init[pcurrindex] = result
                        break
            factor += 1
        if self.debug:
            print("P INIT")
            pprint.pprint(self.p_init)
            print("###############################################")

        ############################################################
        # polulate p_pool
        ############################################################
        tr_center_param_fn = self.state.working_directory.get_log_path(
            "parameter_metadata_1")  # + "_k{}.json".format(k)
        prev_np_param_fn = self.state.working_directory.get_log_path("parameter_metadata_Np")  # + "_k{}.json".format(k)
        for i in range(self.state.k):
            tr_center_param = tr_center_param_fn + "_k{}.json".format(i)
            self.add_params_to_pool(tr_center_param, i, "1")

            prev_np_param = prev_np_param_fn + "_k{}.json".format(i)
            self.add_params_to_pool(prev_np_param, i, "Np")

        if self.p_pool is not None:
            self.i_pool = [True for i in self.p_pool]
            ############################################################
            # Discard points from p_pool that are the very similar
            ############################################################
            for pcurrindex in range(1, len(self.p_pool)):
                pcurr = self.p_pool[pcurrindex]
                for potherindex in range(pcurrindex):
                    if not self.i_pool[potherindex]: continue
                    pother = self.p_pool[potherindex]
                    result = ParameterPointUtil.check_if_same_point(pcurr, pother)
                    if result:
                        self.i_pool[pcurrindex] = False
                        break

            if self.debug:
                print("p_pool after initial discard")
                pprint.pprint(self.p_pool[self.i_pool])
                pprint.pprint(np.array(self.p_pool_at_fidelity)[self.i_pool])
                pprint.pprint(self.i_pool)
                pprint.pprint(self.p_pool_metadata)
                print("###############################################")

        ############################################################
        # Add tr_center to p_sel
        ############################################################
        tr_center_param = tr_center_param_fn + "_k{}.json".format(self.state.k)
        self.add_tr_center_to_selected_params(tr_center_param)
        if self.debug:
            print("p_sel after adding tr_center")
            pprint.pprint(self.p_sel)
            pprint.pprint(self.p_sel_metadata)
            print("###############################################")

        ############################################################
        # Find close matches from p_pool for points in p_init
        # and add to p_sel_prev
        # Sort the p_pool by descending order of p_pool_at_fidelity first
        ############################################################

        if self.p_pool is not None:
            ppf_order = np.argsort(-1 * np.array(self.p_pool_at_fidelity))[:len(self.p_pool_at_fidelity)]
            for pino, pi in enumerate(self.p_init):
                if not self.i_init[pino]: continue
                for ppno in ppf_order:
                    pp = self.p_pool[ppno]
                    if not self.i_pool[ppno]: continue
                    result = ParameterPointUtil.check_if_not_in_min_seperation_dist(pi, pp, self.equivalence_dist)
                    if not result:
                        self.add_prev_param_to_selected_params(ppno)
                        self.i_pool[ppno] = False
                        self.i_init[pino] = False

            if self.debug:
                print("p_sel after matching close matches")
                pprint.pprint(self.p_sel)
                pprint.pprint(self.p_sel_metadata)
                pprint.pprint(self.i_pool)
                pprint.pprint(self.i_init)
                print("###############################################")

        ############################################################
        # If not enough points add points not used before or not in
        # minimum seperation distance from p_pool to p_sel_prev
        # Sort the p_pool by descending order of p_pool_at_fidelity first
        ############################################################
        if self.p_pool is not None:
            ppf_order = np.argsort(-1 * np.array(self.p_pool_at_fidelity))[:len(self.p_pool_at_fidelity)]
            if self.p_sel is None or len(self.p_sel) < self.n_to_get:
                for ppno in ppf_order:
                    pp = self.p_pool[ppno]
                    if not self.i_pool[ppno]: continue
                    result = True
                    if self.p_sel is not None:
                        for psel in self.p_sel:
                            result = ParameterPointUtil.check_if_not_in_min_seperation_dist(pp, psel, self.min_dist)
                            if not result:
                                break

                    if result:
                        self.add_prev_param_to_selected_params(ppno)
                        self.i_pool[ppno] = False
            if self.debug:
                print("p_sel after adding points from p_pool")
                pprint.pprint(self.p_sel)
                pprint.pprint(self.p_sel_metadata)
                pprint.pprint(self.i_pool)
                print("###############################################")
        self.remove_selected_params_from_prev_metadata_and_directory()
        ############################################################
        # If not enough points add points not used before or not in
        # minimum seperation distance from p_init to p_sel_new
        ############################################################
        if self.p_sel is None or len(self.p_sel) < self.n_to_get:
            self.add_new_param_to_selected_params()

        if self.debug:
            print("p_sel after adding (any required) new points")
            pprint.pprint(self.p_sel)
            pprint.pprint(self.p_sel_metadata)
            pprint.pprint(self.i_init)
            print("###############################################")

        ############################################################
        # Save data and exit
        ############################################################
        self.state.update_close_to_min_condition(False)

        if rank == 0:
            if self.p_sel is not None:
                expected_folder_name = self.state.working_directory.get_log_path("MC_RUN_Np_k{}".format(self.state.k))
                self.state.mc_object.write_param(parameters=self.p_sel.tolist(),
                                                 parameter_names=self.state.param_names,
                                                 at_fidelities=self.p_sel_metadata['at fidelity'],
                                                 run_fidelities=self.p_sel_metadata['run fidelity'],
                                                 file=meta_data_file,
                                                 mc_run_folder=self.state.mc_run_folder_path,
                                                 expected_folder_name=expected_folder_name,
                                                 fnamep=self.parameter_file,
                                                 fnamef=self.fidelity_file,
                                                 **self.state.mc_parameters)
                with open(meta_data_file,'r') as f:
                    ds = json.load(f)
                for no,d_from in enumerate(self.p_sel_metadata['param dir']):
                    if d_from is not None:
                        d_to = ds['mc param directory'][no]
                        DiskUtil.copy_directory_contents(d_from,d_to,exclude=[self.parameter_file,self.fidelity_file])

            else:
                raise Exception("Something went horribly wrong in InterpolationSample.build_interpolation_points")

