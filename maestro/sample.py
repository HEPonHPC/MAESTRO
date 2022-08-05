from maestro import OutputLevel, Settings, ParameterPointUtil, DiskUtil
from maestro.mpi4py_ import MPI_
import numpy as np
import json, os
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.16f')
import pprint

class InterpolationSample(object):
    def __init__(self, state,parameter_file="params.dat",run_fidelity_file="run_fidelity.dat",
                 at_fidelity_file="at_fidelity.dat"):
        self.state: Settings = state
        self.debug = OutputLevel.is_debug(self.state.output_level)
        self.n_to_get = 2 * self.state.N_p
        self.parameter_file = parameter_file
        self.run_fidelity_file = run_fidelity_file
        self.at_fidelity_file = at_fidelity_file
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
        self.p_sel_iters = []
        self.p_sel_types = []
        self.p_sel_indices = []

    def add_params_to_pool(self, param_metadata, iterno, paramtype):
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
        params = self.state.mc_object.get_param_from_metadata(param_metadata)
        for pno, p in enumerate(params):
            if ParameterPointUtil.check_if_point_in_TR(p, self.state.tr_center, self.state.tr_radius):
                if not ParameterPointUtil.check_if_same_point(p, self.state.tr_center):
                    if self.p_pool is None:
                        self.p_pool = np.array([p])
                        self.p_pool_metadata["0"] = {"index":pno,"k": iterno, "ptype": paramtype}
                    else:
                        self.p_pool = np.concatenate((self.p_pool, np.array([p])))
                        self.p_pool_metadata[str(len(self.p_pool) - 1)] = {"index":pno,"k": iterno,
                                                                           "ptype": paramtype}
                    self.p_pool_at_fidelity.append(param_metadata['at fidelity'][pno])

    def add_tr_center_to_selected_params(self):
        param_metadata = self.state.get_paramerter_metadata(self.state.k,"1")
        param_specific_metadata = self.state.mc_object.get_parameter_data_from_metadata(param_metadata,0)
        self.add_param_to_selected_params(self.state.tr_center,param_specific_metadata)
        self.p_sel_iters.append(self.state.k)
        self.p_sel_types.append("1")
        self.p_sel_indices.append(0)

    def add_prev_param_to_selected_params(self, pool_index):
        param = self.p_pool[pool_index]
        metadata_index = self.p_pool_metadata[str(pool_index)]['index']
        metadata_type = self.p_pool_metadata[str(pool_index)]['ptype']
        metadata_iter = self.p_pool_metadata[str(pool_index)]['k']
        param_metadata = self.state.get_paramerter_metadata(metadata_iter,metadata_type)
        param_specific_metadata = self.state.mc_object.get_parameter_data_from_metadata(param_metadata, metadata_index)
        self.add_param_to_selected_params(param, param_specific_metadata)
        self.p_sel_iters.append(metadata_iter)
        self.p_sel_types.append(metadata_type)
        self.p_sel_indices.append(metadata_index)

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
                self.add_param_to_selected_params(pi, {"at fidelity":0})
                self.i_init[pino] = False

            if self.p_sel is not None:
                if len(self.p_sel) == self.n_to_get:
                    break

    def add_param_to_selected_params(self, param,param_metadata):
        if self.p_sel is None:
            self.p_sel = np.array([param])
        else:
            self.p_sel = np.concatenate((self.p_sel, np.array([param])))
        if self.p_sel_metadata is None:
            self.p_sel_metadata = {"at fidelity": [], "run fidelity": [],"param dir":[]}
        self.p_sel_metadata["at fidelity"].append(param_metadata['at fidelity'])

        if 'param directory' not in param_metadata:
            param_dir=None
        else:
            dir_to_remove = param_metadata['param directory']
            fname = os.path.basename(dir_to_remove)
            dname = os.path.dirname(dir_to_remove)
            param_dir = os.path.join(dname, "__" + fname)
        self.p_sel_metadata["param dir"].append(param_dir)
        diff = self.state.fidelity - param_metadata['at fidelity']
        self.p_sel_metadata["run fidelity"].append(max(self.state.min_fidelity, diff) if diff > 0 else 0)

    def remove_selected_params_from_prev_metadata_and_directory(self):
        comm = MPI_.COMM_WORLD
        rank = comm.rank
        p_sel_loc = ["{}_{}".format(i,j) for (i,j) in zip(self.p_sel_iters,self.p_sel_types)]
        u_loc, u_argindices, u_count = np.unique(p_sel_loc, return_inverse=True,return_counts=True)
        for (lno,loc) in enumerate(u_loc):
            p_sel_argindices = np.argwhere(u_argindices == lno).ravel()
            l_indicies = np.array(self.p_sel_indices)[p_sel_argindices]
            l_rev_argindices =  np.argsort(l_indicies)
            l_rev_argindices = l_rev_argindices[::-1]

            # Always save metadata file for start/iterate parameters and not for sample set parameters
            k = loc.split('_')[0]
            t = loc.split('_')[1]
            param_metadata = self.state.get_paramerter_metadata(k, t)
            for l_a_index in l_rev_argindices:
                index = l_indicies[l_a_index]
                dir_to_remove = param_metadata['param directory'][index]
                fname = os.path.basename(dir_to_remove)
                dname = os.path.dirname(dir_to_remove)
                new_dir = os.path.join(dname, "__" + fname)
                if rank == 0:
                    DiskUtil.copyanything(dir_to_remove, new_dir)
                    DiskUtil.remove_directory(dir_to_remove)
                if t == "1":
                    self.state.copy_type_in_paramerter_metadata(k, t, "__{}".format(t))
                self.state.delete_data_at_index_from_paramerter_metadata(k, t, index)


        # for iter,type,index in zip(self.p_sel_iters,self.p_sel_types,self.p_sel_indices):
        #     param_metadata = self.state.get_paramerter_metadata(iter,type)
        #     dir_to_remove = param_metadata['param directory'][index]
        #     fname = os.path.basename(dir_to_remove)
        #     dname = os.path.dirname(dir_to_remove)
        #     new_dir = os.path.join(dname, "__" + fname)
        #     if rank == 0:
        #         DiskUtil.copyanything(dir_to_remove, new_dir)
        #         DiskUtil.remove_directory(dir_to_remove)
        #     if type == "1":
        #         self.state.copy_type_in_paramerter_metadata(iter,type,"__{}".format(type))
        #
        #     self.state.delete_data_at_index_from_paramerter_metadata(iter,type,index)


    @staticmethod
    def get_lhs_samples(dim, npoints, criterion, minarr, maxarr, seed=87236):
        # TODO Move to scipy.stats.qmc
        from pyDOE import lhs
        import apprentice
        np.random.seed(seed)
        X = lhs(dim, samples=npoints, criterion=criterion)
        s = apprentice.Scaler(np.array(X, dtype=np.float64), a=minarr, b=maxarr)
        return s.scaledPoints

    def build_interpolation_points(self):
        comm = MPI_.COMM_WORLD
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
        # tr_center_param_fn = self.state.working_directory.get_log_path(
        #     "parameter_metadata_1")  # + "_k{}.json".format(k)
        # prev_np_param_fn = self.state.working_directory.get_log_path("parameter_metadata_Np")  # + "_k{}.json".format(k)
        try:
            self.add_params_to_pool(self.state.get_paramerter_metadata(-1,"Np"), -1, "Np")
        except: pass
        for i in range(self.state.k):
            # tr_center_param = tr_center_param_fn + "_k{}.json".format(i)
            self.add_params_to_pool(self.state.get_paramerter_metadata(i,"1"), i, "1")

            # prev_np_param = prev_np_param_fn + "_k{}.json".format(i)
            self.add_params_to_pool(self.state.get_paramerter_metadata(i,"Np"), i, "Np")

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
        # tr_center_param = tr_center_param_fn + "_k{}.json".format(self.state.k)
        self.add_tr_center_to_selected_params()
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

        if self.p_sel is not None:
            param_meta_data = None
            if rank == 0:
                expected_folder_name = self.state.working_directory.get_log_path("MC_RUN_Np_k{}".format(self.state.k))
                param_meta_data = self.state.mc_object.write_param(parameters=self.p_sel.tolist(),
                                                 parameter_names=self.state.param_names,
                                                 at_fidelities=self.p_sel_metadata['at fidelity'],
                                                 run_fidelities=self.p_sel_metadata['run fidelity'],
                                                 mc_run_folder=self.state.mc_run_folder_path,
                                                 expected_folder_name=expected_folder_name,
                                                 fnamep=self.parameter_file,
                                                 fnamerf=self.run_fidelity_file,
                                                 fnameaf=self.at_fidelity_file)
                for no,d_from in enumerate(self.p_sel_metadata['param dir']):
                    if d_from is not None:
                        d_to = param_meta_data['mc param directory'][no]
                        DiskUtil.copy_directory_contents(d_from,d_to,exclude=[self.parameter_file,self.at_fidelity_file,self.run_fidelity_file])
            param_meta_data = comm.bcast(param_meta_data, root=0)
            self.state.update_parameter_metadata(self.state.k, "Np", param_ds=param_meta_data)
        else:
            raise Exception("Something went horribly wrong in InterpolationSample.build_interpolation_points")


