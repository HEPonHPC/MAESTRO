import pandas
from maestro.mpi4py_ import MPI_
import time, datetime
import sys,os,json
from maestro import OutputLevel,Settings
import pprint
class ModelConstruction(object):

    """
    Construct and interface with all possible surrogate models
    """

    def __init__(self,state,mc_run_folder = None):
        """

        Initialize the surrogate model object

        :param state: algortithm state containing algorithm, configuration and
            other parameters and state information
        :type state: maestro.Settings
        :param mc_run_folder: MC run folder path
        :type mc_run_folder: str

        """
        self.state:Settings = state
        self.mc_run_folder = mc_run_folder
        self.debug = OutputLevel.is_debug(self.state.output_level)
        if self.mc_run_folder is not None:
            (self.mc_data_df,self.additional_data) = self.state.mc_object.convert_mc_output_to_df(self.mc_run_folder)
            self.state.set_data_names(list(self.mc_data_df.columns.values))
        """
        Code works for 
        Code works for 
        >inp = {'MC':{'bin1.P':[[1,2],[3,4],[6,3]],'bin1.V':[19,18,17],'bin2.P':[[1,2],[3,4],[6,3]],'bin2.V':[29,28,27]},
                'DMC':{'bin1.P':[[1,2],[3,4],[6,3]],'bin1.V':[99,98,97],'bin2.P':[[1,2],[3,4],[6,3]],'bin2.V':[89,88,87]}}
        >df = pd.DataFrame(inp)
        >df
                        MC                      DMC    
        bin1.P        [[1,2],[3,4],[6,3]]     [[1,2],[3,4],[6,3]]    
        bin1.V        [19, 18, 17]               [99, 98, 97]
        bin2.P        [[1,2],[3,4],[6,3]]     [[1,2],[3,4],[6,3]]    
        bin2.V        [29, 28, 27]              [89, 88, 87]     
        
        or its transpose
        MC,DMC are data_names. bin1.P,bin1,V,bin2.P,bin2.V are column names 
        
        additional_data can be None and even if it exists may not contain the required keys. always check
        1. whether additional_data is none 
        2. if not, then whether the required key exists in additional data 
        { MC:{xmin:[],xmax:[]}, DMC:{xmin:[],xmax:[]}}
        
        """
    @staticmethod
    def get_model_object(function_str,data):
        """

        Get surrogate model object based on previously calculated model parameters

        :param function_str: function string for the specific data name
        :type function_str: str
        :param data: previously calculated model parameters. The data type of this
            argument should make sence to the the class that this data structure is passed to
        :type data: Any
        :return: model object
        :rtype: Any

        """
        import apprentice
        if function_str in ['appr_pa_m_construct','appr_ra_m_n_construct','appr_ra_m_1_construct','appr_appx_construct']:
            return apprentice.RationalApproximation(initDict=data)
        else:
            raise Exception("Model not implemented")

    def calculate_minimum_tr_radius_bound(self, data_name_index = 0):
        """

        Calculate minimum TR radius bound

        :param data_name_index: data name index (defaults to 0)
        :type data_name_index: int
        :return: minimum TR radius bound depending on the type of model
        :rtype: float

        """
        data_name = self.state.data_names[data_name_index]
        function_str = self.state.get_model_function_str(data_name)
        import numpy as np
        if function_str in ['appr_pa_m_construct','appr_appx_construct']:
            return np.power(self.state.min_gradient_norm,(1/self.state.model_parameters[data_name]['m']))
        elif function_str in ['appr_ra_m_n_construct','appr_ra_m_1_construct']:
            return np.power(self.state.min_gradient_norm,
                            (1/(self.state.model_parameters[data_name]['m'] + self.state.model_parameters[data_name]['n'])))
        else:
            raise Exception("Model not implemented")

    def construct_models(self):
        """

        Contruct the surrogate model. This function will read the function specified
        in the configuration input:model:function_str and call that function from within this
        class

        """
        for data_name in self.state.data_names:
            fh = self.state.get_model_function_handle(data_name)
            fh(self,data_name) if fh is not None else self.appr_pa_m_construct(data_name)

    #TODO: === Change appr_* functions after changes to apprentice
    def appr_pa_m_construct(self,data_name):
        """

        Contruct apprentice.PolynomialApproximation model for order of numerator m

        :param data_name: data/term name
        :type data_name: str
        :return: polynomial approximation surrogate model object
        :rtype: apprentice.PolynomialApproximation

        """
        self.appr_appx_construct(data_name)
    def appr_ra_m_n_construct(self,data_name):
        """

        Contruct apprentice.RationalApproximation model for order of numerator m and
        denominator n

        :param data_name: data/term name
        :type data_name: str
        :return: rational approximation surrogate model object
        :rtype: apprentice.RationalApproximation

        """
        self.appr_appx_construct(data_name)
    def appr_ra_m_1_construct(self,data_name):
        """

        Contruct apprentice.RationalApproximation model for order of numerator m and
        denominator 1. This is a special case of the the model construction function
        appr_ra_m_n_construct

        :param data_name: data/term name
        :type data_name: str
        :return: rational approximation surrogate model object
        :rtype: apprentice.RationalApproximation

        """
        self.appr_appx_construct(data_name)

    def appr_appx_construct(self,data_name):
        """

        Contruct apprentice.PolynomialApproximation or apprentice.RationalApproximation model.

        :param data_name: data/term name
        :type data_name: str
        :return: polynomail or rational approximation surrogate model object
        :rtype: apprentice.RationalApproximation or apprentice.PolynomailApproximation

        """
        t4 = time.time()
        app = {}
        appscaled = {}
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        columnnames = list(self.mc_data_df.index)

        import apprentice
        Sclocal = apprentice.Scaler(self.mc_data_df[data_name]['{}'.format(columnnames[0])],
                                    pnames=self.state.param_names)
        self.state.set_tr_center_scaled(Sclocal.scale(self.state.tr_center).tolist())
        self.state.set_scaled_min_max_parameter_bounds(Sclocal.box_scaled[:,0].tolist(),Sclocal.box_scaled[:,1].tolist())
        X_indicies = [cnum for cnum in range(0,len(columnnames),2)]
        Y_indicies = [cnum+1 for cnum in range(0,len(columnnames),2)]
        assert(len(X_indicies) == len(Y_indicies))
        rank_indicies = None
        if rank == 0:
            rank_indicies = MPI_.chunk_it([i for i in range(len(X_indicies))])
        rank_indicies = comm.scatter(rank_indicies, root=0)
        for ri in rank_indicies:
            X = self.mc_data_df[data_name]['{}'.format(columnnames[X_indicies[ri]])]
            Y = self.mc_data_df[data_name]['{}'.format(columnnames[Y_indicies[ri]])]
            m = self.state.model_parameters[data_name]['m']\
                if data_name in self.state.model_parameters and 'm' in self.state.model_parameters[data_name] \
                else 1
            n = self.state.model_parameters[data_name]['n'] \
                if data_name in self.state.model_parameters and 'n' in self.state.model_parameters[data_name] \
                else 0
            if self.debug:
                if ((ri + 1) % 5 == 0):
                    now = time.time()
                    tel = now - t4
                    ttg = tel * (len(columnnames)/2 - ri) / (ri + 1)
                    eta = now + ttg
                    eta = datetime.datetime.fromtimestamp(now + ttg)
                    sys.stdout.write(
                        "{}[{}] {}/{} (elapsed: {:.1f}s, to go: {:.1f}s, ETA: {})\r".format(
                            80 * " " if rank > 0 else "", rank, ri + 1, len(columnnames)/2, tel, ttg,
                            eta.strftime('%Y-%m-%d %H:%M:%S')), )
                    sys.stdout.flush()
            try:
                val = apprentice.RationalApproximation(X, Y, order=(m,n), pnames=self.state.param_names)
                if self.additional_data is not None and data_name in self.additional_data:
                    if '_xmin' in self.additional_data[data_name] and '_xmax' in self.additional_data[data_name]:
                        val._xmin = self.additional_data[data_name]["_xmin"][ri]
                        val._xmax = self.additional_data[data_name]["_xmax"][ri]

                Xscaled = [Sclocal.scale(x) for x in X]
                valscaled = apprentice.RationalApproximation(Xscaled, Y, order=(m,n), pnames=self.state.param_names)
            except AssertionError as error:
                raise(error)
            term_name = columnnames[X_indicies[ri]].split('.')[0]
            app[term_name] = val.asDict
            appscaled[term_name] = valscaled.asDict
        all_apps = comm.gather(app, root=0)
        all_apps_scaled = comm.gather(appscaled, root=0)
        val_out_file = self.state.working_directory.get_log_path(
            "{}_model_k{}.json".format(data_name,self.state.k))
        scaled_val_out_file = self.state.working_directory.get_log_path(
            "{}_model_scaled_k{}.json".format(data_name,self.state.k))
        if rank == 0:
            t5 = time.time()
            if self.debug: print("Approximation calculation took {} seconds".format(t5 - t4))
            sys.stdout.flush()
            from collections import OrderedDict
            JD = OrderedDict()
            a = {}
            for apps in all_apps:
                a.update(apps)
            for k in a.keys():
                JD[k] = a[k]

            with open(val_out_file, "w") as f:
                json.dump(JD, f,indent=4)

            JD = OrderedDict()
            a = {}
            for apps in all_apps_scaled:
                a.update(apps)
            for k in a.keys():
                JD[k] = a[k]
            with open(scaled_val_out_file, "w") as f:
                json.dump(JD, f,indent=4)

        self.state.update_f_structure_model_parameters('model',{data_name:val_out_file})
        self.state.update_f_structure_model_parameters('model_scaled',{data_name:scaled_val_out_file})
