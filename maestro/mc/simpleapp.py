from maestro.mc import MCTask
from maestro import DiskUtil
import numpy as np
import pprint,os

class SimpleApp(MCTask):
    """
    Simple but noisy MC Task
    """
    def run_functions(self,param,term_names,term_class,term_counts):
        F = []
        for (tname,tclss) in zip(term_names,term_class):
            if tclss is None:
                continue
            if tname.split(":")[0] == "F90NLS":
                fnret = tclss.mapping(param,m=term_counts[tname],function_name=tname.split(":")[1])
                for ff in fnret:
                    F.append(ff)
            else:
                F.append(tclss.mapping(param))
        return F

    def run_mc(self):
        """

        Run the simple app MC task

        """
        dirlist = self.get_param_directory_array(self.mc_run_folder) # from super class
        (term_names,term_class,term_counts) = self.get_functions()
        for dno,d in enumerate(dirlist):
            param = self.get_param_from_directory(d) # from super class
            run_fidelity = self.get_fidelity_from_directory(d) # from super class
            sd_weight = self.mc_parmeters['standard_deviation_weight'] \
                if 'standard_deviation_weight' in self.mc_parmeters else 1

            if run_fidelity>0:
                F = self.run_functions(param,term_names,term_class,term_counts)
                Y = [np.random.normal(f,sd_weight / np.sqrt(run_fidelity), 1)[0]
                                        for f in F]
                DY = [sd_weight / np.sqrt(run_fidelity) for clss in term_class]

                outfile = os.path.join(d,"out_curr.csv")
                self.save_mc_out_as_csv(header="name,MC,DMC",
                                        term_names=term_names,data=[Y,DY],out_path=outfile
                                        ) # from super class

    def get_functions(self):
        """

        Get the class handle for the term names given in the
        configuration input:mc:parameters:functions array

        :return: term names, the corresponding class handles for the
            term names, and counts
        :rtype: list,list,dict

        """
        term_names = []
        term_class = []
        count = {}
        import maestro.mc
        for t in self.mc_parmeters['functions']:
            if len(t.split(':'))>1:
                t_split_arr = t.split(':')
                name = t_split_arr[0]+":"+ t_split_arr[1]
                c = t_split_arr[2]
                for i in range(int(c)):
                    count[name] = 1 if name not in term_names else count[name] + 1
                    sct = "" if count[name] == 1 else str(count[name])
                    term_names.append("{}{}".format(name,sct))
                    if i==0:
                        try:
                            mc_class = getattr(maestro.mc,name.split(':')[0])
                            term_class.append(mc_class)
                        except:
                            raise Exception("MC term class \""+t+"\" not found in maestro.mc")
                    else: term_class.append(None)
            else:
                count[t] = 1 if t not in term_names else count[t] + 1
                sct = "" if count[t] == 1 else str(count[t])
                term_names.append("{}{}".format(t,sct))
                try:
                    mc_class = getattr(maestro.mc,t)
                    term_class.append(mc_class)
                except:
                    raise Exception("MC term class \""+t+"\" not found in maestro.mc")
        return term_names,term_class,count

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
        df = self.convert_csv_data_to_df(all_param_directory=all_param_directory,
                                         mc_out_file_name="out.csv")
        df = self.check_df_structure_sanity(df)
        additional_data = None
        return (df,additional_data)

    def merge_statistics_and_get_max_sigma(self):
        """

        Merge MC output statistics and find the maximum standard deviation of the
        MC output.

        :return: maximum standard deviation of the MC output
        :rtype: float

        """
        from maestro import MPI_
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        import pandas as pd
        import math
        dirlist = self.get_param_directory_array(self.mc_run_folder)
        (term_names,term_class,term_counts) = self.get_functions()
        max_sigma = 0
        for dno,d in enumerate(dirlist):
            at_fidelity = self.get_fidelity_from_directory(d,fnamef="at_fidelity.dat")
            run_fidelity = self.get_fidelity_from_directory(d)
            prev_mc_out_path = os.path.join(d,"out.csv")
            curr_mc_out_path = os.path.join(d,"out_curr.csv")
            if rank == 0:
                if not os.path.exists(prev_mc_out_path):
                    DiskUtil.copyanything(curr_mc_out_path,prev_mc_out_path)
                if not os.path.exists(curr_mc_out_path):
                    DiskUtil.copyanything(prev_mc_out_path,curr_mc_out_path)
            curr_df = None
            if rank == 0:
                curr_df = pd.read_csv(curr_mc_out_path)
            curr_df = comm.bcast(curr_df, root=0)
            rownames = list(curr_df.columns.values)
            columnnames = list(curr_df.index)
            prev_df = None
            if rank == 0:
                prev_df = pd.read_csv(prev_mc_out_path)
            prev_df = comm.bcast(prev_df, root=0)
            _Y = []
            _E = []

            sd_weight = self.mc_parmeters['standard_deviation_weight'] \
                if 'standard_deviation_weight' in self.mc_parmeters else 1
            for cno in range(len(columnnames)):
                curr_val = curr_df[rownames[1]][columnnames[cno]]
                prev_val = prev_df[rownames[1]][columnnames[cno]]
                if not math.isnan(curr_val) and not math.isinf(curr_val) \
                    and not math.isnan(prev_val) and not math.isinf(prev_val):
                    _Y.append(np.average([curr_val,prev_val]))
                else:
                    _Y.append(np.nan)
                _E.append(sd_weight / np.sqrt((at_fidelity+run_fidelity)))
            self.save_mc_out_as_csv(header="name,MC,DMC",
                                    term_names=term_names,data=[_Y,_E],out_path=prev_mc_out_path
                                    ) # from super class
            if rank == 0:
                DiskUtil.remove_file(curr_mc_out_path)
            max_sigma = max(max_sigma,np.max(_E))
        return max_sigma

class DeterministicApp(SimpleApp):
    """
    Simple & Deterministic MC Task
    """
    def run_mc(self):
        """

        Run the deterministic app MC task

        """
        dirlist = self.get_param_directory_array(self.mc_run_folder) # from super class
        (term_names,term_class,term_counts) = self.get_functions() # from super class
        for dno,d in enumerate(dirlist):
            param = self.get_param_from_directory(d) # from super class
            run_fidelity = self.get_fidelity_from_directory(d) # from super class
            if run_fidelity>0:
                Y = self.run_functions(param,term_names,term_class,term_counts)
                DY = [1. for clss in term_class]

                outfile = os.path.join(d,"out_curr.csv")
                self.save_mc_out_as_csv(header="name,MC,DMC",
                                        term_names=term_names,data=[Y,DY],out_path=outfile
                                        ) # from super class

class SumOfDiffPowers():
    """
    Sum of different powers from https://www.sfu.ca/~ssurjano/sumpow.html
    """
    @staticmethod
    def mapping(x,**kwargs):
        """

        y = f(x)

        :param x:parameter
        :type x: list
        :return: function value
        :rtype: float

        """
        # https://www.sfu.ca/~ssurjano/sumpow.html
        s = 0
        for i in range(len(x)):
            n = (abs(x[i])) ** (i + 2)
            s = s + n
        return s

class RotatedHyperEllipsoid():
    """
    Rotates Hyper Ellipsoid from https://www.sfu.ca/~ssurjano/rothyp.html
    """
    @staticmethod
    def mapping(x,**kwargs):
        """

        y = f(x)

        :param x:parameter
        :type x: list
        :return: function value
        :rtype: float

        """
        # https://www.sfu.ca/~ssurjano/rothyp.html
        outer = 0.
        for i in range(len(x)):
            inner = 0.
            for j in range(i):
                inner = inner + x[j]**2
            outer = outer + inner
        return outer

class Sphere():
    """
    Sphere from https://www.sfu.ca/~ssurjano/spheref.html
    """
    @staticmethod
    def mapping(x,**kwargs):
        """

        y = f(x)

        :param x:parameter
        :type x: list
        :return: function value
        :rtype: float

        """
        # https://www.sfu.ca/~ssurjano/spheref.html
        s = 0
        for i in range(len(x)):
            s = s + x[i]**2
        return s


class SumSquares():
    """
    Sum of Squares function from https://www.sfu.ca/~ssurjano/sumsqu.html
    """
    @staticmethod
    def mapping(x,**kwargs):
        """

        y = f(x)

        :param x:parameter
        :type x: list
        :return: function value
        :rtype: float

        """
        # https://www.sfu.ca/~ssurjano/sumsqu.html
        s = 0
        for i in range(len(x)):
            s = s + (i+1)*x[i]**2
        return s

class F90NLS():
    """
    Fortran 90 Nonlinear least squares (NLS) test problems from https://people.math.sc.edu/Burkardt/f_src/test_nls/test_nls.html
    """
    @staticmethod
    def mapping(x, m, function_name,**kwargs):
        """

        y = [f_1(x),...,f_m(x)]. This function should be called once for all terms with
        the name ``term_name``

        :param x:parameter
        :type x: list
        :param m: number of terms
        :type m: int
        :param function_name: name of the F90 NLS function
        :type function_name: str
        :return: function value for all the terms
        :rtype: list
        """
        import testnls
        fntocall = getattr(testnls,function_name)
        values = np.empty(m,dtype=np.float64)
        fntocall(m=m,n=len(x),x=np.array(x),f=values)
        return values








