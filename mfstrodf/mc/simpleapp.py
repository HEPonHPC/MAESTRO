from mfstrodf.mc import MCTask,DiskUtil
import numpy as np
import pprint,os

class SimpleApp(MCTask):
    @staticmethod
    def mapping(x):
        raise Exception("This function must be implemented in the derived class")

    def run_mc(self):
        dirlist = self.get_param_directory_array(self.mc_run_folder) # from super class
        term_names = ["Term1","Term2","Term3"]
        for dno,d in enumerate(dirlist):
            param = self.get_param_from_directory(d) # from super class
            run_fidelity = self.get_fidelity_from_directory(d) # from super class
            if run_fidelity>0:
                Y = [np.random.normal(factor * self.mapping(param),
                                      1 / np.sqrt(run_fidelity), 1)[0]
                     for factor in range(1,4)]
                DY = [1 / np.sqrt(run_fidelity) for factor in range(1,4)]

                outfile = os.path.join(d,"out_curr.csv")
                self.save_mc_out_as_csv(header="name,MC,DMC",
                                        term_names=term_names,data=[Y,DY],out_path=outfile
                                        ) # from super class

    def convert_mc_output_to_df(self, all_param_directory):
        df = self.convert_csv_data_to_df(all_param_directory=all_param_directory,
                                         mc_out_file_name="out.csv")
        additional_data = None
        return (df,additional_data)

    def merge_statistics_and_get_max_sigma(self):
        import pandas as pd
        import math
        dirlist = self.get_param_directory_array(self.mc_run_folder)
        term_names = ["Term1","Term2","Term3"]
        max_sigma = 0
        for dno,d in enumerate(dirlist):
            at_fidelity = self.get_fidelity_from_directory(d,fnamef="at_fidelity.dat")
            run_fidelity = self.get_fidelity_from_directory(d)
            prev_mc_out_path = os.path.join(d,"out.csv")
            curr_mc_out_path = os.path.join(d,"out_curr.csv")
            if not os.path.exists(prev_mc_out_path):
                DiskUtil.copyanything(curr_mc_out_path,prev_mc_out_path)
            if not os.path.exists(curr_mc_out_path):
                DiskUtil.copyanything(prev_mc_out_path,curr_mc_out_path)
            curr_df = pd.read_csv(curr_mc_out_path)
            rownames = list(curr_df.columns.values)
            columnnames = list(curr_df.index)
            prev_df = pd.read_csv(prev_mc_out_path)
            _Y = []
            _E = []

            for cno in range(len(columnnames)):
                curr_val = curr_df[rownames[1]][columnnames[cno]]
                prev_val = prev_df[rownames[1]][columnnames[cno]]
                if not math.isnan(curr_val) and not math.isinf(curr_val) \
                    and not math.isnan(prev_val) and not math.isinf(prev_val):
                    _Y.append(np.average([curr_val,prev_val]))
                else:
                    _Y.append(np.nan)
                _E.append(1 / np.sqrt((at_fidelity+run_fidelity)))
            self.save_mc_out_as_csv(header="name,MC,DMC",
                                    term_names=term_names,data=[_Y,_E],out_path=prev_mc_out_path
                                    ) # from super class
            DiskUtil.remove_file(curr_mc_out_path)
            max_sigma = max(max_sigma,np.max(_E))
        return max_sigma

class SumOfDiffPowers(SimpleApp):
    @staticmethod
    def mapping(x):
        # https://www.sfu.ca/~ssurjano/sumpow.html
        sum = 0
        for ii in range(len(x)):
            xi = x[ii]
            new = (abs(xi)) ** (ii + 2)
            sum = sum + new
        return sum








