from mfstrodf.mc import MCTask
import numpy as np
import pprint,os

class SimpleApp(MCTask):
    @staticmethod
    def mapping(x):
        raise Exception("This function must be implemented in the derived class")

    def run_mc(self):
        dirlist = self.get_param_directory_array(self.mc_run_folder) # from super class
        for dno,d in enumerate(dirlist):
            param = self.get_param_from_directory(d) # from super class
            run_fidelity = self.get_run_fidelity_from_directory(d) # from super class
            if run_fidelity>0:
                Y = [np.random.normal(factor * self.mapping(param),
                                      1 / np.sqrt(run_fidelity), 1)[0]
                     for factor in range(1,4)]
                DY = [1 / np.sqrt(run_fidelity) for factor in range(1,4)]

                outfile = os.path.join(d,"out_curr.csv")
                term_names = ["Term1","Term2","Term3"]

                self.save_mc_out_as_csv(header="name,MC,DMC",
                                        term_names=term_names,data=[Y,DY],out_path=outfile
                                        ) # from super class


    def convert_mc_output_to_df(self, all_param_directory):
        #TODO temp replace with out.csv after merge_statistics_and_get_max_sigma is implemented
        df = self.convert_csv_data_to_df(all_param_directory=all_param_directory,
                                         mc_out_file_name="out_curr.csv")
        additional_data = None
        return (df,additional_data)

    # Keeping fidelity fixed for now
    def merge_statistics_and_get_max_sigma(self):
        #TODO to merge statistics
        return None

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








