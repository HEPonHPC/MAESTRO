import os
import shutil,errno

import numpy as np
class OutputLevel():
    """
    Output level utility class
    """
    @staticmethod
    def get_out_level_dict():
        """

        Get output level dict

        :return: output level dict
        :rtype: dict

        """
        return  {
            "0": ["Silent"],
            "10": ["1lineoutput"],
            "11": ["1lineoutput","PKp1"],
            "20": ["1lineoutput","PKp1","interpolationPoints"],
            "30": ["1lineoutput","PKp1","interpolationPoints","MC_RA_functionValue"],
            "40": ["1lineoutput","PKp1","interpolationPoints","MC_RA_functionValue","NormOfStep"],
            "50": ["1lineoutput","PKp1","interpolationPoints","MC_RA_functionValue","NormOfStep","All"]
        }

    @staticmethod
    def get_first_out_level_with_option(option):
        """

        Get the first output level that contains the option in output level dictionary

        :param option: option to check
        :type option: str
        :return: first output level that contains the option in output level dictionary
        :rtype: str

        """
        out_level_dict = OutputLevel.get_out_level_dict()
        arr = [int(i) for i in out_level_dict.keys()]
        sarr = np.sort(arr)
        for level in sarr:
            if option in out_level_dict[str(level)]:
                return level
        return -1

    @staticmethod
    def get_output_level_def(output_level):
        """

        Get output level definition given the output level

        :param output_level: output level at which to get the definition
        :type output_level: int
        :return: output level definition given the output level
        :rtype: str

        """
        return OutputLevel.get_out_level_dict()[str(int(output_level))]

    @staticmethod
    def is_debug(output_level):
        """

        Is output level at debug

        :param output_level: output level to check for at debug
        :type output_level: int
        :return: true if  output level to check for at debug, false otherwise
        :rtype: bool

        """
        return True \
            if "All" in OutputLevel.get_output_level_def(output_level) \
            else False

    @staticmethod
    def is_one_line_output(output_level):
        """

        Is output level at one line output

        :param output_level: output level to check for whether at one line output
        :type output_level: int
        :return: true if  output level is at one line output, false otherwise
        :rtype: bool

        """
        return True \
            if "1lineoutput" in OutputLevel.get_output_level_def(output_level) \
            else False

    @staticmethod
    def is_param_kp1(output_level):
        """

        Is output level at print iterate parameter values of iteration k+1

        :param output_level: output level to check whether to print iterate parameter values of iteration k+1
        :type output_level: int
        :return: true if output level at print iterate parameter values of iteration k+1, , false otherwise
        :rtype: bool

        """
        return True \
            if "PKp1" in OutputLevel.get_output_level_def(output_level) \
            else False

    @staticmethod
    def is_norm_of_step(output_level):
        """

        Is output level at printing norm of step

        :param output_level: output level to check whether to print norm of step
        :type output_level: int
        :return: true if output level at print norm of step, false otherwise
        :rtype: bool

        """
        return True \
            if "NormOfStep" in OutputLevel.get_output_level_def(output_level) \
            else False

class DiskUtil():
    """
    Disk operation utility
    """
    @staticmethod
    def remove_file(file):
        """

        Remove file

        :param file: file path to remove
        :type file: str

        """
        os.remove(file)

    @staticmethod
    def moveanything(src, dst):
        """

        Move file or folder from source to destination

        :param src: source path
        :type src: str
        :param dst: destination path
        :type dst: str

        """
        if os.path.exists(dst):
            try:
                shutil.rmtree(dst)
            except: DiskUtil.remove_file(dst)
        shutil.move(src, dst)

    @staticmethod
    def copyanything(src, dst):
        """

        Copy file or folder from source to destination

        :param src: source path
        :type src: str
        :param dst: destination path
        :type dst: str

        """
        try:
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        except OSError as exc:
            if exc.errno == errno.ENOTDIR:
                shutil.copy(src, dst)
            else: raise


    @staticmethod
    def remove_directory(d):
        """

        Remove directory

        :param d: direcotry path to remove
        :type d: str

        """
        try:
            if os.path.exists(d):
                shutil.rmtree(d)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    @staticmethod
    def copy_directory_contents(dir_from, dir_to,exclude=None):
        """

        Copy directory contents with the option to exclude paths

        :param dir_from: source directory
        :type dir_from: str
        :param dir_to: destination directory
        :type dir_to: str
        :param exclude: files and folder paths to exclude
        :type exclude: list

        """
        import glob
        indir_list = glob.glob(os.path.join(dir_from, "*"))
        for f in indir_list:
            if exclude is not None and os.path.basename(f) not in exclude:
                DiskUtil.copyanything(f,os.path.join(dir_to,os.path.basename(f)))

class ParameterPointUtil():
    """
    Parameter point utility
    """
    @staticmethod
    def check_if_not_in_min_seperation_dist(point1,point2,min_dist):
        """

        Check if two parameters are not at minimum distance from each other (in infty norm)

        :param point1: first parameter
        :param point2: second parameter
        :param mindist: minimum distance
        :type point1: list
        :type point2: list
        :type mindist: float
        :return: true if infty norm of parameter distance is >= minimum distance, false otherwise
        :rtype: bool

        """
        distarr = [np.abs(point1[vno] - point2[vno]) for vno in range(len(point1))]
        infn = max(distarr)
        return infn >= min_dist

    @staticmethod
    def check_if_point_in_TR(point, tr_center, tr_radius):
        """

        Check if parameter is within the trust region

        :param point: parameter value
        :param tr_center: trust region center
        :param tr_radius: trust region radius
        :type point: list
        :type tr_center: list
        :type tr_radius: list
        :return: true if parameter is within trust region, false otherwise
        :rtype: bool

        """
        distarr = [np.abs(point[vno] - tr_center[vno]) for vno in range(len(point))]
        infn = max(distarr)
        return infn <= tr_radius

    @staticmethod
    def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
        """

        Is a point close to another point

        :param a: first parameter
        :type a: float
        :param b: second parameter
        :type b: float
        :param rel_tol: relative tolerance
        :type rel_tol: float
        :param abs_tol: absolute tolerance
        :type abs_tol: float
        :return: true if first parameter is close enough to the second parameter,
            false otherwise
        :rtype: bool

        """
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    @staticmethod
    def check_if_same_point(point1,point2):
        """
        Check if two parameters are the same with the relative tolerance of 1e-09

        :param point1: first parameter
        :param point2: second parameter
        :type point1: list
        :type point2: list
        :return: true if the parameters are the same, false otherwise
        :rtype: bool

        """
        for vno in range(len(point1)):
            if not ParameterPointUtil.is_close(point1[vno],point2[vno]):
                return False
        return True

    @staticmethod
    def get_infinity_norm(point):
        """

        Get infinity norm

        :param point: parameter point
        :type point: list
        :return: infinity norm
        :rtype: float

        """
        distarr = [np.abs(p) for p in point]
        infn = max(distarr)
        return infn

    @staticmethod
    def order_of_magnitude(number):
        """

        Get order of magnitude of a number

        :param number: number of which the order of magnitude is desired
        :type number: float
        :return: order of magnitude
        :rtype: int
        """
        return np.floor(np.log10(number))