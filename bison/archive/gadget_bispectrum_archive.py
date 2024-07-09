import numpy as np
import copy
import sys
import os
import json

import lace
from lace.setup_simulations import read_genic, read_gadget
from lace.archive.base_archive import BaseArchive
from lace.utils.exceptions import ExceptionList


class GadgetBispectrumArchive(BaseArchive):

    def __init__(
        self,
        force_recompute_linP_params=False,
        kp_Mpc=0.7,
        drop_sim=None,
    ):
        """
        Initialize the archive object.

            kp_Mpc (None or float): Optional. Pivot point used in linear power parameters.
                If specified, the parameters will be recomputed in the archive. Default is None.
            fore_recompute_linP_params (boolean). If set, it will recompute linear power parameters even if kp_Mpc match. Default is False.

        Returns:
            None

        """

        self.list_sim_test = [
            "mpg_central"]#,
            #"mpg_seed",
            #"mpg_growth",
            #"mpg_neutrinos",
            #"mpg_curved",
            #"mpg_running",
            #"mpg_reio",
        #]
        self.kp_Mpc = kp_Mpc
        # list of hypercube simulations
        self.list_sim_cube = []
        for ii in range(30):
            self.list_sim_cube.append("mpg_" + str(ii))
        # list all simulations
        self.list_sim = self.list_sim_cube + self.list_sim_test
        ## done set simulation list
        self.drop_sim=drop_sim

        # list all redshifts
        self.list_sim_redshifts = np.arange(2, 4.6, 0.25)
        self.list_sim_axes = [0, 1, 2]
        self.list_scalings = [0.9, 0.95, 1,1.05,1.1]
  

        # get relevant flags for post-processing
        self._set_info_postproc()

        # load power spectrum measurements
        self._load_data(force_recompute_linP_params)
        self._set_training_data()
        self._set_testing_data()

        # extract indexes from data
        #self._set_labels()


    def _sim2file_name(self, sim_label):
        """
        Convert simulation labels to file names.

        Args:
            sim_label (int or str): Selected simulation.

        Returns:
            tuple: A tuple containing the simulation file names and parameter file tag.

        """


        dict_conv = {
            "mpg_central": "sim_pair_30",
            "mpg_seed": "diffSeed",
            "mpg_growth": "sim_pair_h",
            "mpg_neutrinos": "nu_sim",
            "mpg_curved": "curved_003",
            "mpg_running": "running",
            "mpg_reio": "P18",
        }
        dict_conv_params = {
            "mpg_central": "central",
            "mpg_seed": "diffSeed_sim",
            "mpg_growth": "h_sim",
            "mpg_neutrinos": "nu_sim",
            "mpg_curved": "curved_003",
            "mpg_running": "running_sim",
            "mpg_reio": "P18_sim",
        }

        dict_conv_params = dict_conv

        for ii in range(30):
            dict_conv["mpg_" + str(ii)] = "sim_pair_" + str(ii)
            dict_conv_params["mpg_" + str(ii)] = "sim_pair_" + str(ii)

        if sim_label in self.list_sim_test:
            tag_param = self.tag_param
        else:
            tag_param = "parameter.json"

        return dict_conv[sim_label], dict_conv_params[sim_label], tag_param


    def _load_data(self, force_recompute_linP_params):
        self.data = []
        keys_in = list(self.key_conv.keys())
        
        ## read file containing information about simulation suite
        cube_json = self.fulldir + "/latin_hypercube.json"
        try:
            with open(cube_json) as json_file:
                self.cube_data = json.load(json_file)
        except FileNotFoundError:
            print(f"Error: Cube JSON file '{cube_json}' not found.")
        else:
            self.nsamples = self.cube_data["nsamples"]  

        ## read info from all sims, all snapshots, all rescalings
        # iterate over simulations
        for sim_label in self.list_sim:
            cosmo_params, linP_params = self._get_emu_cosmo(
                sim_label, force_recompute_linP_params
            )

            # iterate over snapshots
            for ind_z in range(linP_params["z"].shape[0]):
                for pp in range(len(self.list_scalings)):
                # set linear power parameters describing snapshot
                    snap_data = {}
                    # identify simulation
                    snap_data["sim_label"] = sim_label
                    snap_data["ind_snap"] = ind_z
                    snap_data["ind_rescaling"] = pp
                    snap_data["scale_tau"] = self.list_scalings[pp]
                    snap_data["cosmo_params"] = cosmo_params

                    
                    for lab in linP_params.keys():
                        if lab == "kp_Mpc":
                            snap_data[lab] = linP_params[lab]
                        else:
                            snap_data[lab] = linP_params[lab][ind_z]
                            
                    squeezed_bispectrum_Mpc, spead_bispectrum_Mpc, p1d_Mpc, k_squeezed_Mpc, k_spread_Mpc,k1d,  params_snap = self._get_sim(
                        sim_label,
                        ind_z,  
                        ind_sc=pp)


                    snap_data["squeezed_bispectrum_Mpc"] = squeezed_bispectrum_Mpc
                    snap_data["spread_bispectrum_Mpc"] = spead_bispectrum_Mpc
                    snap_data['p1d_Mpc'] = p1d_Mpc
                    snap_data["k_squeezed_Mpc"]=k_squeezed_Mpc
                    snap_data["k_spread_Mpc"]=k_spread_Mpc
                    snap_data["k1d_Mpc"]=k1d
                   

                    snap_data["f_p"] = params_snap['linP_zs']["f_p"]
                    snap_data["Delta2_p"] = params_snap['linP_zs']["Delta2_p"]
                    snap_data["n_p"] = params_snap['linP_zs']["n_p"]
                    snap_data["alpha_p"] = params_snap['linP_zs']["alpha_p"]

                    snap_data["mF"] = params_snap['igm']["mF"]
                    snap_data["gamma"] = params_snap['igm']["gamma"]
                    snap_data["sigT_Mpc"] = params_snap['igm']["sigT_Mpc"]
                    snap_data["kF_Mpc"] = params_snap['igm']["kF_Mpc"]

                    


                    self.data.append(snap_data)
                        
    def _set_training_data(self):
        self.training_data = [d for d in self.data if d['sim_label'] in self.list_sim_cube if d['sim_label']!=self.drop_sim]

    
    def get_training_data(self):
        return self.training_data 
        

    def _set_testing_data(self):
        self.testing_data = [d for d in self.data if d['sim_label'] in self.list_sim_test and d['scale_tau']==1]
        
    def get_testing_data(self):
        return self.testing_data
                                
            
    def _set_info_postproc(self):

        self.basedir = "data/sim_suites/post_768"
        self.n_phases = 2
        self.n_axes = 3
        self.sk_label = "Ns768_wM0.05"
        self.basedir_params = "data/sim_suites/post_768"
        self.sk_label_params = "Ns500_wM0.05"
        self.tag_param = "parameter_redundant.json"
        self.scalings_avail = np.arange(5, dtype=int)
        self.training_val_scaling = "all"
        self.training_z_min = 0
        self.training_z_max = 10
        self.testing_ind_rescaling = 0
        self.testing_z_min = 0
        self.testing_z_max = 10

        repo = os.path.dirname(lace.__path__[0]) + "/"


        self.fulldir = repo + self.basedir
        self.fulldir_param = repo + self.basedir_params

        self.key_conv = {
            "mF": "mF",
            "sim_T0": "T0",
            "sim_gamma": "gamma",
            "sim_sigT_Mpc": "sigT_Mpc",
            "kF_Mpc": "kF_Mpc",
            "k_Mpc": "k_Mpc",
            "p1d_Mpc": "p1d_Mpc",
            "scale_tau": "over",
        }

        self.scaling_cov = {
            1: 0,
            0.90: 1,
            0.95: 2,
            1.05: 3,
            1.1: 4,
        }

    def _sim2file_name(self, sim_label):
        """
        Convert simulation labels to file names.

        Args:
            sim_label (int or str): Selected simulation.

        Returns:
            tuple: A tuple containing the simulation file names and parameter file tag.

        """


        dict_conv = {
            "mpg_central": "sim_pair_30",
            "mpg_seed": "diffSeed",
            "mpg_growth": "sim_pair_h",
            "mpg_neutrinos": "nu_sim",
            "mpg_curved": "curved_003",
            "mpg_running": "running",
            "mpg_reio": "P18",
        }
        dict_conv_params = {
            "mpg_central": "central",
            "mpg_seed": "diffSeed_sim",
            "mpg_growth": "h_sim",
            "mpg_neutrinos": "nu_sim",
            "mpg_curved": "curved_003",
            "mpg_running": "running_sim",
            "mpg_reio": "P18_sim",
        }

        dict_conv_params = dict_conv

        for ii in range(30):
            dict_conv["mpg_" + str(ii)] = "sim_pair_" + str(ii)
            dict_conv_params["mpg_" + str(ii)] = "sim_pair_" + str(ii)

        if sim_label in self.list_sim_test:
            tag_param = self.tag_param
        else:
            tag_param = "parameter.json"

        return dict_conv[sim_label], dict_conv_params[sim_label], tag_param

    def _get_emu_cosmo(self, sim_label, force_recompute_linP_params=False):
        """
        Get the cosmology and parameters describing linear power spectrum from simulation.

        Args:
            sim_label: Selected simulation.
            force_recompute_linP_params: recompute linP even if kp_Mpc matches

        Returns:
            tuple: A tuple containing the following info:
                - cosmo_params (dict): contains cosmlogical parameters
                - linP_params (dict): contains parameters describing linear power spectrum

        """

        # figure out whether we need to compute linP params
        compute_linP_params = False

        if force_recompute_linP_params:
            compute_linP_params = True
        else:
            # open file with precomputed values to check kp_Mpc
            try:
                file_cosmo = np.load(
                    self.fulldir + "/mpg_emu_cosmo.npy", allow_pickle=True
                )
            except IOError:
                raise IOError("The file " + self.fulldir + "mpg_emu_cosmo.npy" + " does not exist")

            for ii in range(len(file_cosmo)):
                if file_cosmo[ii]["sim_label"] == sim_label:
                    # if kp_Mpc not defined, use precomputed value
                    if self.kp_Mpc is None:
                        self.kp_Mpc = file_cosmo[ii]["linP_params"]["kp_Mpc"]

                    # if kp_Mpc different from precomputed value, compute
                    if self.kp_Mpc != file_cosmo[ii]["linP_params"]["kp_Mpc"]:
                        if self.verbose:
                            print("Recomputing kp_Mpc at " + str(self.kp_Mpc))
                        compute_linP_params = True
                    else:
                        cosmo_params = file_cosmo[ii]["cosmo_params"]
                        linP_params = file_cosmo[ii]["linP_params"]
                    break

        if compute_linP_params == True:
            # this is the only place you actually need CAMB
            from lace.cosmo import camb_cosmo, fit_linP

            _, sim_name_param, tag_param = self._sim2file_name(sim_label)
            pair_dir = self.fulldir_param + "/" + sim_name_param

            # read gadget file
            gadget_fname = pair_dir + "/sim_plus/paramfile.gadget"
            gadget_cosmo = read_gadget.read_gadget_paramfile(gadget_fname)
            zs = read_gadget.snapshot_redshifts(gadget_cosmo)

            # setup cosmology from GenIC file
            genic_fname = pair_dir + "/sim_plus/paramfile.genic"
            cosmo_params = read_genic.camb_from_genic(genic_fname)

            # setup CAMB object
            sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo_params)

            # compute linear power parameters at each z (in Mpc units)
            linP_zs = fit_linP.get_linP_Mpc_zs(sim_cosmo, zs, self.kp_Mpc)
            # compute conversion from Mpc to km/s using cosmology
            dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array(zs))

            linP_params = {}
            linP_params["kp_Mpc"] = self.kp_Mpc
            labels = ["z", "dkms_dMpc", "Delta2_p", "n_p", "alpha_p", "f_p"]
            for lab in labels:
                linP_params[lab] = np.zeros(zs.shape[0])
                for ii in range(zs.shape[0]):
                    if lab == "z":
                        linP_params[lab][ii] = zs[ii]
                    elif lab == "dkms_dMpc":
                        linP_params[lab][ii] = dkms_dMpc_zs[ii]
                    else:
                        linP_params[lab][ii] = linP_zs[ii][lab]

        return cosmo_params, linP_params

    def _get_file_names(self, sim_label, ind_phase, ind_z, ind_axis, ind_sc):
        """
        Get the file names for the specified simulation parameters and snapshot.

        Args:
            sim_label (int or str): Selected simulation.
            ind_phase (int): Index of the simulation phase.
            ind_z (int): Index of the redshift.
            ind_axis (int): Index of the simulation axis.

        Returns:
            tuple: A tuple containing the file names for data and parameter JSON files.
                - data_json (str): File name for the data JSON file.
                - param_json (str): File name for the parameter JSON file.

        Notes:
            - The `sim_label` argument refers to the selected simulation.
            - The `ind_phase` argument refers to the index of the simulation phase.
            - The `ind_z` argument refers to the index of the redshift.
            - The `ind_axis` argument refers to the index of the simulation axis.
        """

        sim_name, sim_name_param, tag_param = self._sim2file_name(sim_label)

        if ind_phase == 0:
            tag_phase = "sim_plus"
        else:
            tag_phase = "sim_minus"

        if ind_sc in [1,3]:
            p1d_label = "p1d_setau"
        else:
            p1d_label = "p1d_stau"
            
  
        _sk_label_data = f"bispectrum_{ind_z}_stau_{ind_sc}_Ns768_wM0.05_axis{ind_axis+1}" 
        _sk_label_params = self.sk_label_params

        # path to measurements

        filename_bispectrum = (
                "/Users/lauracabayol/Documents/DESI/Australia20"
                + "/"
                + sim_name
                + "/"
                + tag_phase
                + "/"
                + "bispectrum/"
                + _sk_label_data
                + ".json"
                              )

        filename_p1d = (
                "/Users/lauracabayol/Documents/DESI/Australia20"
                + "/"
                + sim_name
                + "/"
                + tag_phase
                + "/"
                + p1d_label
                + "_"
                + str(ind_z)
                + "_"
                + f"Ns768_wM0.05_axis{ind_axis+1}"
                + ".json"
            )
            

        # path to parameters
        param_json = (
            self.fulldir_param
            + "/"
            + sim_name
            + "/"
            + "parameter.json"
        )
        

        return filename_bispectrum, filename_p1d,  param_json

    def _get_sim(self, sim_label, ind_z, ind_sc):
        """
        Get the data and parameter information for the specified simulation parameters and snapshot.

        Args:
            self (object): The instance of the class containing this method.
            sim_label (str): Label of the simulation to retrieve.
            ind_z (int): Index of the redshift.
            ind_axis (int): Index of the simulation axis.

        Returns:
            tuple: A tuple containing the data and parameter information.
                - phase_data (list): List of dictionaries containing the data for each phase.
                - phase_params (list): List of dictionaries containing the parameter information for each phase.
                - arr_phase (list): List of phase indices corresponding to each data entry.

        Note:
            This function retrieves the data and parameter information for the specified simulation parameters and snapshot.
            The data is obtained by reading JSON files stored at specific paths.

        """
        #tmp hack
        idx_conv = {0:0,
                 1:0,
                 2:1,
                 3:1,
                 4:2}

        
        squeezed_bispectrum_data = []
        spread_bispectrum_data = []
        p1d_data = []
        igm=[]

        # open sim_plus and sim_minus 
        for ind_phase in range(self.n_phases):
            for ind_axis in range(self.n_axes):
                bispectrum_filename, p1d_filename, fparam_filename = self._get_file_names(sim_label, ind_phase, ind_z, ind_axis, ind_sc)
                
                f = open(bispectrum_filename, "r") 
                bispectrum_json = json.loads(f.read())
                
                f = open(p1d_filename, "r") 
                p1d_json = json.loads(f.read())


                p1d = p1d_json['p1d_data'][idx_conv[ind_sc]]['p1d_Mpc']
                k1d = p1d_json['p1d_data'][idx_conv[ind_sc]]['k_Mpc']
                
                

                igm.append(bispectrum_json['snapshot_data']['sim_mf'][0])
                igm.append(bispectrum_json['snapshot_data']['sim_gamma'][0])
                igm.append(bispectrum_json['snapshot_data']['sim_sigT_Mpc'][0])
                igm.append(bispectrum_json['snapshot_data']['kF_Mpc'])


                squeezed_bispectrum_data.append(bispectrum_json['bispectrum_data'][0]['bispectrum_Mpc_squeezed'])
                spread_bispectrum_data.append(bispectrum_json['bispectrum_data'][0]['bispectrum_Mpc_spread'])

                p1d_data.append(p1d)


        igm = np.array(igm).reshape(6,4).mean(0)

        squeezed_bispectrum_data = np.array(squeezed_bispectrum_data).mean(0)
        spread_bispectrum_data = np.array(spread_bispectrum_data).mean(0)
        p1d_data = np.array(p1d_data).mean(0)

        kMpc_squeezed = bispectrum_json['bispectrum_data'][0]['k_Mpc_squeezed']
        kMpc_spread = bispectrum_json['bispectrum_data'][0]['k_Mpc_spread']
            
        # read params
        fparam = open(fparam_filename, "r") 
        params_json = json.loads(fparam.read())
        params_snap = {}
        params_snap['linP_zs'] = {}
        params_snap['linP_zs'] = params_json['linP_zs'][ind_z]
        params_snap['igm'] = {}
        params_snap['igm']['mF'] = igm[0]
        params_snap['igm']['gamma'] = igm[1]
        params_snap['igm']['sigT_Mpc'] = igm[2]
        params_snap['igm']['kF_Mpc'] = igm[3]

        
        return squeezed_bispectrum_data, spread_bispectrum_data, p1d_data, kMpc_squeezed, kMpc_spread, k1d, params_snap
