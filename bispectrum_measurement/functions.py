import numpy as np
import json
from lace.setup_simulations import read_genic
import fake_spectra.spectra as spec
import h5py

def get_skewers_filename(num,n_skewers,width_Mpc,scale_T0=None,scale_gamma=None):
    """Filename storing skewers for a particular temperature model"""

    filename='skewers_'+str(num)+'_Ns'+str(n_skewers)
    filename+='_wM'+str(int(1000*width_Mpc)/1000)
    if scale_T0:
        filename+='_sT'+str(int(1000*scale_T0)/1000)
    if scale_gamma:
        filename+='_sg'+str(int(1000*scale_gamma)/1000)
    filename+='.hdf5'
    return filename 


def get_snapshot_json_filename(num,n_skewers,width_Mpc):
    """Filename describing the set of skewers for a given snapshot"""

    filename='snap_skewers_'+str(num)+'_Ns'+str(n_skewers)
    filename+='_wM'+str(int(1000*width_Mpc)/1000)
    filename+='.json'
    return filename 

def get_transmitted_flux_fraction(skewers,scale_tau):
    """ Read optical depth from skewers object, and rescale it with scale_tau"""

    tau = skewers.get_tau(elem='H', ion=1, line=1215)
    return np.exp(-scale_tau*tau)

def measure_bispectrum_squeezed(fft_deltas,k, data, kmax_Mpc=4):
    power_skewer = np.abs(fft_deltas)**2*fft_deltas[:,0][:,None]
    mean_power = np.sum(power_skewer, axis=0)/data['nspec']
    bispectrum_Mpc = mean_power * data['L_Mpc']**2 / data['npix']**3

    k_Mpc = k * (2.0*np.pi) * data['npix'] / data['L_Mpc']
    bispectrum_Mpc = bispectrum_Mpc[(k_Mpc>0) & (k_Mpc<kmax_Mpc)]
    k_Mpc = k_Mpc[(k_Mpc>0) & (k_Mpc<kmax_Mpc)]
    return bispectrum_Mpc, k_Mpc

def measure_bispectrum_spread(fft_deltas,k ,data, kmax_Mpc=4):
    k_Mpc = k * (2.0*np.pi) * data['npix'] / data['L_Mpc']
    k_Mpc = k_Mpc[(k_Mpc>0) & (k_Mpc<kmax_Mpc)]
    k_index = np.arange(1,len(k_Mpc),1)
    k2_index = 2*k_index
    
    power_skewer = fft_deltas[:,k_index]*fft_deltas[:,k_index]*np.conj(fft_deltas[:,k2_index])
    mean_power = np.sum(power_skewer, axis=0)/data['nspec']
    
    bispectrum_Mpc = mean_power * data['L_Mpc']**2 / data['npix']**3    
    #bispectrum_Mpc = bispectrum_Mpc[(k_Mpc>0) & (k_Mpc<kmax_Mpc)]
    
    return bispectrum_Mpc, k_Mpc[1:]

def get_fftdeltas(skewers, data, scale_tau=1):
    print('it starts')
    F = get_transmitted_flux_fraction(skewers,scale_tau=scale_tau)
    mF = np.mean(F)
    delta_F = F / mF - 1.0

    (nspec, npix) = np.shape(delta_F)
    fft_deltas = np.fft.rfft(delta_F, axis=1)
    k = np.fft.rfftfreq(npix)
    data['nspec']=nspec
    data['npix']=npix

    return fft_deltas, k, data

def open_skewers(dir_, axis, skewers_description_filename, snapnum):
    f = open(dir_+f'/skewers_{axis}/'+skewers_description_filename, "r")
    data = json.loads(f.read())
            
    data['skewers_dir']=dir_+f'/skewers_{axis}/'#+skewers_filename
    data['post_dir']=dir_
    data['snap_num']=snapnum
    data['kF_Mpc']=None      

    genic_file=data['post_dir']+'/paramfile.genic'
    L_Mpc=read_genic.L_Mpc_from_paramfile(genic_file,verbose=True)
    data['L_Mpc'] = L_Mpc


    # read skewers from HDF5 file
    skewers=spec.Spectra(data['snap_num'],base="NA",cofm=None,axis=None,
            savedir=data['skewers_dir'],savefile=data['sk_files'][0],res=None,
            reload_file=False,load_snapshot=False,quiet=False)
    return skewers, data


def measure_bispectrum(sim_directory,
                       phase,
                       axis,
                       snapnum, 
                       scale_tau=1,
                    basedir='/data/desi/common/HydroData/Emulator/post_768/Australia20'):
    
    dir_ = basedir+'/'+sim_directory+'/'+phase
    skewers_description_filename=get_snapshot_json_filename(snapnum,768,0.05)
    skewers_filename=get_skewers_filename(snapnum,768,0.05, scale_T0=1,scale_gamma=1)
    skewers, data =  open_skewers(dir_, axis, skewers_description_filename, snapnum)
    fft_deltas, k, data = get_fftdeltas(skewers,
                                       data,
                                       scale_tau =scale_tau) 
    
    bispectrum_Mpc_squeezed, k_Mpc_squeezed = measure_bispectrum_squeezed(fft_deltas,
                                                                          k,
                                                                          data, 
                                                                          kmax_Mpc=4)


    bispectrum_Mpc_spread, k_Mpc_spread = measure_bispectrum_spread(fft_deltas,
                                                                    k ,
                                                                    data, 
                                                                    kmax_Mpc=4)    

    return bispectrum_Mpc_squeezed, k_Mpc_squeezed, bispectrum_Mpc_spread, k_Mpc_spread, data

    