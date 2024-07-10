import numpy as np
import matplotlib.pyplot as plt

def plot_bispectrum_precision(emulator, test_data):
    if emulator.emu_quantity=='squeezed':
        bispec_true = test_data[0]['squeezed_bispectrum_Mpc']
        p1d = test_data[0]['p1d_Mpc'][1:(len(bispec_true)+1)]
        rbispec_true = bispec_true/(p1d**2+2*p1d*p1d[0])
        
        coeffs = np.polyfit(np.log(emulator.k_Mpc_bisqueezed), bispec_true/(p1d**2+2*p1d*p1d[0]), deg=emulator.ndeg)
        fit_p1d = np.poly1d(coeffs)
        testing_bispec = fit_p1d(np.log(emulator.k_Mpc_bisqueezed))   
        
        emulated_bispectrum = emulator.emulate_Bispec_Mpc(test_data)
        
    elif emulator.emu_quantity=='spread':
        bispec_true = test_data[0]['spread_bispectrum_Mpc']
        rbispec_true = bispec_true/(p1ds_spread_k**2+2*p1ds_spread_k*p1ds_spread_2k)

        p1d = test_data[0]['p1d_Mpc']
        k1d = np.array(test_data[0]['k1d_Mpc'])
    
        k1d = k1d[(k1d>0) & (k1d<4)]
        k_index = np.arange(1,len(k1d),1)
        k2_index = 2*k_index
        p1ds_spread_k = p1d[k_index]  
        p1ds_spread_2k = p1d[k2_index]  
        
        coeffs = np.polyfit(np.log(emulator.k_Mpc_bispread), bispec/(p1ds_spread_k**2+2*p1ds_spread_k*p1ds_spread_2k), deg=emu.ndeg)
        fit_p1d = np.poly1d(coeffs)
        testing_bispec = fit_p1d(np.log(emulator.k_Mpc_bispread))
    
        emulated_bispectrum = emulator.emulate_Bispec_Mpc(test_data)

    
    # Calculate ratios
    ratio = (emulated_bispectrum / testing_bispec - 1)*100
    
    # Plotting
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    
    # Main plot
    ax_main = fig.add_subplot(gs[0])
    ax_main.plot(emulator.k_Mpc_training, emulated_bispectrum, label='Emulated bispectrum', color='crimson', marker='^')
    ax_main.plot(emulator.k_Mpc_training,testing_bispec , label = 'Smoothed measured bispectrum', color='black', ls='--')
    ax_main.plot(emulator.k_Mpc_training,rbispec_true , label='Measured bispectrum', color='black',ls=':', alpha=0.5)
    
    
    if emulator.emu_quantity=='squeezed':
        ax_main.set_title('Squeezed bispectrum B(k,-k,0)', fontsize=20)
        ax_main.set_ylabel(r'$Q(k,-k,0)$', fontsize=14)
    elif emulator.emu_quantity=='spread':
        ax_main.set_title('Spread bispectrum B(k,k,-2k)', fontsize=20)
        ax_main.set_ylabel(r'$Q(k,k,-2k$)', fontsize=14)
    
    ax_main.legend(fontsize=14)
    ax_main.set_xscale('log')
    
    # First ratio plot
    ax_ratio1 = fig.add_subplot(gs[1], sharex=ax_main)
    ax_ratio1.semilogx(emulator.k_Mpc_training, ratio, color='crimson')
    
    ax_ratio1.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax_ratio1.set_ylabel('Percent error', fontsize=12)
    ax_ratio1.set_ylim(-10,10)
    ax_ratio1.set_xlabel(r'$k$ [1/Mpc]', fontsize=14)
    
    # Hide x labels for the main and first ratio plots
    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.setp(ax_ratio1.get_xticklabels(), visible=False)
    
    #plt.savefig('polyfit_bisp.pdf', bbox_inches='tight')
    
    plt.show()

