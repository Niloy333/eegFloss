def plot_hypnogram_for_paper(src_night, eeg_signals, eeg_samp_rate, acc_agg, acc_samp_rate, agg_scores, lights_out_ep, lights_on_ep, mobility_scores, plot_data):
    dest_night = os.path.normpath(src_night.replace(Raw_Data_Dir, Output_Dir))
    file_path = os.path.normpath(os.path.join(dest_night, Hypnogram_Flname))
    if eeg_signals is None:
        eeg_signals, eeg_samp_rates, org_dur = read_data(src_night, 'EEG')
        if eeg_signals is None: return 
        eeg_signals, eeg_samp_rate = adjust_samp_rate(eeg_signals, eeg_samp_rates)
        eeg_signals = crop_n_make_nparray(eeg_signals, eeg_samp_rate)    
    if agg_scores is None:
        try:
            agg_scores_df, start_end_info_df, agg_scores_header = read_scores(dest_night, Artifact_Rejected_Scores_Flname)
            agg_scores = agg_scores_df.values.astype(np.int8)
        except:
            print(f"{Fore.RED}{Style.BRIGHT}\tAggregated sleep scores could not be read.\n\tHypnogram can not be plotted.{Style.RESET_ALL}")
            return False
    if int(eeg_signals.shape[1]/eeg_samp_rate/Sleep_Scores_Epoch_Length) != len(agg_scores):
        print(f"{Fore.RED}{Style.BRIGHT}\tData incompatible with sleep scores.\n\tHypnogram will not be plotted.{Style.RESET_ALL}")
        return False
    print("\tPlotting hypnogram...")
    n_sigs = eeg_signals.shape[0]    
    if ACC_Channels is not None and acc_agg is None:
        acc_signals, acc_samp_rates, _ = read_data(src_night, 'ACC')
        acc_signals, acc_samp_rate = adjust_samp_rate(acc_signals, acc_samp_rates, eeg_samp_rate)
        acc_signals = crop_n_make_nparray(acc_signals, acc_samp_rate)
        acc_agg = euclidean_norm(acc_signals)

    if acc_agg is not None:
        times_acc = np.linspace(1/acc_samp_rate, len(acc_agg)/acc_samp_rate, len(acc_agg))

    if plot_data is None:        
        plot_data = spectrogram_plot_calc(eeg_signals, eeg_samp_rate)

    num_plot = n_sigs + 2 - 1
    if mobility_scores is not None:
        num_plot = num_plot + 1
    
    agg_scores_plot = agg_scores.copy()
    if Unusable_Label != -1:
        agg_scores_plot[agg_scores_plot == Unusable_Label] = -1
    
    plt.ioff()
    #plt.rcParams['font.family'] = 'Arial'
    import string
    subplot_labels = list(string.ascii_lowercase)
    subplot_label_x, subplot_label_y = -.05, -.2
    fig, axs = plt.subplots(num_plot, 1, figsize=(11, num_plot * 1.75))

    #title_base = f"Hypnogram with (aggregated) data usability, device: {Device_Name}, sleep scores: '{Sleep_Scores_Flname}' in {Sleep_Scores_Epoch_Length}-sec epochs, usability model: eegUsability {Usability_Model_Version} in {Usability_Epoch_Length}-sec epochs\nData source: '{src_night}'"
    
    #fig.suptitle(title_base, fontsize=10, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.95)
    
    total_sec = agg_scores.shape[0] * Sleep_Scores_Epoch_Length
    label = 'Elapsed time (hh:mm)' if total_sec >= 3600 else 'Elapsed time (mm:ss)'
    for i in range(n_sigs):
        ch_name = EEG_Channels[i]
        # Spectrogram
        axs[i].pcolormesh(plot_data['times'], plot_data['freqs'],
                    10 * np.log10(np.maximum(plot_data['specs'][i], 1e-10)),
                    shading='gouraud', cmap='seismic', rasterized=True)
        axs[i].set_ylabel('Frequency (Hz)', fontsize=9)
        if i == 0:
            axs[i].set_title(f'Spectrogram of channel {ch_name} [colorbar: (high-power) red>>white>>blue (low-power)]', fontsize=9, fontweight='bold')
        else:
            axs[i].set_title(f'Spectrogram of channel {ch_name}', fontsize=9, fontweight='bold')
        axs[i].tick_params(axis='y', labelsize=9)
        axs[i].tick_params(axis='x', labelsize=9)
        format_time_axis_signal(axs[i], plot_data['times'])
        axs[i].set_xlabel(label, fontsize=9)
        axs[i].text(
            subplot_label_x, subplot_label_y, f"({subplot_labels[i]})",
            transform=axs[i].transAxes,
            fontsize=9, fontweight="bold",
            ha="left", va="bottom"
        )
    axs_id = i

    # Subplot 1: Aggregated Accelerometer
    axs_id = axs_id + 1
    if acc_agg is None:
        axs[axs_id].text(0.5, 0.5, 'No accelerometer data', horizontalalignment='center',
                    verticalalignment='center', transform=axs[n_sigs].transAxes,
                    fontsize=13, fontweight='bold', color='red')
        axs[axs_id].set_xticks([])
        axs[axs_id].set_yticks([])
    else:
        axs[axs_id].plot(times_acc, acc_agg, color='black', linewidth=0.6)
        axs[axs_id].set_xlim(times_acc.min(), times_acc.max())
        axs[axs_id].set_title('Aggregated acceleration', fontsize=9, fontweight='bold')
        axs[axs_id].set_ylabel('Acceleration (g)', fontsize=9)
        axs[axs_id].grid(True, axis='y', color='gainsboro')
        axs[axs_id].yaxis.set_major_locator(MultipleLocator(1.0))  # Full number major ticks
        axs[axs_id].tick_params(axis='y', labelsize=9)
        axs[axs_id].tick_params(axis='x', labelsize=9)
        format_time_axis_signal(axs[axs_id], times_acc)
        labela = 'Elapsed time (hh:mm)' if times_acc.max() >= 3600 else 'Elapsed time (mm:ss)'
        axs[axs_id].set_xlabel(labela, fontsize=9)
        axs[axs_id].text(
            subplot_label_x, subplot_label_y, f"({subplot_labels[axs_id]})",
            transform=axs[axs_id].transAxes,
            fontsize=9, fontweight="bold",
            ha="left", va="bottom"
        )
    
    #Agg_scores
    #axs_id = axs_id + 1
    # y = agg_scores.copy()
    # x = np.arange(len(y))
    # points = np.array([x, y]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # colors = ['crimson' if y[j] == -1 and y[j+1] == -1 else 'mediumblue' for j in range(len(y) - 1)]
    # linewidths = [1 if y[j] == -1 and y[j+1] == -1 else 0.75 for j in range(len(y) - 1)]
    # lc = LineCollection(segments, colors=colors, linewidths=linewidths)
    # axs[axs_id].add_collection(lc)
    # axs[axs_id].set_xlim(x[0], x[-1])
    # axs[axs_id].set_ylim(-1.2, 4.2)
    # axs[axs_id].grid(True, axis='y', color='gainsboro')
    # axs[axs_id].set_title(f'Hypnogram (with aggregated data usability)', fontsize=9, fontweight='bold')
    # axs[axs_id].tick_params(axis='y', labelsize=9)
    # axs[axs_id].tick_params(axis='x', labelsize=9)
    # format_time_from_index(axs[axs_id], len(agg_scores), Sleep_Scores_Epoch_Length)
    # axs[axs_id].set_xlabel(label, fontsize=9)
    # axs[axs_id].set_yticks(np.linspace(-1, 4, 6))
    # axs[axs_id].set_yticklabels(['Unscorable', 'Wake', 'N1', 'Light/N2', 'SWS/Deep/N3', 'REM'], fontsize=9)
    # axs[axs_id].text(
    #     subplot_label_x, subplot_label_y, f"({subplot_labels[axs_id]})",
    #     transform=axs[axs_id].transAxes,
    #     fontsize=10, fontweight="bold",
    #     ha="left", va="bottom"
    # )

    #Mobility _scores
    axs_id = axs_id + 1
    if mobility_scores is not None:
        axs[axs_id].plot(mobility_scores, linewidth=0.8, color='teal')
        axs[axs_id].set_xlim(0, len(mobility_scores))
        axs[axs_id].set_ylim(-0.2, 4)
        axs[axs_id].grid(True, axis='y', color='gainsboro')
        axs[axs_id].set_title(f'Degree of Mobility (the purple line indicates the identified time-in-bed)', fontsize=9, fontweight='bold')    
        axs[axs_id].tick_params(axis='y', labelsize=9)
        axs[axs_id].tick_params(axis='x', labelsize=9)
        format_time_from_index(axs[axs_id], len(mobility_scores), Mobility_Epoch_Length)
        axs[axs_id].set_xlabel(label, fontsize=9)
        axs[axs_id].set_yticks(np.linspace(0, 3, 4))
        axs[axs_id].set_yticklabels(['Idle', 'Lying', 'Stationary', 'Mobile'], fontsize=9)
        if lights_out_ep is not None and lights_out_ep != lights_on_ep:
            lights_out_ep, lights_on_ep = lights_out_ep + 2, lights_on_ep - 2
            axs[axs_id].plot(range(lights_out_ep, lights_on_ep), [3.5] * (lights_on_ep - lights_out_ep), color='purple', linewidth=1.25, linestyle='-')  
        axs[axs_id].text(
                subplot_label_x, subplot_label_y, f"({subplot_labels[axs_id]})",
                transform=axs[axs_id].transAxes,
                fontsize=9, fontweight="bold",
                ha="left", va="bottom"
            )

    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path + '.pdf', format='pdf', dpi=300)
    plt.close(fig)
    plt.close('all')
    gc.collect()
    return True
