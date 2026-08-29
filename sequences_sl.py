import numpy as np

def write_sequence_sl(seq_defs: dict = None,
                   seq_fn: str = 'protocol.seq'):
    """
    NOTE: I updated this 8/22/24 to reflect the 60deg pulses before + after CESL

    Create preclinical continous-wave sequence for CEST with simple readout
    :param seq_defs: sequence definitions
    :param seq_fn: sequence filename
    :return: sequence object
    """
    # Resolve pypulseq at call time (not module import) so the caller can route
    # this to the vendored Pulseq-1.3.1 build.  The bundled C++ BMCSimulator only
    # parses .seq ≤ v1.3.1; the pip pypulseq 1.5 (pulled in by BMCTool) writes a
    # v1.5 file the simulator misreads → empty magnetisation.  See
    # my_gui/worker.py::_write_sequence which activates the vendored 1.3.1.
    import pypulseq as pp

    # >>> Gradients and scanner limits - see pulseq doc for more info
    # Mostly relevant for clinical scanners
    # lims =  mr.opts('MaxGrad',30,'GradUnit','mT/m',...
    #     'MaxSlew',100,'SlewUnit','T/m/s', ...
    #     'rfRingdownTime', 50e-6, 'rfDeadTime', 200e-6, 'rfRasterTime',1e-6)
    # <<<

    # gamma
    gyro_ratio_hz = 42.5764  # for H [Hz/uT]
    gyro_ratio_rad = gyro_ratio_hz * 2 * np.pi  # [rad/uT]

    # DK addition: parameter values for B-SL pulses: 90 degree pulses on either 
    # side of spin-lock pulse (keep as block pulses); refocusing pulses with opposite phases
    pw_90 = 0.1e-3  #90 degree pulse width, in s
    pw_180 = 0.2e-3  #180 degree pulse width, in s
    
    # This is the info for the 2d readout sequence. As gradients etc ar
    # simulated as delay, we can just add a delay afetr the imaging pulse for
    # simulation which has the same duration as the actual sequence

    # the duration of the readout sequence:
    te = 20e-3
    imaging_delay = pp.make_delay(te)

    # init sequence
    seq = pp.Sequence()

    # Loop b1s
    for idx in range(seq_defs['num_meas']):
        exctip = seq_defs['excFA'][idx]
        B1 = seq_defs['B1pa'][idx]
        ppmoff = seq_defs['offsets_ppm'][idx]
        tp = seq_defs['tp'][idx]
    #     td = seq_defs['td'][idx]
        trec = seq_defs['Trec'][idx-1]
        isSL = seq_defs['SLflag'][idx]
        SLtip = seq_defs['SLFA'][idx]

        if idx > 0:  # add relaxtion block after first measurement
            seq.add_block(pp.make_delay(trec - te))  # net recovery time

        # saturation/spinlock pulse
        current_offset_hz = ppmoff * seq_defs['B0'] * gyro_ratio_hz
        fa_sat = B1 * gyro_ratio_rad * tp  # flip angle of sat pulse, in rad/s
        
        # excitation pulse, prior to readout
        imaging_pulse = pp.make_block_pulse(exctip * np.pi / 180, duration=2.1e-3)

        
        # pre- and post-spinlock pulses
        # DK mod 9/5/24: because I don't believe Pulseq assumes things are 
        # phase-continuous as we hop between rotating frames, we must account for 
        # the extra phase the water accumulates as it "follows" the B_eff field, 
        # which is stationary in the off-resonant rotating frame but in water's 
        # frame precesses about the z-axis! (and brings the water along with it!) 
        # See write_sequence_clinical() in original sequences.py file 
        accum_phase = np.mod(current_offset_hz * 2 * np.pi * tp, 2 * np.pi)
        
        # DK mod 9/5/24: We also need to reverse the pre- and post-SL pulse phases, 
        # similar to write_sequence_clinical() in original sequences.py file
        pre_spinlock_pulse = pp.make_block_pulse(SLtip * np.pi / 180, duration=pw_90, freq_offset=0, phase_offset=90 * np.pi / 180)
        post_spinlock_pulse = pp.make_block_pulse(SLtip * np.pi / 180, duration=pw_90, freq_offset=0, phase_offset=270 * np.pi / 180 + accum_phase)
        bsl_refpulse_1 = pp.make_block_pulse(180 * np.pi / 180, duration=pw_180, freq_offset=0, phase_offset=180 * np.pi / 180 + accum_phase/4)
        bsl_refpulse_2 = pp.make_block_pulse(180 * np.pi / 180, duration=pw_180, freq_offset=0, phase_offset=0 * np.pi / 180 + accum_phase*3/4)
        # add pulses
        # DK modification: detect if current_offset_hz is 0, and if so
        # add in 90deg pre-pulse for on-resonance spin-lock
        # if current_offset_hz < 1e-3:
        if isSL:
            seq.add_block(pre_spinlock_pulse)
            for n_p in range(seq_defs['n_pulses']):
                # ### REGULAR SPIN-LOCK ###
                # # If B1 is 0 simulate delay instead of any pulses
                # if B1 == 0:
                #     seq.add_block(pp.make_delay(tp))  # net recovery time
                # else:
                #     sl_lockpulse = pp.make_block_pulse(fa_sat, duration=tp, freq_offset=current_offset_hz, phase_offset=0 * np.pi / 180)
                #     # =='system', lims== should be added for clinical scanners
                #     seq.add_block(sl_lockpulse)
                # # delay between pulses
                # if n_p < seq_defs['n_pulses'] - 1:
                #     seq.add_block(pp.make_delay(seq_defs['td']))
                # ### END REGULAR SPIN-LOCK ###

                # ### BALANCED SPIN-LOCK ###
                # # If B1 is 0 simulate delay instead of any pulses
                # if B1 == 0:
                #     seq.add_block(pp.make_delay(tp))  # net recovery time
                # else:
                #     bsl_lockpulse_beg = pp.make_block_pulse(fa_sat/4, duration=tp/4, freq_offset=current_offset_hz, phase_offset=0 * np.pi / 180)
                #     bsl_lockpulse_mid = pp.make_block_pulse(fa_sat/2, duration=tp/2, freq_offset=current_offset_hz, phase_offset=180 * np.pi / 180  + accum_phase/4)
                #     bsl_lockpulse_end = pp.make_block_pulse(fa_sat/4, duration=tp/4, freq_offset=current_offset_hz, phase_offset=0 * np.pi / 180 + accum_phase*3/4)
                #     # =='system', lims== should be added for clinical scanners
                #     seq.add_block(bsl_lockpulse_beg)
                #     seq.add_block(bsl_refpulse_1)
                #     seq.add_block(bsl_lockpulse_mid)
                #     seq.add_block(bsl_refpulse_2)
                #     seq.add_block(bsl_lockpulse_end)
                # # delay between pulses
                # if n_p < seq_defs['n_pulses'] - 1:
                #     seq.add_block(pp.make_delay(seq_defs['td']))
                # ### END BALANCED SPIN-LOCK ###

                ### COMBO SPIN-LOCK : REGULAR IF OFF-RES, BALANCED IF ON-RES ###
                # If B1 is 0 simulate delay instead of any pulses
                if B1 == 0:
                    seq.add_block(pp.make_delay(tp))  # net recovery time
                elif current_offset_hz < 1e-3:  # balanced spin-lock if on-res
                    bsl_lockpulse_beg = pp.make_block_pulse(fa_sat/4, duration=tp/4, freq_offset=current_offset_hz, phase_offset=180 * np.pi / 180)
                    bsl_lockpulse_mid = pp.make_block_pulse(fa_sat/2, duration=tp/2, freq_offset=current_offset_hz, phase_offset=0 * np.pi / 180  + accum_phase/4)
                    bsl_lockpulse_end = pp.make_block_pulse(fa_sat/4, duration=tp/4, freq_offset=current_offset_hz, phase_offset=180 * np.pi / 180 + accum_phase*3/4)
                    # =='system', lims== should be added for clinical scanners
                    seq.add_block(bsl_lockpulse_beg)
                    seq.add_block(bsl_refpulse_1)
                    seq.add_block(bsl_lockpulse_mid)
                    seq.add_block(bsl_refpulse_2)
                    seq.add_block(bsl_lockpulse_end)
                else:   # regular spin-lock if off-res
                    sl_lockpulse = pp.make_block_pulse(fa_sat, duration=tp, freq_offset=current_offset_hz, phase_offset=180 * np.pi / 180)
                    # =='system', lims== should be added for clinical scanners
                    seq.add_block(sl_lockpulse)                  
                # delay between pulses
                if n_p < seq_defs['n_pulses'] - 1:
                    seq.add_block(pp.make_delay(seq_defs['td']))
                ### END COMBO SPIN-LOCK ###
                
            seq.add_block(post_spinlock_pulse)
        else:
            for n_p in range(seq_defs['n_pulses']):
                # If B1 is 0 simulate delay instead of a saturation pulse
                if B1 == 0:
                    seq.add_block(pp.make_delay(tp))  # net recovery time
                else:
                    sat_pulse = pp.make_block_pulse(fa_sat, duration=tp, freq_offset=current_offset_hz)
                    # =='system', lims== should be added for clinical scanners
                    seq.add_block(sat_pulse)
                # delay between pulses
                if n_p < seq_defs['n_pulses'] - 1:
                    seq.add_block(pp.make_delay(seq_defs['td']))

        # DK note: maybe I need a spoiler here??

        # Imaging pulse
        seq.add_block(imaging_pulse)
        seq.add_block(imaging_delay)
        pseudo_adc = pp.make_adc(1, duration=1e-3)
        seq.add_block(pseudo_adc)

    def_fields = seq_defs.keys()
    for field in def_fields:
        seq.set_definition(field, seq_defs[field])

    seq.write(seq_fn)
    return seq


# ─────────────────────────────────────────────────────────────────────────────
# Optional scanner-limit writer (opt-in).  Same spin-lock CEST-MRF family as
# write_sequence_sl above, but every RF/gradient block is created with an
# explicit `system=lims` (a pypulseq Opts built from the target scanner's
# hardware limits) plus real spoiler gradients and hardware delays.  With one
# set of limits this writes a hardware-valid .seq for ANY scanner.
#
#   type='scanner'    → single-sample pseudo-ADC readout + hardware delays
#   type='simulation' → readout abstracted as a FLASH-like delay train
#
# The default simulation/matching path still uses write_sequence_sl (limit-free);
# this function is only called when the user enables scanner limits.
# ─────────────────────────────────────────────────────────────────────────────
def write_sequence_clinical(seq_defs: dict, seq_fn: str, lims=None, type='scanner'):
    """
    Create clinical pulsed-wave sequence for CEST with complex readout.
    :param seq_defs: sequence definitions (keys: gamma_hz, freq, b0, offsets_ppm,
                     b1, tp, td, trec, n_pulses, spoiling)
    :param seq_fn:   sequence filename
    :param lims:     scanner limits — a pypulseq Opts object (required)
    :param type:     'scanner' (hardware-valid) or 'simulation'
    :return:         sequence object
    """
    import numpy as np
    import pypulseq as pp

    if lims is None:
        raise ValueError("write_sequence_clinical requires a pypulseq Opts `lims`.")

    GAMMA_HZ = seq_defs["gamma_hz"]

    tp_sl = 1e-3  # duration of tipping pulse for sl
    td_sl = lims.rf_dead_time + lims.rf_ringdown_time  # delay between tip and sat pulse
    sl_time_per_sat = 2 * (tp_sl + td_sl)  # additional time of sl pulses for 1 sat pulse
    assert (np.asarray(seq_defs["trec"]) >= sl_time_per_sat).all(), \
        "DC too high for SL preparation pulses!"

    sl_pause_time = 250e-6

    # spoiler
    spoil_amp = 0.8 * lims.max_grad  # Hz/m
    rise_time = 1.0e-3  # spoiler rise time in seconds
    spoil_dur = 4500e-6 + rise_time  # complete spoiler duration in seconds

    gx_spoil, gy_spoil, gz_spoil = [
        pp.make_trapezoid(channel=c, system=lims, amplitude=spoil_amp,
                          duration=spoil_dur, rise_time=rise_time)
        for c in ["x", "y", "z"]
    ]
    if type == 'scanner':
        sl_pause_time = sl_pause_time - lims.rf_dead_time - lims.rf_ringdown_time

    min_fa = 1

    pseudo_adc = pp.make_adc(num_samples=1, duration=1e-3)
    offsets_hz = np.asarray(seq_defs["offsets_ppm"]) * seq_defs["freq"]  # ppm → Hz

    phase_cycling = 50 / 180 * np.pi
    seq = pp.Sequence()

    for m, b1 in enumerate(seq_defs["b1"]):
        # prep and set rf pulse
        flip_angle_sat = b1 * GAMMA_HZ * 2 * np.pi * seq_defs["tp"]
        sat_pulse = pp.make_block_pulse(flip_angle=flip_angle_sat, duration=seq_defs["tp"],
                                        freq_offset=offsets_hz[m], system=lims)
        accum_phase = np.mod(offsets_hz[m] * 2 * np.pi * seq_defs["tp"], 2 * np.pi)

        # prep spin lock pulses
        flip_angle_tip = np.arctan(b1 / (seq_defs["offsets_ppm"][m] * seq_defs["b0"] + 1e-8))
        pre_sl_pulse = pp.make_block_pulse(flip_angle=flip_angle_tip, duration=tp_sl,
                                           phase_offset=-(np.pi / 2), system=lims)
        post_sl_pulse = pp.make_block_pulse(flip_angle=flip_angle_tip, duration=tp_sl,
                                            phase_offset=accum_phase + (np.pi / 2), system=lims)

        sat_pulse.freq_offset = offsets_hz[m]
        for n in range(seq_defs["n_pulses"]):
            pre_sl_pulse.phase_offset = pre_sl_pulse.phase_offset + phase_cycling
            sat_pulse.phase_offset = sat_pulse.phase_offset + phase_cycling
            post_sl_pulse.phase_offset = post_sl_pulse.phase_offset + phase_cycling

            if b1 == 0:
                seq.add_block(pp.make_delay(seq_defs["tp"]))
                seq.add_block(pp.make_delay(sl_time_per_sat))
            else:
                if flip_angle_tip > min_fa / 180 * np.pi:
                    seq.add_block(pre_sl_pulse)
                    seq.add_block(pp.make_delay(sl_pause_time))
                else:
                    seq.add_block(pp.make_delay(pp.calc_duration(pre_sl_pulse) + sl_pause_time))

                seq.add_block(sat_pulse)

                if flip_angle_tip > min_fa / 180 * np.pi:
                    seq.add_block(pp.make_delay(sl_pause_time))
                    seq.add_block(post_sl_pulse)
                else:
                    seq.add_block(pp.make_delay(pp.calc_duration(post_sl_pulse) + sl_pause_time))

            if n < seq_defs["n_pulses"] - 1:
                seq.add_block(pp.make_delay(seq_defs["td"] - sl_time_per_sat))

        if type == 'scanner':
            seq.add_block(pp.make_delay(100e-6))  # hardware related delay

        if seq_defs["spoiling"]:
            seq.add_block(gx_spoil, gy_spoil, gz_spoil)
            if type == 'scanner':
                seq.add_block(pp.make_delay(100e-6))  # hardware related delay

        # Readout sequence
        if type == 'scanner':
            seq.add_block(pseudo_adc)
            readout_time = pp.calc_duration(pseudo_adc)
        else:
            n_shots = 45
            tr1 = 28.370e-3
            tp1 = 2e-3
            tpause = tr1 - tp1
            fa1 = 15 * np.pi / 180
            flip_pulse = pp.make_block_pulse(flip_angle=fa1, duration=tp1)
            relax_time_readout = pp.make_delay(tpause)
            for ii in range(n_shots):
                seq.add_block(flip_pulse)
                seq.add_block(relax_time_readout)
                if ii == 0:
                    seq.add_block(pseudo_adc)
            readout_time = 1e-3

        # add delay
        if m < len(seq_defs["b1"]) - 1:
            seq.add_block(pp.make_delay(seq_defs["trec"][m] - readout_time))

    for field in seq_defs.keys():
        seq.set_definition(field, seq_defs[field])

    seq.write(seq_fn)
    return seq
