#!/usr/bin/python

import numpy as np

def simulate_packet_loss(audio_data, loss_rate):
    num_samples = len(audio_data)
    lost_samples = int(loss_rate * num_samples)
    indices_to_drop = np.random.choice(num_samples, lost_samples, replace=False)

    simulated_data = np.delete(audio_data, indices_to_drop)
    return simulated_data