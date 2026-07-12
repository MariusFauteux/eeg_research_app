import numpy as np
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter

board = BoardIds.GANGLION_NATIVE_BOARD
data = DataFilter.read_file("../recordings/20260630_145329_Simblee/20260630_145329_Simblee_raw.csv")        # rows = channels
fs   = 200          # 200
eeg  = BoardShim.get_eeg_channels(board)[3]        # Ch1 row
pkt  = BoardShim.get_package_num_channel(board)    # sample counter row
pkt = 13

x = data[eeg]
d = np.abs(np.diff(x))
spikes = np.where(d > 3.8 * np.std(d))[0]            # crude spike finder; tune the 5
gaps = np.diff(spikes)
print("inter-spike gap (samples):", np.median(gaps), "->", np.median(gaps) / fs, "s")
print("counter value at spikes:", data[pkt][spikes][:20].astype(int))