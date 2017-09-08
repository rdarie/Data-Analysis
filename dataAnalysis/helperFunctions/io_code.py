from nptdms import TdmsFile
import pdb
import warnings

def TDMS_to_dict(TDMS_filename, input_chan, output_chan):
    """Convert raw data from a Labview VI into a dictionary of Python dictionaries
    Usage:
    TDMS_filename (string): full path to the input data filename
    input_chan (list of int): List of channel to be saved
    output_chan (list of strings): Identifier for each of the saved channels
    """

    tdms_file = TdmsFile(TDMS_filename)
    list_out = {}
    for idx in range(len(input_chan)):
        channel_name = 'Dev2/ai'+str(input_chan[idx])
        #pdb.set_trace()
        channel = tdms_file.object('Untitled', channel_name)
        data = channel.data
        time = channel.time_track()
        sr = 1/(time[1] - time[0])
        entry = {'data':data, 'time':time, 'sr':sr}
        list_out[output_chan[idx]] = entry

    return list_out
