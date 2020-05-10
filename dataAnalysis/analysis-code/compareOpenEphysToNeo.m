filename = 'Z:/data/rdarie/Murdoc Neural Recordings/raw/201901070700-ProprioRC/open_ephys/Block001_EMG/100_CH1.continuous';
% [data, timestamps, info] = load_open_ephys_data_faster(filename);
[data, timestamps, info] = load_open_ephys_data(filename);
plot(data(1:3e2));
folderPath = 'Z:/data/rdarie/Murdoc Neural Recordings/raw/201901070700-ProprioRC/open_ephys/Block001_EMG/';

