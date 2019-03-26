clear; clc; close all;

dataSet = openNSx('read', 'uV', '../201901271000-Proprio/Trial003.ns5', 't:00:01','sec');


%motorA = dataSet.Data(97,:) - dataSet.Data(99,:);
%motorB = dataSet.Data(98,:) - dataSet.Data(100,:);
synchHammer = dataSet.Data;
maxSample = size(dataSet.Data, 2) / 3e4;
%%
subsample = 1;
sampleStart = 0 * 3e4 + 1;
sampleStop = 880 * 3e4;

t = (sampleStart : subsample : sampleStop) / 3e4;

%%
figure();
ax1 = subplot(1,1,1);
plot(t, synchHammer(sampleStart : subsample : sampleStop), 'DisplayName', 'SynchLine');
xlabel('Time (msec)');
hAx = gca;             % handle to current axes
hAx.XAxis.Exponent=0;  % don't use exponent
ylabel('mV')
hold on;

% ax2 = subplot(2,1,2);
% plot(t, meanData, 'DisplayName', 'Murdoc');
% xlabel('Time (msec)');
% hAx = gca;             % handle to current axes
% hAx.XAxis.Exponent=0;  % don't use exponent
% ylabel('a.u.');
% title('Murdoc 20190108');
% legend;
% linkaxes([ax1,ax2],'x');

%%
nevSet = openNEV('201902010900-Proprio/Trial002.nev', 'nomat');
%%
ainp7Mask = nevSet.Data.Spikes.Electrode == 135;
ainp7timeStamps = nevSet.Data.Spikes.TimeStamp(ainp7Mask) + 12;
plot(t(ainp7timeStamps), synchHammer(ainp7timeStamps), 'y*');


%%
chanDataSet = openNSx('read', 'uV', '201902010900-Proprio/Trial001.ns5', 't:00:01', 'min');
