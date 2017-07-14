clearvars -except batch; close all; clc

numBat = 46;
numCycles = 12;
forEvery = 1;
PCAdata = [];
startAt = 200;

for i = 1:numBat
    PCAdata_row = [];
    for j = 1:forEvery:numCycles
        cycle = j+startAt;
        PCAdata_row = [PCAdata_row, batch(i).cycles(cycle).discharge_dQdVvsV.dQdV(1,:)];
    end
    PCAdata = vertcat(PCAdata, PCAdata_row);
end

PCAdata(isnan(PCAdata)) = 0;

%% create battery color grade for future plots
bat_label = zeros(numBat,1);
for j = 1:numBat
    bat_label(j,1) = numel(batch(j).cycles); % TODO: change to last cycle before degraded
end

max_cycle = max(bat_label) + 1;
min_cycle = min(bat_label) - 1;
%CM = colormap('jet');

batt_color_grade = zeros(numBat, 1);

for i = 1:numBat
batt_color_grade(i) = ceil((bat_label(i) - min_cycle) ./ (max_cycle - min_cycle) * 64);
end

%% Perform PCA
[coeff, score, latent, ~, explained, mu] = pca(PCAdata);

X = ones(46,1);
X = [X, score];

[b,bint,r,rint,stats] = regress(bat_label, X);

figure()
plot(b)

disp('stp')