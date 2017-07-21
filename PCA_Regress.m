function PCA_Regress()
% clearvars -except batch batch_test batch_train held_out_unfinished; 
close all; clc

%{
numBat = numel(batch_train);
numCycles = 12;
forEvery = 1;
PCAdata = [];
startAt = 50;

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
%}

X_ones = ones(numBat,1);
X = [score(:,1), X_ones];

[b,bint,r,rint,stats] = regress(bat_label, X);

Y_pred = X * b;

figure()
plot(Y_pred, bat_label, 'o')
hold on
plot(linspace(500, 1100),linspace(500,1100), 'k')
xlabel('Predicted Cycle Number')
ylabel('Current Cycle Number')
print(gcf,'PredCycle_CurrCycle','-dpng')
