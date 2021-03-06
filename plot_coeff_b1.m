clearvars -except batch batch_test batch_train held_out_unfinished; 
close all; clc

%load 2017-05-12_batchdata_modified.mat
%load train_test_partition.mat

coeffients_50_182 = zeros(12,12000);
numBat = numel(batch_train);
numCycles = 12;
forEvery = 1;

for i = 1:12
    PCAdata = [];
    startAt = 50 + (i-1)*12;

    %% Generate data for PCA input
    for x = 1:numBat
        PCAdata_row = [];
        for j = 1:forEvery:numCycles
            cycle = j+startAt;
            PCAdata_row = [PCAdata_row, ...
                batch_train(x).cycles(cycle).discharge_dQdVvsV.dQdV(1,:)];
        end
        PCAdata = vertcat(PCAdata, PCAdata_row);
    end

    PCAdata(isnan(PCAdata)) = 0;
    
    [coeff, score, latent, ~, explained, mu] = pca(PCAdata);
    coeffients_50_182(i,:) = coeff(:,1)';
end

%% Create time color gradient
cm = colormap(jet(15));

%% Plot
for j = 1:12
    plot(coeffients_50_182(j,:), 'Color', cm(j,:))
    hold on
end

numCycles = 10;
forEvery = 1;
startAt = 182;
PCAdata = [];
%% Generate data for PCA input
for x = 1:numBat
    PCAdata_row = [];
    for j = 1:forEvery:numCycles
        cycle = j+startAt;
        PCAdata_row = [PCAdata_row, ...
            batch_train(x).cycles(cycle).discharge_dQdVvsV.dQdV(1,:)];
    end
    PCAdata = vertcat(PCAdata, PCAdata_row);
end

%% weird 10 cycle section
PCAdata(isnan(PCAdata)) = 0;

[coeff, score, latent, ~, explained, mu] = pca(PCAdata);

plot(coeff(:,1)', 'Color', cm(13,:))
hold on

for i = 14:15
    PCAdata = [];
    startAt = 50 + (i-1)*12;

    %% Generate data for PCA input
    for x = 1:numBat
        PCAdata_row = [];
        for j = 1:forEvery:numCycles
            cycle = j+startAt;
            PCAdata_row = [PCAdata_row, ...
                batch_train(x).cycles(cycle).discharge_dQdVvsV.dQdV(1,:)];
        end
        PCAdata = vertcat(PCAdata, PCAdata_row);
    end

    PCAdata(isnan(PCAdata)) = 0;
    
    [coeff, score, latent, ~, explained, mu] = pca(PCAdata);
    plot(coeff(:,1)', 'Color', cm(i,:))
end

ylim([-0.1 0.1])
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
legend('Cycle 51-62', 'Cycle 63-74', 'Cycle 75-86', ...
    'Cycle 87-98', 'Cycle 99-110', 'Cycle 111-122', 'Cycle 123-134', ...
    'Cycle 135-146', 'Cycle 147-158', 'Cycle 159-170', ...
    'Cycle 171-182', 'Cycle 183-192', 'Cycle 193-206', ...
    'Cycle 207-128', 'Location', 'Northeastoutside')
% colorbar('Ticks', [0:(1/6):1], 'TickLabels', {'Cycle 51-62', 'Cycle 63-74', 'Cycle 75-86', ...
%     'Cycle 87-98', 'Cycle 99-110', 'Cycle 111-122', 'Cycle 123-134', ...
%     'Cycle 135-146', 'Cycle 147-158', 'Cycle 159-170', ...
%     'Cycle 171-182', 'Cycle 183-192', 'Cycle 193-206', ...
%     'Cycle 207-128'})