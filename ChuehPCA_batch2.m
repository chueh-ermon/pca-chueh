% Script that runs PCA on the batch 2 dQ/dV curves dataset, builds a 
% predictive model from the results, and quantifies the fit of the model
%
% Zi Yang (zya@umich.edu) 07-2017
%
% Requires: files named 'train_test_partition_b2.mat'

clearvars -except batch batch_test batch_train batch_outlier; 
close all;

% this .mat file contains 3 variables: batch_test, batch_train, and
% batch_outliers
% load train_test_partition_b2.mat

numBat = numel(batch_train);
PCAdata = [];


%% Variables to change to test different data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numCycles = 10;
forEvery = 1;
startAt = 180;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Generate data for PCA input
for i = 1:numBat
    PCAdata_row = [];
    for j = 1:forEvery:numCycles
        cycle = j+startAt;
        PCAdata_row = [PCAdata_row, ...
            batch_train(i).cycles(cycle).discharge_dQdVvsV.dQdV(1,:)];
    end
    PCAdata = vertcat(PCAdata, PCAdata_row);
end

PCAdata(isnan(PCAdata)) = 0;

%% Create battery color grade for future plots
bat_label = zeros(numBat,1);
for j = 1:numBat
    bat_label(j,1) = batch_train(j).last_cycle;
end

max_cycle = max(bat_label) + 1;
min_cycle = min(bat_label) - 1;
CM = colormap('jet');

batt_color_grade = zeros(numBat, 1);

for i = 1:numBat
batt_color_grade(i) = ceil((bat_label(i) - min_cycle) ./ ...
    (max_cycle - min_cycle) * 64);
end

%% Perform PCA
[coeff, score, latent, ~, explained, mu] = pca(PCAdata);

%% Save PCA data in correct folder
path = '/Users/ziyang/Desktop/2017_Chueh_Ermon_Research/pca-chueh-b2/';
cd (char(path))

new_dir = strcat('dQdV_', num2str(startAt), '_', ...
    num2str(forEvery), '_', num2str(numCycles));
folder_path = strcat(path, new_dir);

if exist(new_dir, 'dir') == 0
    mkdir(new_dir);
end

cd (folder_path)
save(strcat('pcaResultsTest_', string(startAt), '_', string(forEvery), ...
    '_', string(numCycles)), 'coeff', 'score', 'latent', 'explained', 'mu')

%% Plot percent variance explained
plot(explained,'o-')
ylabel('Percent Variance Explained')
xlabel('PC Index')
title('Percent Variance Explained')
file_name = char(strcat('PerVariExpTest_B2_', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles)));
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
% savefig(gcf, file_name);
print(gcf, file_name,'-dpng')

%% Plot the PCAdata 
figure()
for i = 1:numBat
    color_ind = batt_color_grade(i);
    plot(1:numCycles*1000, PCAdata(i,:), ...
        'Color', CM(color_ind,:))
    hold on
end
xlabel('PCAdata Index')
ylabel('dQ/dV (Ahs/V)')
colormap('jet')
tick_range = 326:22.7:553;
cb = colorbar('TickLabels', tick_range);
ylabel(cb, 'Observed Cycle Life')
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
title('PCAdata ')

%% Plot score vs battery using batt_color_range
figure('NumberTitle', 'off', 'Name', 'Score vs Battery Index');
for j = 1:size(score,2)
    subplot(6,6,j)
    hold on
    for i = 1:numBat
        color_ind = batt_color_grade(i);
        plot(i, score(i,j),'.','Color',CM(color_ind,:),'MarkerSize',16)
    end
    xlabel('Battery Index')
    ylabel(['Score ', num2str(j)])
end
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
file_name = char(strcat('ScorevsBatteryTest_B2_', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles)));
% savefig(gcf, file_name);
print(gcf, file_name,'-dpng')

%% Plot first 12 PC score vs battery using batt_color_range
figure('NumberTitle', 'off', 'Name', ...
    'Score vs Battery Index (first 12 PCs)');
for j = 1:12
    subplot(3,4,j)
    hold on
    for i = 1:numBat
        color_ind = batt_color_grade(i);
        plot(i, score(i,j),'.','Color',CM(color_ind,:),'MarkerSize',16)
    end
    xlabel('Battery Index')
    ylabel(['Score ', num2str(j)])
end
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
file_name = char(strcat('ScorevsBattery12Test_B2_', string(startAt), ...
    '_', string(forEvery), '_', string(numCycles)));
% savefig(gcf, file_name);
print(gcf, file_name,'-dpng')

%{
%% Plot first principle component score using batt_color_range
figure('NumberTitle', 'off', 'Name', 'First Principal Component Score');
for i = 1:numBat
    color_ind = batt_color_grade(i);
    plot(i, score(i,1),'.','Color',CM(color_ind,:),'MarkerSize',16)
    hold on
end
title('First Principal Component Score')
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
file_name = char(strcat('FirstPCScoreTest_B2_', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles)));
% savefig(gcf, file_name);
print(gcf, file_name,'-dpng')
%}
%% Plot score vs score for first 12 PCs
figure()
for j = 1:12
    subplot(3,4,j)
    hold on
    for i = 1:numBat
        color_ind = batt_color_grade(i);
        plot(score(i,j),score(i,j+1),'.','Color',CM(color_ind,:), ...
            'MarkerSize',16)
        xlabel(['Score ',num2str(j)])
        ylabel(['Score ',num2str(j+1)])
    end
end
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
file_name = char(strcat('ScorevsScore12Test_B2_', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles)));
% savefig(gcf, file_name);
print(gcf, file_name,'-dpng')

%% PCA regression
color_train = colormap(winter(numBat));
markers = {'+','o','*','.','x','s','d','^','v','>','<','p','h'};

X_ones = ones(numBat,1);
X = [score(:,15), X_ones]; %% changed to score 15

[b,bint,r,rint,stats] = regress(bat_label, X);

Y_pred = X * b;

rmse_train = sqrt(mean((bat_label - Y_pred) .^2));

disp(['RMSE Train: ', num2str(rmse_train)])

figure()
for i = 1:numBat
    plot(Y_pred(i), bat_label(i), markers{mod(i,numel(markers))+1}, ...
        'Color', color_train(i,:))
    hold on
end
hold on
plot(linspace(200, 700),linspace(200,700), 'k')
xlabel('Predicted Cycle Life')
ylabel('Current Cycle Life')
title(['Cycle ' num2str(startAt+1), '-', num2str(startAt+numCycles)]) 
hold on

%% Apply model to test data
numTestBat = numel(batch_test);
color_test = colormap(autumn(numTestBat));
PCAtest = [];
for i = 1:numTestBat
    PCAtest_row = [];
    for j = 1:forEvery:numCycles
        cycle = j+startAt;
        PCAtest_row = [PCAtest_row, ...
            batch_test(i).cycles(cycle).discharge_dQdVvsV.dQdV(1,:)];
    end
    centeredPCAtest = PCAtest_row - mu;
    PCAtest = vertcat(PCAtest, centeredPCAtest);
end
scores = PCAtest * coeff(:,15); %% change to score 15
X_ones = ones(numTestBat,1);
X_test = [scores,X_ones];

bat_label_test = zeros(numTestBat,1);
for j = 1:numTestBat
    bat_label_test(j,1) = batch_test(j).last_cycle;
end

Y_test_pred = X_test * b;

rmse_test = sqrt(mean((bat_label_test - Y_test_pred) .^2));

disp(['RMSE Test: ', num2str(rmse_test)])

for i = 1:numTestBat
    plot(Y_test_pred(i), bat_label_test(i), ...
        markers{mod(i,numel(markers))+1}, 'Color', color_test(i,:))
    hold on
end

policy_names = {};
for j = 1:numBat
    policy_names = [policy_names, batch_train(j).policy_readable];
end
policy_names = [policy_names, 'Test Policies Below'];
for j = 1:numTestBat
    policy_names = [policy_names, batch_test(j).policy_readable];
end
legL = legend(policy_names,'Location','NortheastOutside');
%legR = legend(test_names,'Location','NorthwestOutside');
name = strcat('PredCycle_CurrCycle', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles));
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
print(gcf,char(name),'-dpng')

%% Plot residuals train
figure()
scatter(Y_pred, r, 'b')
hold on
refline(0,0)
hold on
xlabel('Predicted Cycle Life')
ylabel('Residual')
title(['Train Residuals for model based on cycles ', ...
    num2str(startAt+1), '-', num2str(numCycles+startAt)])
name = strcat('Residuals_', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles));
%set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
print(gcf,char(name),'-dpng')

%% Plot residuals test
r_test = bat_label_test - Y_test_pred;
figure()
scatter(Y_test_pred, r_test, 'r')
hold on
refline(0,0)
xlabel('Predicted Cycle Life')
ylabel('Residual')
title(['Test Residuals for model based on cycles ', num2str(startAt+1), ...
    '-', num2str(numCycles+startAt)])
name = strcat('TestResiduals_', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles));
%set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
print(gcf,char(name),'-dpng')

%% Output r^2 and percent error
disp(['R^2: ', num2str(stats(1))])
