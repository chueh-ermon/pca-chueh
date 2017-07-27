% Script that runs PCA on the dQ/dV differenced dataset, builds a
% predictive model from the results, and quantifies the fit of the model
%
% Zi Yang (zya@umich.edu) 07-2017
%
% Requires: files named 'train_test_partition.mat' or
% 'train_test_partition_b2.mat'

%% Edit this to change which batch to use %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
batch_num = 2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Can uncomment the remaider of this line to keep variables %%%%%%%%%%%%%%
clearvars -except batch_num batch batch_test batch_train ...
    held_out_unfinished batch_outlier; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;

%% Can comment this code if variables are already loaded %%%%%%%%%%%%%%%%%%
% if batch_num == 1
%     load train_test_partition.mat
%     % this .mat file contains 3 variables: batch_test, batch_train, and
%     % held_out_unfinished
% elseif batch_num == 2
%     load train_test_partition_b2.mat
%     % this .mat file contains 3 variables: batch_test, batch_train, and
%     % batch_outliers
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Can change these variables to generate different models %%%%%%%%%%%%%%%% 
startAt = 200;
less = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numBat = numel(batch_train);
PCAdata = [];

%% Generate data for PCA input
for i = 1:numBat
    PCAdata(i,:) = ...
        batch_train(i).cycles(startAt).discharge_dQdVvsV.dQdV(1,:) ...
        - batch_train(i).cycles(less).discharge_dQdVvsV.dQdV(1,:);
end

PCAdata(isnan(PCAdata)) = 0;

%% Create battery label from each battery's cycle life
bat_label = zeros(numBat,1);
for j = 1:numBat
    bat_label(j,1) = batch_train(j).last_cycle;
end

max_cycle = max(bat_label) + 1;
min_cycle = min(bat_label) - 1;
CM = colormap('jet');
close

%% Create battery color grade for future plots
batt_color_grade = zeros(numBat, 1);

for i = 1:numBat
batt_color_grade(i) = ceil((bat_label(i) - min_cycle) ./ ...
    (max_cycle - min_cycle) * 64);
end

%% Plot the PCAdata 
figure()
for i = 1:numBat
    color_ind = batt_color_grade(i);
    plot(batch_train(i).cycles(200).discharge_dQdVvsV.V, PCAdata(i,:), ...
        'Color', CM(color_ind,:))
    hold on
end
xlabel('Voltage')
ylabel(strcat('Difference in dQ/dV (cycle ', num2str(startAt), ...
    ' - cycle ', num2str(less),')'))
colormap('jet')
if batch_num == 1
    tick_range = 524:55:1074;
elseif batch_num == 2
    tick_range = 326:22.7:553;
end
cb = colorbar('TickLabels', tick_range);
ylabel(cb, 'Observed Cycle Life')
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
title('Difference in dQ/dV (cycle 200 - cycle 5)')

%% Perform PCA
[coeff, score, latent, ~, explained, mu] = pca(PCAdata);
path = strcat('/Users/ziyang/Desktop/2017_Chueh_Ermon_Research/', ...
    'pca-chueh-difference/batch', num2str(batch_num));
cd (char(path))
save(strcat('pcaResultsTest_', startAt, '-', less), 'coeff', 'score', ...
    'latent', 'explained', 'mu')


%% Plot percent variance explained
figure()
plot(explained,'o-')
ylabel('Percent Variance Explained')
xlabel('PC Index')
title('Percent Variance Explained')
file_name = char(strcat('PerVariExpTest_', string(startAt), '-', ...
    string(less)));
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
savefig(gcf, file_name);
print(gcf, file_name,'-dpng')


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
file_name = char(strcat('ScorevsBatteryTest_', string(startAt), '-', ...
    string(less)));
savefig(gcf, file_name);
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
file_name = char(strcat('ScorevsBattery12Test_', string(startAt), '-', ...
    string(less)));
savefig(gcf, file_name);
print(gcf, file_name,'-dpng')

%% Plot first principle component score using batt_color_range
figure('NumberTitle', 'off', 'Name', 'First Principal Component Score');
for i = 1:numBat
    color_ind = batt_color_grade(i);
    plot(i, score(i,1),'.','Color',CM(color_ind,:),'MarkerSize',16)
    hold on
end
title('First Principal Component Score')
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
colormap('jet')
cb = colorbar('TickLabels', [524:55:1074]);
ylabel(cb, 'Observed Cycle Life')
file_name = char(strcat('FirstPCScoreTest_', string(startAt), '-', ...
    string(less)));
savefig(gcf, file_name);
print(gcf, file_name,'-dpng')

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
file_name = char(strcat('ScorevsScore12Test_', string(startAt), '-', ...
    string(less)));
savefig(gcf, file_name);
print(gcf, file_name,'-dpng')


path = strcat('/Users/ziyang/Desktop/2017_Chueh_Ermon_Research/', ...
    'pca-chueh-difference/PredvsObs/batch', num2str(batch_num));
cd (char(path))

%% PCA regression
color_train = colormap(winter(numBat));
markers = {'+','o','*','.','x','s','d','^','v','>','<','p','h'};

X_ones = ones(numBat,1);
X = [score(:,1), X_ones];

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

if batch_num == 1
    linmin = 500;
    linmax = 1100;
elseif batch_num == 2
    linmin = 300;
    linmax = 600;
end

plot(linspace(linmin, linmax),linspace(linmin,linmax), 'k')
xlabel('Predicted Cycle Life')
ylabel('Current Cycle Life')
title(['Predictive Model from Difference PCA (dQdV at ', ...
    num2str(startAt), ' - dQdV at ', num2str(less),')']) 
hold on

%% Apply predictive model to test data
numTestBat = numel(batch_test);
color_test = colormap(autumn(numTestBat));
PCAtest = [];
for i = 1:numTestBat
    PCAtest(i,:) = ...
        (batch_test(i).cycles(startAt).discharge_dQdVvsV.dQdV(1,:) ...
        - batch_test(i).cycles(less).discharge_dQdVvsV.dQdV(1,:)) - mu;
end
scores = PCAtest * coeff(:,1);
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
name = strcat('PredCycle_CurrCycle_b1_', string(startAt), '-', ...
    string(less));
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
print(gcf,char(name),'-dpng')

%% Plot residuals train
figure()
scatter(1:numBat, r, 'b')
hold on
refline(0,0)
hold on
xlabel('Battery')
ylabel('Residual')
title(['Train Residuals for model based dQ/dV difference cycles ', ...
    num2str(startAt), '-', num2str(less)])
name = strcat('Residuals_', string(startAt), '-', string(less));
%set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
print(gcf,char(name),'-dpng')

%% Plot residuals test
r_test = bat_label_test - Y_test_pred;
figure()
scatter(1:numTestBat, r_test, 'r')
hold on
refline(0,0)
xlabel('Battery')
ylabel('Residual')
title(['Test Residuals for model based dQ/dV difference cycles ', ...
    num2str(startAt), '-', num2str(less)])
name = strcat('TestResiduals_', string(startAt), '-', string(less));
%set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
print(gcf,char(name),'-dpng')

%% Output r^2 and percent error
disp(['R^2: ', num2str(stats(1))])
