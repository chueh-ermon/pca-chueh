% Script that runs PCA on the batch 1 dQ/dV curves dataset, builds a 
% predictive model from the results, and quantifies the fit of the model
%
% Zi Yang (zya@umich.edu) 07-2017
%
% Requires: files named 'b1_b2_batchdata.mat'

clearvars -except batch1 batch2 batch2_heldout batch1_heldout
close all;

% this .mat file contains 3 variables:
% batch1, batch2, batch1_heldout, batch2_heldout
% load b1_b2_batchdata.mat

%% Variables to change to test different data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numCycles = 10;
forEvery = 1;
startAt = 150;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numBat = numel(batch1);
PCAdata = zeros(numBat - floor(numBat/4), numCycles*1000);

num_test = 0;
%% Generate data for PCA input
for i = 1:numBat
    if mod(i,4) ~= 0
        PCAdata_row = zeros(1, numCycles*1000);
        for j = 1:forEvery:numCycles
            cycle = j+startAt;
            PCAdata_row(1,((j-1)*1000+1:(j)*1000)) = batch1(i).cycles(cycle).discharge_dQdVvsV.dQdV(1,:);
        end
        PCAdata(i - num_test,:) = PCAdata_row;
    else
        num_test = num_test + 1;
    end
end

PCAdata(isnan(PCAdata)) = 0;

%% Create battery color grade for future plots
bat_label = zeros(numBat,1);
bat_label_train = zeros(numBat-num_test,1);
bat_label_test = [];
num_test = 0;
for j = 1:numBat
    bat_label(j,1) = batch1(j).last_cycle;
    if mod(j,4) ~= 0
        bat_label_train(j-num_test,1) = batch1(j).last_cycle;
    else
        bat_label_test = [bat_label_test, batch1(j).last_cycle];
        num_test = num_test + 1;
    end
end
bat_label_test = bat_label_test';

max_cycle = max(bat_label) + 1;
min_cycle = min(bat_label) - 1;
CM = colormap('jet');
close

batt_color_grade = zeros(numBat, 1);

for i = 1:numBat
batt_color_grade(i) = ceil((bat_label(i) - min_cycle) ./ ...
    (max_cycle - min_cycle) * 64);
end


%% Perform PCA
[coeff, score, latent, ~, explained, mu] = pca(PCAdata);


%% Save PCA data in correct folder
path = '/Users/ziyang/Desktop/2017_Chueh_Ermon_Research/pca-chueh-b1/';
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

%% Plot the PCAdata 
num_test = 0;
figure ()
for i = 1:numBat
    if mod(i,4) ~= 0
        color_ind = batt_color_grade(i);
        plot(1:1000*numCycles, PCAdata((i-num_test),:), ...
        'Color', CM(color_ind,:))
        hold on
    else
        num_test = num_test + 1;
    end
    
end

%% Plot the PCAdata minus mu and the PC1 score
num_test = 0;
figure ()
subplot(2,1,1)
% yyaxis left
for i = 1:numBat
    if mod(i,4) ~= 0
        color_ind = batt_color_grade(i);
        plot(1:1000*numCycles, PCAdata((i-num_test),:) - mu, ... %% changed
        'Color', CM(color_ind,:))
        hold on
    else
        num_test = num_test + 1;
    end
    
end

xlabel('PCAdata Index')
ylabel('dQ/dV (Ahs/V)')
title('PCA Input: dQ/dV Curves - mu')
colormap('jet')

% tick_range = min_cycle:((max_cycle - min_cycle)/10):max_cycle;
% cb = colorbar('TickLabels', tick_range);
% ylabel(cb, 'Observed Cycle Life')
set(gcf, 'Position', get(0,'Screensize'));

% figure()
subplot(2,1,2)
% yyaxis right
plot(1:1000*numCycles, coeff(:,1), 'k')
xlabel('PCAdata Index')
ylabel('Principal Component 1 Coefficient')
title('PCA Output: Principal Component 1 Coefficients')


%% Plot percent variance explained
figure()
plot(explained,'o-')
ylabel('Percent Variance Explained')
xlabel('PC Index')
title('Percent Variance Explained')
file_name = char(strcat('PerVariExpTest_', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles)));
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
% savefig(gcf, file_name);
print(gcf, file_name,'-dpng')


%% Plot score vs battery using batt_color_range
figure('NumberTitle', 'off', 'Name', 'Score vs Battery Index');
for j = 1:size(score,2)
    num_test = 0;
    subplot(8,4,j)
    hold on
    for i = 1:numBat
        if mod(i,4) ~= 0
            color_ind = batt_color_grade(i);
            plot(bat_label(i), score(i-num_test,j),'.','Color',CM(color_ind,:),'MarkerSize',16)
        else
            num_test = num_test + 1;
        end
    end
    xlabel('Battery Cycle Life')
    ylabel(['Score ', num2str(j)])
end
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
file_name = char(strcat('ScorevsBatteryTest_', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles)));
% savefig(gcf, file_name);
print(gcf, file_name,'-dpng')

%% Plot first 12 PC score vs battery using batt_color_range
figure('NumberTitle', 'off', 'Name', ...
    'Score vs Battery Index (first 12 PCs)');
for j = 1:12
    num_test = 0;
    subplot(3,4,j)
    hold on
    for i = 1:numBat
        if mod(i,4) ~= 0
            color_ind = batt_color_grade(i);
            plot(bat_label(i), score(i-num_test,j),'.','Color',CM(color_ind,:),'MarkerSize',16)
        else
            num_test = num_test + 1;
        end
    end
    xlabel('Battery Cycle Life')
    ylabel(['Score ', num2str(j)])
end
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
file_name = char(strcat('ScorevsBattery12Test_', string(startAt), ...
    '_', string(forEvery), '_', string(numCycles)));
% savefig(gcf, file_name);
print(gcf, file_name,'-dpng')

% %% Plot first principle component score using batt_color_range
% figure('NumberTitle', 'off', 'Name', 'First Principal Component Score');
% for i = 1:numBat
%     color_ind = batt_color_grade(i);
%     plot(i, score(i,1),'.','Color',CM(color_ind,:),'MarkerSize',16)
%     hold on
% end
% title('First Principal Component Score')
% set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
% file_name = char(strcat('FirstPCScoreTest_', string(startAt), '_', ...
%     string(forEvery), '_', string(numCycles)));
% % savefig(gcf, file_name);
% print(gcf, file_name,'-dpng')

%% Plot score vs score for first 12 PCs
figure()
for j = 1:12
    subplot(3,4,j)
    hold on
    num_test = 0;
    for i = 1:numBat
        if mod(i,4) ~= 0
            color_ind = batt_color_grade(i);
            plot(score(i-num_test,j),score(i-num_test,j+1),'.','Color',CM(color_ind,:), ...
                'MarkerSize',16)
            xlabel(['Score ',num2str(j)])
            ylabel(['Score ',num2str(j+1)])
        else
            num_test = num_test + 1;
        end
    end
end
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
file_name = char(strcat('ScorevsScore12Test_', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles)));
% savefig(gcf, file_name);
print(gcf, file_name,'-dpng')


%% PCA regression
color_train = colormap(winter(numBat - num_test));
markers = {'+','o','*','.','x','s','d','^','v','>','<','p','h'};

X_ones = ones(numBat - num_test,1);
X = [score(:,1), X_ones];

[b,bint,r,rint,stats] = regress(bat_label_train, X);

Y_pred = X * b;

rmse_train = sqrt(mean((bat_label_train - Y_pred) .^2));

disp(['RMSE Train: ', num2str(rmse_train)])

train_pererror = (abs(bat_label_train - Y_pred) ./ Y_pred);
train_allpererr = mean(train_pererror);

disp(['Percent error for Train: ', num2str(train_allpererr * 100), '%'])

num_test = 0;
figure()
for i = 1:numBat
    if mod(i, 4) ~= 0
        plot(Y_pred(i-num_test), bat_label(i), markers{mod(i-num_test,numel(markers))+1}, ...
        'Color', color_train(i-num_test,:))
    hold on
    else
        num_test = num_test + 1;
    end
end
hold on
plot(linspace(400, 1800),linspace(400,1800), 'k')
xlabel('Predicted Cycle Life')
ylabel('Current Cycle Life')
title(['Cycle ' num2str(startAt+1), '-', num2str(startAt+numCycles)]) 
hold on

%% Apply model to test data
numTestBat = num_test;
color_test = colormap(autumn(numTestBat));
PCAtest = zeros(numTestBat, 1000*numCycles);
for i = 1:numTestBat
    PCAtest_row = zeros(1,1000*numCycles);
    for j = 1:forEvery:numCycles
        cycle = j+startAt;
        PCAtest_row(1,(j-1)*1000+1:(j)*1000) = batch1(i*4).cycles(cycle).discharge_dQdVvsV.dQdV(1,:);
    end
    centeredPCAtest = PCAtest_row - mu;
    PCAtest(i,:) = centeredPCAtest;
end
test_score = PCAtest * coeff(:,1);
X_ones = ones(numTestBat,1);
X_test = [test_score,X_ones];

Y_test_pred = X_test * b;

rmse_test = sqrt(mean((bat_label_test - Y_test_pred) .^2));
disp(['RMSE Test: ', num2str(rmse_test)])

test_pererror = (abs(bat_label_test - Y_test_pred) ./ Y_test_pred);
test_allpererr = mean(test_pererror);

disp(['Percent error for Test: ', num2str(test_allpererr * 100), '%'])

for i = 1:numTestBat
    plot(Y_test_pred(i), bat_label(i*4), ...
        markers{mod(i,numel(markers))+1}, 'Color', color_test(i,:))
    hold on
end

num_test = 0;
policy_names = {};
for j = 1:numBat
    if mod(j,4) ~= 0
        policy_names = [policy_names,batch1(j-num_test).policy_readable];
    else
        num_test = num_test + 1;
    end
end
policy_names = [policy_names,'Test Policies Below'];
for i = 1:numTestBat
    policy_names = [policy_names,batch1(i*4).policy_readable];
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
ylim([-300 300])
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
ylim([-300 300])
title(['Test Residuals for model based on cycles ', num2str(startAt+1), ...
    '-', num2str(numCycles+startAt)])
name = strcat('TestResiduals_', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles));
%set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
print(gcf,char(name),'-dpng')

%% Output r^2 and percent error
disp(['R^2: ', num2str(stats(1))])


%% Apply this model to Batch2
b2_numBat = numel(batch2);
b2_PCAdata = zeros(b2_numBat, 1000*numCycles);
for i = 1:b2_numBat
    PCAdata_row = zeros(1, numCycles*1000);
    for j = 1:forEvery:numCycles
        cycle = j+startAt;
        PCAdata_row(1,((j-1)*1000+1:(j)*1000)) = ...
            batch2(i).cycles(cycle).discharge_dQdVvsV.dQdV(1,:);
    end
    PCAdata_row = PCAdata_row - mu;
    b2_PCAdata(i,:) = PCAdata_row;
end

b2_scores = b2_PCAdata * coeff(:,1);
X_ones = ones(b2_numBat, 1);
b2_X = [b2_scores,X_ones];
b2_Y = b2_X * b;

b2_bat_label = zeros(b2_numBat, 1);
for i = 1:b2_numBat
    b2_bat_label(i,1) = batch2(i).last_cycle;
end

figure()
scatter(b2_Y, b2_bat_label)
hold on
plot(linspace(0, 900),linspace(0,900), 'k')
set(gcf, 'Position', get(0,'Screensize'));
xlabel('Predicted Cycle Life')
ylabel('Observed Cycle Life')
title('Batch 2 predictions using derived model')


b2_rmse = sqrt(mean((b2_bat_label - b2_Y) .^ 2));
disp(['RMSE for Batch2: ', num2str(b2_rmse)])

b2_pererror = (abs(b2_bat_label - b2_Y) ./ b2_Y);
b2_allpererr = mean(b2_pererror);

disp(['Percent error for batch 2 predictions: ', num2str(b2_allpererr*100), '%'])

b2_policy_names = [];
for i = 1:b2_numBat
    b2_policy_names = [b2_policy_names, batch2(i).policy_readable];
end
