clearvars -except batch batch_test batch_train held_out_unfinished; 
close all; clc

%load 2017-05-12_batchdata_modified.mat
%load train_test_partition.mat

numBat = numel(batch_train);
numCycles = 10;
forEvery = 1;
PCAdata = [];
startAt = 182;

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

PCAdata_smooth = zeros(size(PCAdata));
for i = 1:numBat
    PCAdata_smooth(i,:) = smooth(PCAdata(i,:));
end

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
batt_color_grade(i) = ceil((bat_label(i) - min_cycle) ./ (max_cycle - min_cycle) * 64);
end

%% Perform PCA
[coeff, score, latent, ~, explained, mu] = pca(PCAdata);
% path = strcat('/Users/ziyang/Desktop/2017_Chueh_Ermon_Research/pca-chueh-images/dQdV_', string(startAt), '_', string(forEvery), '_', string(numCycles));
% cd (char(path))
% save(strcat('pcaResultsTest_', string(startAt), '_', string(forEvery), '_', string(numCycles)), 'coeff', 'score', 'latent', 'explained', 'mu')

%{
%% Plot percent variance explained
plot(explained,'o-')
ylabel('Percent Variance Explained')
xlabel('PC Index')
title('Percent Variance Explained')
file_name = char(strcat('PerVariExpTest_', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles)));
%set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
savefig(gcf, file_name);
print(gcf, file_name,'-dpng')


%% Plot score vs battery using batt_color_range
figure('NumberTitle', 'off', 'Name', 'Score vs Battery Index');
for j = 1:size(score,2)
    subplot(5,6,j)
    hold on
    for i = 1:numBat
        color_ind = batt_color_grade(i);
        plot(i, score(i,j),'.','Color',CM(color_ind,:),'MarkerSize',16)
    end
    xlabel('Battery Index')
    ylabel(['Score ', num2str(j)])
end
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
file_name = char(strcat('ScorevsBatteryTest_', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles)));
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
file_name = char(strcat('ScorevsBattery12Test_', string(startAt), ...
    '_', string(forEvery), '_', string(numCycles)));
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
file_name = char(strcat('FirstPCScoreTest_', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles)));
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
file_name = char(strcat('ScorevsScore12Test_', string(startAt), '_', ...
    string(forEvery), '_', string(numCycles)));
savefig(gcf, file_name);
print(gcf, file_name,'-dpng')
%}

path = strcat('/Users/ziyang/Desktop/2017_Chueh_Ermon_Research/pca-chueh-images/PredvsObs');
cd (char(path))

%% PCA regression
color_train = colormap(winter(numBat));
markers = {'+','o','*','.','x','s','d','^','v','>','<','p','h'};

X_ones = ones(numBat,1);
X = [score(:,1), X_ones];

[b,bint,r,rint,stats] = regress(bat_label, X);

Y_pred = X * b;

figure()
for i = 1:numBat
    plot(Y_pred(i), bat_label(i), markers{mod(i,numel(markers))+1}, ...
        'Color', color_train(i,:))
    hold on
end
hold on
plot(linspace(500, 1100),linspace(500,1100), 'k')
xlabel('Predicted Cycle Number')
ylabel('Current Cycle Number')
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
scores = PCAtest * coeff(:,1);
X_ones = ones(numTestBat,1);
X_test = [scores,X_ones];

bat_label_test = zeros(numTestBat,1);
for j = 1:numTestBat
    bat_label_test(j,1) = batch_test(j).last_cycle; % TODO: change to last cycle before degraded
end

Y_test_pred = X_test * b;
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

close all
%{
%% Graph the first principle component on top of the PCAdata
figure()
for i = 1:numBat
    yyaxis left
    color_ind = batt_color_grade(i);
    plot(1:numCycles*1000, PCAdata(i,:), '-', 'Color', CM(color_ind,:))
    hold on
end
ylabel('dQ/dV (Ahs/V)')
yyaxis right
PC1 = coeff(:,1)';
plot(1:numCycles*1000, PC1, 'Color', 'k')
ylabel('Principal Component 1 Coefficients')
title('First Principal Component Coefficients over PCAdata')

figure()
for i = 1:numBat
    yyaxis left
    color_ind = batt_color_grade(i);
    plot(1:numCycles*1000, PCAdata(i,:), '-', 'Color', CM(color_ind,:))
    hold on
end
ylabel('dQ/dV (Ahs/V)')
yyaxis right
PC2 = coeff(:,2)';
plot(1:numCycles*1000, PC2, 'Color', 'k')
ylabel('Principal Component 2 Coefficients')
title('Second Principal Component Coefficients over PCAdata')

figure()
% PC1 = coeff(:,1)';
% plot(1:numCycles*1000, PC1, 'Color', 'r')
% hold on
% PC2 = coeff(:,2)';
% plot(1:numCycles*1000, PC2, 'Color', 'b')
% legend('Principal Component 1', 'Principal Component 2')
for j = 1:6
    PC = coeff(:,j)';
    plot(1:numCycles*1000, PC)
    hold on
end
legend('PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5', 'PC 6')%, 'PC 7', 'PC 8', 'PC 9', 'PC 10', 'PC 11', 'PC 12')

figure('NumberTitle', 'off', 'Name', '1st PC Coefficients over PCAdata dQdV 201 to 212');
for j = 1:12
    subplot(3,4,j)
    yyaxis left
    for i = 1:numBat
    color_ind = batt_color_grade(i);
    plot(1:1000, PCAdata(i,(j*1000 - 999):(j*1000)), '-', 'Color', CM(color_ind,:))
    hold on
    end
    xlabel(['Cycle', j])
    ylabel('dQ/dV (Ahs/V)')
    yyaxis right
    ylabel('Principal Component 1 Coefficients')
    PC1 = coeff((j*1000 - 999):(j*1000),1)';
    plot(1:1000, PC1, 'Color', 'k', 'LineWidth', 1.5)
end

figure('NumberTitle', 'off', 'Name', '2nd PC Coefficients over PCAdata dQdV 201 to 212');
for j = 1:12
    subplot(3,4,j)
    for i = 1:numBat
    color_ind = batt_color_grade(i);
    plot(1:1000, PCAdata(i,(j*1000 - 999):(j*1000)), 'Color', CM(color_ind,:))
    hold on
    end
    xlabel(['Cycle', j])
    ylabel('dQ/dV (Ahs/V)')
    yyaxis right
    ylabel('Principal Component 2 Coefficients')
    PC2 = coeff((j*1000 - 999):(j*1000),2)';
    plot(1:1000, PC1, 'Color', 'k', 'LineWidth', 1.5)
end
%}