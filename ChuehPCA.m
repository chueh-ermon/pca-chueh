clearvars -except batch; 
close all; clc

%load 2017-05-12_batchdata_final.mat

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

PCAdata_smooth = zeros(size(PCAdata));
for i = 1:numBat
    PCAdata_smooth(i,:) = smooth(PCAdata(i,:));
end

%% create battery color grade for future plots
bat_label = zeros(numBat,1);
for j = 1:numBat
    bat_label(j,1) = numel(batch(j).cycles); % TODO: change to last cycle before degraded
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

%% Plot percent variance explained
%figure('NumberTitle', 'off', 'Name', 'Per Variance Explained dQdV 201 to 212');
plot(explained,'o-')
ylabel('Percent Variance Explained')
xlabel('PC Index')
title('Percent Variance Explained (dQdV for cycles 201-212)')

%% Plot score vs battery using batt_color_range
figure('NumberTitle', 'off', 'Name', 'Score vs Battery Index dQdV 201 to 212');
for j = 1:size(score,2)
    subplot(5,9,j)
    hold on
    for i = 1:numBat
        color_ind = batt_color_grade(i);
        plot(i, score(i,j),'.','Color',CM(color_ind,:),'MarkerSize',16)
    end
    xlabel('Battery Index')
    ylabel(['Score ', num2str(j)])
end

%% Plot first 12 PC score vs battery using batt_color_range
figure('NumberTitle', 'off', 'Name', 'Score vs Battery Index dQdV 201 to 212');
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

%% Plot first principle component score using batt_color_range
figure('NumberTitle', 'off', 'Name', 'First Principal Component Score 201 to 212');
for i = 1:numBat
    color_ind = batt_color_grade(i);
    plot(i, score(i,2),'.','Color',CM(color_ind,:),'MarkerSize',16)
    hold on
end
title('First Principal Component Score')

%% Plot score vs score for first 12 PCs
figure()
for j = 1:numCycles
    subplot(3,4,j)
    hold on
    for i = 1:numBat
        color_ind = batt_color_grade(i);
        plot(score(i,j),score(i,j+1),'.','Color',CM(color_ind,:),'MarkerSize',16)
        xlabel(['Score ',num2str(j)])
        ylabel(['Score ',num2str(j+1)])
    end
end

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
    plot(1:12000, PC)
    hold on
end
legend('PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5', 'PC 6')%, 'PC 7', 'PC 8', 'PC 9', 'PC 10', 'PC 11', 'PC 12')

figure('NumberTitle', 'off', 'Name', '1st PC Coefficients over PCAdata dQdV 201 to 212');
for j = 1:numCycles
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
for j = 1:numCycles
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


%{
clear; close all; clc

load ChenLiProcessed.mat

numCyc = 100; % number of cycles to be used in analysis
numBat = 24; % total number of batteries in dataset
lenCyc = 1050; % length of each dQdV dataset

%create dataset based on the difference between cycle 1 and cycle 100 dQdV
PCAdata = zeros(numBat,lenCyc);
for i = 1:numBat
    PCAdata(i,:) = master_dQdV{i}(:,500) - master_dQdV{i}(:,1);
end

PCAdata_smooth = zeros(size(PCAdata));
for i = 1:numBat
    PCAdata_smooth(i,:) = smooth(PCAdata(i,:));
end

bat_label = zeros(numBat,1);
for j = 1:numBat
    
    bat_label(j) = master_Q{j}(550);
    
end


max_Q = 0.9;
min_Q = 0.45;
colormap('jet')
CM = colormap('jet');
feat_check = zeros(numBat,4);
feat_check(:,1) = mean(PCAdata(:,868:1030),2);
[feat_check(:,2),feat_check(:,3)] = max(PCAdata(:,415:505),[],2);
%feat_check(:,3) = PCAdata(:,576) - PCAdata(:,556);
%feat_check(:,4) = mean(PCAdata(:,586:687),2);
hFig = figure()
for i = 1:3

for j = 1:numBat
     subplot(1,3,i)
     hold on
     color_ind = ceil((bat_label(j) - min_Q)./(max_Q - min_Q)*64);
     plot(j,feat_check(j,i),'.','Color',CM(color_ind,:),'MarkerSize',15)
        
end

end


%suptitle('Relative Capacity and Internal Capacity: All cycles')
h = colorbar;
set(h,'Position',[0.92, 0.075 0.01 0.8])
h.Label.String = 'Relative Capacity';
h.TickLabels = linspace(min_Q,max_Q,11);
set(hFig,'position',get(0,'Screensize'))
print('AllData_Diff','-dpng')
print('AllData_Diff','-depsc')

max_Q = 0.9;
min_Q = 0.45;
colormap('jet')
CM = colormap('jet');
hFig=figure();
for j = 1:numBat
    if j == 1
        surf(peaks,colormap('jet'))
        cla
    end
    subplot(1,1,1)    
    hold on
    color_ind = ceil((bat_label(j) - min_Q)./(max_Q - min_Q)*64);
        plot(xVoltage{j},PCAdata_smooth(j,:),'Color',CM(color_ind,:))
 
    %ylim([0,25])
    xlim([3.2,4.4])
    xlabel('Voltage (V)')
    ylabel('Incremental Capacity (mAh/V)')

end
%suptitle('Relative Capacity and Internal Capacity: All cycles')
h = colorbar;
set(h,'Position',[0.92, 0.075 0.01 0.8])
h.Label.String = 'Relative Capacity';
h.TickLabels = linspace(min_Q,max_Q,11);
set(hFig,'position',get(0,'Screensize'))
print('AllData_DiffSmooth','-dpng')
print('AllData_DiffSmooth','-depsc')

PCAdata(isnan(PCAdata)) = 0;

[coeff, score, latent, ~, explained, mu] = pca(PCAdata);

figure()
plot(explained,'o-')
ylabel('Percent Variance Explained')
xlabel('PC Index')

figure()
for j = 1:size(score,2)
    subplot(4,6,j)
    hold on
    for i = 1:numBat
        color_ind = ceil((bat_label(i) - min_Q)./(max_Q - min_Q)*64);
        plot(i, score(i,j),'.','Color',CM(color_ind,:),'MarkerSize',16)
    end
    xlabel('Battery Index')
    ylabel(['Score ', num2str(j)])
end


figure()
for j = 1:12
    subplot(3,4,j)
    hold on
    for i = 1:numBat
        color_ind = ceil((bat_label(i) - min_Q)./(max_Q - min_Q)*64);
        plot(score(i,j),score(i,j+1),'.','Color',CM(color_ind,:),'MarkerSize',16)
        xlabel(['Score ',num2str(j)])
        ylabel(['Score ',num2str(j+1)])
    end
    
end

color_stor = zeros(numBat,3);
for i = 1:numBat
    color_ind = ceil((bat_label(i) - min_Q)./(max_Q - min_Q)*64);
    color_stor(i,:) = CM(color_ind,:);
end

figure()
scatter3(score(:,1),score(:,2),score(:,3),25*ones(numBat,1),color_stor)

[B1,BINT1,R1,RINT1,STATS1] = regress(bat_label,[score(:,1),ones(numBat,1)]);

rmse1 = sqrt(mean(R1.^2))

[B2,BINT2,R2,RINT2,STATS2] = regress(bat_label,[score(:,1:2),ones(numBat,1)]);

rmse2 = sqrt(mean(R2.^2))

[B3,BINT3,R3,RINT3,STATS3] = regress(bat_label,[feat_check,ones(numBat,1)]);

rmse3 = sqrt(mean(R3.^2))

[B4,BINT4,R4,RINT4,STATS4] = regress(bat_label,[feat_check(:,1:2),ones(numBat,1)]);

rmse4 = sqrt(mean(R4.^2))

figure()
plot(xVoltage{1},coeff(:,1))
hold on
plot(xVoltage{1},coeff(:,2))
legend('Coeff 1','Coeff 2')
%}