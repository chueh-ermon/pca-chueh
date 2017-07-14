function [] = chuehDataVisual(batch, batch_date)

numBat = 46;

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
%% Plot dQdV curve at cycle 200 using the batt_color_range
figure()
for i = 1:numBat
    color_ind = batt_color_grade(i);
    plot(batch(i).cycles(200).discharge_dQdVvsV.V, ...
        batch(i).cycles(200).discharge_dQdVvsV.dQdV,'Color', CM(color_ind,:))
    hold on
end

%% Plot Remaining Capacity vs Cycle, raw and normalized - batt_color_range
figAbsolute = figure();
figNormalized = figure();

for i = 1:numBat
    color_ind = batt_color_grade(i);
    figure(figAbsolute)
    plot(batch(i).summary.cycle, batch(i).summary.QDischarge, ...
        'Color', CM(batt_color_grade(i),:), 'MarkerSize',15);
    hold on
    plot(batch(i).summary.cycle, batch(i).summary.QCharge, ...
        'Color', CM(batt_color_grade(i,1),:), 'MarkerSize',15);
    hold on
    
end
for i = 1:numBat
    color_ind = batt_color_grade(i);
    figure(figNormalized)
    max_Qd = max(batch(i).summary.QDischarge);
    max_Qc = max(batch(i).summary.QCharge);
    plot(batch(i).summary.cycle, batch(i).summary.QDischarge ./ max_Qd, ...
        'Color', CM(batt_color_grade(i),:), 'MarkerSize',15);
    hold on
    plot(batch(i).summary.cycle, batch(i).summary.QCharge ./ max_Qc, ...
        'Color', CM(batt_color_grade(i,1),:), 'MarkerSize',15);
    hold on
end

figure(figNormalized)
xlabel('Cycle')
ylabel('Normalized Remaining Capacity (%)')
ylim([.8,1])
figure(figAbsolute)
xlabel('Remaing Capacity (Ah)')
ylabel('Cycle')
ylim([0.88,1.1])


end