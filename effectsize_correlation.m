%% ============================================================
% 1. DATA LOADING (FULL 80-TRIAL EXTRACTION)
% ============================================================
clear; clc; close all;

data_root = ''; % add root
subjects_to_use = 1:70; 
all_data = table();  

for s = subjects_to_use
    subj_name = sprintf('sub-%02d', s);
    feature_path = fullfile(data_root, subj_name, 'processed_EEG', 'EEG_features.mat');
    
    if exist(feature_path, 'file')
        load(feature_path); 
        
        % Ensure DeltaB calculation (Shift in Response)
        resp_vals = double(TrialTable.Response);
        % DeltaB: difference from previous trial within this subject
        TrialTable.DeltaB = [0; abs(diff(resp_vals))]; 
        
        all_data = [all_data; TrialTable];
        fprintf('Loaded %d trials from %s\n', height(TrialTable), subj_name);
    end
end

% Filter out first trials and ensure we have the full trial set
analysis_data = all_data(all_data.DeltaB > 0, :);
fprintf('Total usable trials for analysis: %d\n', height(analysis_data));

%% ============================================================
% 2. STATISTICAL ANALYSIS & PLOTTING
% ============================================================
mid_pt = median(analysis_data.DeltaB);
high_idx = analysis_data.DeltaB > mid_pt;
low_idx  = analysis_data.DeltaB <= mid_pt;

%% --- FIGURE 1: ERP ANALYSIS (P300 & N200) ---
figure('Name', 'ERP Significance Analysis', 'Color', 'w', 'Position', [100 100 900 400]);
erp_feats = {'P300', 'N200'};
for i = 1:2
    subplot(1, 2, i); hold on;
    feat = erp_feats{i};
    y = analysis_data.(feat);
    x = analysis_data.DeltaB;
    [r, p] = corr(x, y, 'Type', 'Spearman');
    d = calculate_cohen_d(y(high_idx), y(low_idx));
    scatter(x, y, 30, [0.2 0.4 0.6], 'filled', 'MarkerFaceAlpha', 0.5);
    lsline;
    title(sprintf('%s: r=%.2f (%s)\nd = %.2f', feat, r, p_to_star(p), d));
    xlabel('\Delta Belief Magnitude'); ylabel('Amplitude (\muV)');
    set(gca, 'Box', 'off', 'TickDir', 'out');
end

%% --- FIGURE 2: BAND POWER (DELTA to GAMMA) ---
figure('Name', 'Full Spectral Significance Analysis', 'Color', 'w', 'Position', [100 100 1100 700]);
power_feats = {'delta', 'theta', 'alpha', 'beta', 'gamma'};
for i = 1:length(power_feats)
    subplot(2, 3, i); hold on;
    feat = power_feats{i};
    y_all  = analysis_data.(feat);
    y_high = y_all(high_idx);
    y_low  = y_all(low_idx);
    boxplot(y_all, high_idx, 'Labels', {'Low \DeltaB', 'High \DeltaB'}, 'Colors', 'k', 'Symbol', 'r.');
    [~, p_ttest] = ttest2(y_high, y_low);
    d = calculate_cohen_d(y_high, y_low);
    y_lims = ylim; y_range = diff(y_lims);
    text(1.5, y_lims(2) - (y_range * 0.05), p_to_star(p_ttest), ...
        'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'r');
    title(sprintf('%s Power\nd = %.2f', upper(feat), d));
    ylabel('Normalized Power');
    grid on; set(gca, 'Box', 'off', 'TickDir', 'out');
end

%% --- FIGURE 3: ENTROPY ANALYSIS ---
figure('Name', 'Entropy Significance Analysis', 'Color', 'w', 'Position', [200 200 450 450]); hold on;
y = analysis_data.Entropy;
x = analysis_data.DeltaB;
[r, p] = corr(x, y);
d = calculate_cohen_d(y(high_idx), y(low_idx));
scatter(x, y, 40, [0.6 0.1 0.1], 'filled', 'MarkerFaceAlpha', 0.6);
lsline;
title(sprintf('Entropy (r=%.2f, %s)\nd = %.2f', r, p_to_star(p), d));
xlabel('\Delta Belief Magnitude'); ylabel('Sample Entropy');
set(gca, 'Box', 'off', 'TickDir', 'out');

%% --- FIGURE 4: GLOBAL INTERACTION HEATMAP ---
figure('Name', 'Global Feature Interaction', 'Color', 'w', 'Position', [100 100 850 750]);
vars_to_plot = {'DeltaB', 'SocialNum', 'PhotoNum', 'P300', 'N200', ...
                'delta', 'theta', 'alpha', 'beta', 'gamma', 'Entropy'};
plot_labels = {'\Delta Belief', 'Social Bias', 'Visual Bias', 'P300', 'N200', ...
               '\delta', '\theta', '\alpha', '\beta', '\gamma', 'Entropy'};
data_mat = table2array(analysis_data(:, vars_to_plot));
[R, P] = corr(data_mat, 'Type', 'Spearman', 'Rows', 'complete');
imagesc(R); colormap(sky(100)); colorbar; clim([-1 1]);
for i = 1:size(R,1)
    for j = 1:size(R,2)
        txt_col = 'k'; if abs(R(i,j)) > 0.5; txt_col = 'w'; end
        stars = p_to_star(P(i,j));
        text(j, i, sprintf('%.2f\n%s', R(i,j), stars), 'HorizontalAlignment', 'center', ...
            'Color', txt_col, 'FontSize', 9, 'FontWeight', 'bold');
    end
end
xticks(1:length(plot_labels)); xticklabels(plot_labels);
yticks(1:length(plot_labels)); yticklabels(plot_labels);
xtickangle(45); title('Neural-Environmental Interaction Matrix');

%% ============================================================
% 3. LOCAL STATISTICAL HELPERS (MUST BE AT END OF FILE)
% ============================================================
function s = p_to_star(p)
    if p < 0.001; s = '***'; 
    elseif p < 0.01; s = '**'; 
    elseif p < 0.05; s = '*'; 
    else; s = 'ns'; end
end

function d = calculate_cohen_d(x1, x2)
    n1 = length(x1); n2 = length(x2);
    s_pooled = sqrt(((n1-1)*var(x1) + (n2-1)*var(x2)) / (n1+n2-2));
    d = (mean(x1) - mean(x2)) / s_pooled;
    if isnan(d); d = 0; end
end
