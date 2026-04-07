%% =========================================
% PART 1A: EEG → DISCRETE BELIEF UPDATING MODEL
% =========================================
clear; clc; close all;

% --- Publication Aesthetics ---
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultLineLineWidth', 1.5);

%% =========================================
% 1. LOAD ALL SUBJECT DATA
% =========================================
data_root = 'C:\Users\ria20\Desktop\BME499_Final\';
subjects_to_use = 1:80; 
ALL = [];

for s = subjects_to_use
    subj_name = sprintf('sub-%02d', s);
    feature_path = fullfile(data_root, subj_name, 'processed_EEG', 'EEG_features.mat');
    
    if exist(feature_path, 'file')
        load(feature_path); % loads TrialTable
        ALL = [ALL; TrialTable];
    end
end
fprintf('Total trials loaded: %d\n', height(ALL));

%% =========================================
% 2. EXTRACT + NORMALIZE VARIABLES TO [0, 1]
% =========================================
% --- RESPONSE (Belief) ---
% Normalize 1-7 scale to 0.0 - 1.0. 
% (Note: If your raw responses only go from 1 to 5, change the 6 to a 4).
R = (double(ALL.Response(:)) - 1) / 6; 

% --- EVIDENCE COMPONENTS ---
photo  = double(ALL.PhotoNum(:));        % 0 = oppose, 1 = support
social = double(ALL.SocialNum(:));       % 1=low, 2=med, 3=high
social = (social - 1) / 2;               % Maps to 0.0, 0.5, 1.0

% --- EEG FEATURES (Z-Scored) ---
EEG_data = struct();
EEG_data.P300    = zscore(double(ALL.P300(:)));
EEG_data.N200    = zscore(double(ALL.N200(:)));
EEG_data.entropy = zscore(double(ALL.Entropy(:)));
EEG_data.delta   = zscore(double(ALL.delta(:)));
EEG_data.theta   = zscore(double(ALL.theta(:)));
EEG_data.alpha   = zscore(double(ALL.alpha(:)));
EEG_data.beta    = zscore(double(ALL.beta(:)));
EEG_data.gamma   = zscore(double(ALL.gamma(:)));

%% =========================================
% 3. ALIGN TRIALS FOR UPDATE TARGET
% =========================================
dR = diff(R);             % Actual belief change (Target)
R_prev = R(1:end-1);      % B_t
photo_use = photo(1:end-1);
social_use = social(1:end-1);

% Align EEG arrays
feature_names = fieldnames(EEG_data);
for i = 1:length(feature_names)
    fn = feature_names{i};
    EEG_use.(fn) = EEG_data.(fn)(1:end-1);
end

%% =========================================
% 4. OPTIMIZATION SETUP
% =========================================
% x(1) = w1 (Visual weight). Social weight w2 is automatically (1 - w1).
% x(2) = c0 (Baseline alpha)
% x(3:10) = EEG coefficients 
%      w1   c0   P300 N200 Ent  Del  The  Alp  Bet  Gam
x0 = [0.5, 0.1,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
lb = [0.0, 0.0, -2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0];
ub = [1.0, 1.0,  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

options = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 500);

%% =========================================
% 5. EXECUTE OPTIMIZATION
% =========================================
best_x = fmincon(@(x) loss_update(x, R_prev, photo_use, social_use, dR, EEG_use),...
                 x0, [], [], [], [], lb, ub, [], options);

fprintf('\n--- OPTIMIZATION RESULTS ---\n');
fprintf('Visual Weight (w1) = %.3f\n', best_x(1));
fprintf('Social Weight (w2) = %.3f\n', 1 - best_x(1));
fprintf('Baseline Alpha     = %.3f\n', best_x(2));

%% =========================================
% 6. PREDICT BELIEF UPDATES
% =========================================
[dR_pred, alpha_t, E_t] = forward_model(best_x, R_prev, photo_use, social_use, EEG_use);

%% =========================================
% 7. VISUALIZATION PACKAGE
% =========================================

%% ===== FIGURE 1: MODEL FIT (R²) =====
figure('Position', [100, 100, 600, 500]);
scatter(dR, dR_pred, 30, [0.2 0.4 0.6], 'filled', 'MarkerFaceAlpha', 0.6);
hold on; grid on;

SS_res = sum((dR - dR_pred).^2, 'omitnan');
SS_tot = sum((dR - mean(dR, 'omitnan')).^2, 'omitnan');
R2 = 1 - SS_res / SS_tot;

plot([-1 1], [-1 1], 'k--', 'LineWidth', 1); 
lsline; 

xlabel('Observed Human Belief Change (\DeltaB_{actual})', 'Interpreter', 'tex', 'FontWeight', 'bold'); 
ylabel('Predicted Model Belief Change (\DeltaB_{model})', 'Interpreter', 'tex', 'FontWeight', 'bold');
title('Short-Term Model Calibration Fit', 'FontSize', 14);

text(min(xlim)+0.1, max(ylim)-0.1, sprintf('Model R^2 = %.3f', R2),...
    'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'EdgeColor', 'k', 'Margin', 5);

%% ===== FIGURE 2: CONDITION vs EEG FEATURE MEANS (5x8 MATRIX) =====
figure('Position', [150, 150, 900, 450]);

eeg_data = [
    EEG_use.P300, EEG_use.N200, EEG_use.entropy,...
    EEG_use.alpha, EEG_use.beta, EEG_use.gamma,...
    EEG_use.theta, EEG_use.delta
];
x_labels = {'P300', 'N200', 'Entropy', '\alpha', '\beta', '\gamma', '\theta', '\delta'};

heatmap_data = zeros(5, 8);

% FIXED: Now correctly extracting 0 (Oppose) and 1 (Support)
heatmap_data(1,:) = mean(eeg_data(photo_use == 0, :), 1, 'omitnan'); 
heatmap_data(2,:) = mean(eeg_data(photo_use == 1, :), 1, 'omitnan'); 

% Added rounding protection for floating-point comparisons
heatmap_data(3,:) = mean(eeg_data(round(social_use,1) == 0.0, :), 1, 'omitnan'); 
heatmap_data(4,:) = mean(eeg_data(round(social_use,1) == 0.5, :), 1, 'omitnan'); 
heatmap_data(5,:) = mean(eeg_data(round(social_use,1) == 1.0, :), 1, 'omitnan'); 

y_labels = {'Photo: Oppose', 'Photo: Support', 'Social: Low', 'Social: Medium', 'Social: High'};

imagesc(heatmap_data);

n_colors = 256;
cmap = zeros(n_colors, 3);
half = round(n_colors/2);
cmap(1:half, 1) = linspace(0, 1, half);
cmap(1:half, 2) = linspace(0, 1, half);
cmap(1:half, 3) = 1;
cmap(half:end, 1) = 1;
cmap(half:end, 2) = linspace(1, 0, n_colors-half+1);
cmap(half:end, 3) = linspace(1, 0, n_colors-half+1);
colormap(cmap);

max_val = max(abs(heatmap_data(:))); 
if max_val == 0; max_val = 1; end 
set(gca, 'CLim', [-max_val, max_val]); 

cb = colorbar; 
cb.Label.String = 'Mean Z-Score (\mu)';

title('Mean Neural Responses by Environmental Condition', 'FontSize', 14);
xticks(1:length(x_labels)); 
xticklabels(x_labels); 
set(gca, 'TickLabelInterpreter', 'tex'); 
yticks(1:length(y_labels)); 
yticklabels(y_labels);

hold on;
for i = 0.5:1:8.5
    plot([i i], [0.5 5.5], 'k-', 'LineWidth', 0.5);
end
for i = 0.5:1:5.5
    plot([0.5 8.5], [i i], 'k-', 'LineWidth', 0.5);
end

for i = 1:size(heatmap_data,1)
    for j = 1:size(heatmap_data,2)
        val = heatmap_data(i,j);
        if isnan(val)
            text(j, i, 'NaN', 'HorizontalAlignment','center', 'Color', 'k', 'FontSize', 10);
        else
            txt_color = 'k';
            if abs(val) > max_val * 0.5 
                txt_color = 'w'; 
            end
            fw = 'normal';
            if abs(val) > max_val * 0.3
                fw = 'bold';
            end
            text(j, i, sprintf('%+.2f', val), 'HorizontalAlignment','center', ...
                 'Color', txt_color, 'FontSize', 10, 'FontWeight', fw);
        end
    end
end

%% ===== FIGURE 3: ALPHA SUSCEPTIBILITY DISTRIBUTION =====
figure('Position', [200, 200, 600, 400]);
histogram(alpha_t, 30, 'FaceColor', [0.85 0.33 0.10], 'EdgeColor', 'k');
grid on;

xlabel('Calculated Learning Rate (\alpha)', 'Interpreter', 'tex', 'FontWeight', 'bold');
ylabel('Frequency (Trials)', 'FontWeight', 'bold');
title('Distribution of EEG-Derived Susceptibility', 'FontSize', 14);

xl = xline(mean(alpha_t, 'omitnan'), 'k--', 'LineWidth', 2);
text(mean(alpha_t, 'omitnan'), max(ylim)*0.95, sprintf(' Mean \\alpha = %.3f', mean(alpha_t, 'omitnan')),...
    'Interpreter', 'tex', 'FontWeight', 'bold', 'HorizontalAlignment', 'left');

%% =========================================
% 8. FUNCTIONS
% =========================================
function L = loss_update(x, R_prev, photo, social, dR, EEG)
    [dR_pred, ~, ~] = forward_model(x, R_prev, photo, social, EEG);
    L = mean((dR_pred - dR).^2, 'omitnan'); 
end

function [dR_pred, alpha_t, E_t] = forward_model(x, R_prev, photo, social, EEG)
    w1 = x(1);
    w2 = 1 - w1; 
    c0 = x(2);
    
    E_t = w1 * photo + w2 * social;
    
    alpha_t = c0 ...
        + x(3)*EEG.P300 + x(4)*EEG.N200 + x(5)*EEG.entropy ...
        + x(6)*EEG.delta + x(7)*EEG.theta + x(8)*EEG.alpha ...
        + x(9)*EEG.beta + x(10)*EEG.gamma;
        
    alpha_t = max(0, min(1, alpha_t));
    
    dR_pred = alpha_t .* (E_t - R_prev);
end
