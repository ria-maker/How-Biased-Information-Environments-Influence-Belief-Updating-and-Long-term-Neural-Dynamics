%% =========================================
% PART 1A: EEG → BELIEF UPDATING MODEL
% =========================================

clear; clc; close all;

%% =========================================
% 1. LOAD ALL SUBJECT DATA (1–80)
% =========================================

data_root = ''; % add data root
subjects_to_use = 1:70;

ALL = [];

for s = subjects_to_use
%for s = 1
    subj_name = sprintf('sub-%02d', s);
    feature_path = fullfile(data_root, subj_name, ...
        'processed_EEG', 'EEG_features.mat');
    
    if exist(feature_path, 'file')
        load(feature_path); % loads TrialTable
        ALL = [ALL; TrialTable];
        fprintf('Loaded %s\n', subj_name);
    else
        fprintf('Missing %s\n', subj_name);
    end
end

fprintf('Total trials: %d\n', height(ALL));

%% =========================================
% 2. EXTRACT + NORMALIZE VARIABLES
% =========================================

% --- RESPONSE (belief) ---
R = (double(ALL.Response(:)) - 1) / 6; % normalize 1–7 → 0–1

% --- EVIDENCE COMPONENTS ---
photo  = double(ALL.PhotoNum(:));   % 0 = oppose, 1 = support
photo = 2*photo - 1;   % now: -1 (oppose), +1 (support)

social = double(ALL.SocialNum(:));  % 1=low, 2=med, 3=high

% Normalize social to [0,1]
social = (social - 1) / 2;  % 0, 0.5, 1

% Combined evidence
E = 0.6*photo + 0.4*(2*social - 1);

% --- EEG FEATURES ---
EEG.P300    = zscore(double(ALL.P300(:)));
EEG.N200    = zscore(double(ALL.N200(:)));
EEG.entropy = zscore(double(ALL.Entropy(:)));
EEG.theta   = zscore(double(ALL.theta(:)));
EEG.alpha   = zscore(double(ALL.alpha(:)));

T = length(R);

%% =========================================
% 3. BELIEF UPDATE TARGET
% =========================================

dR = diff(R);              % belief change
R_prev = R(1:end-1);
E_use  = E(1:end-1);

% Align EEG
EEG_use.P300    = EEG.P300(1:end-1);
EEG_use.N200    = EEG.N200(1:end-1);
EEG_use.entropy = EEG.entropy(1:end-1);
EEG_use.theta   = EEG.theta(1:end-1);
EEG_use.alpha   = EEG.alpha(1:end-1);

T_use = length(dR);

%% =========================================
% 4. OPTIMIZATION SETUP
% =========================================

% Parameters:
% α(t) = a0 + a1*P300 + a2*Entropy
% η(t) = c0 + c1*theta
% τ(t) = b0 + b1*N200

x0 = [0.2 0.1 0.1 ...  % a0 a1 a2
      0.5 0.2 ...      % c0 c1
      1.0 0.5];        % b0 b1

lb = [-2 -2 -2   0 0   0 0];
ub = [ 2  2  2   5 5   5 5];

options = optimoptions('fmincon',...
    'Display','iter',...
    'MaxIterations',150,...
    'UseParallel',false);

%% =========================================
% 5. OPTIMIZE
% =========================================

best_x = fmincon(@(x) loss_update(x, R_prev, E_use, dR, EEG_use),...
                 x0,[],[],[],[],lb,ub,[],options);

disp('Fitted parameters:');
disp(best_x);

%% =========================================
% 6. PREDICT BELIEF UPDATES
% =========================================

[dR_pred, alpha_t] = forward_model(best_x, R_prev, E_use, EEG_use);

%% =========================================
% 7. VISUALIZATION
% =========================================

%% ===== (A) TRUE vs PREDICTED WITH R² =====
figure;

scatter(dR, dR_pred, 20, 'filled');
xlabel('Actual ΔR'); 
ylabel('Predicted ΔR');
title('Belief Updating Fit');
grid on; hold on;

% ===== COMPUTE R² =====
SS_res = sum((dR - dR_pred).^2);
SS_tot = sum((dR - mean(dR)).^2);
R2 = 1 - SS_res / SS_tot;

% ===== LINE OF BEST FIT =====
lsline;

% ===== DISPLAY R² IN CENTER =====
x_mid = mean(xlim);
y_mid = mean(ylim);

text(x_mid, y_mid, sprintf('R^2 = %.3f', R2),...
    'HorizontalAlignment','center',...
    'FontSize',12,...
    'FontWeight','bold',...
    'BackgroundColor','w');


%% ===== (B1) PHOTO vs EEG: MEAN + CORR + ANOVA =====
figure;

photo_use = photo(1:end-1);

EEG_matrix = [
    EEG_use.P300,...
    EEG_use.N200,...
    EEG_use.entropy,...
    EEG_use.theta,...
    EEG_use.alpha
];

feature_names = {'P300','N200','Entropy','Theta','Alpha'};
photo_vals = [-1, 1];

heatmap_data = zeros(2, size(EEG_matrix,2));
r_vals = zeros(1, size(EEG_matrix,2));
eta2_vals = zeros(1, size(EEG_matrix,2));
p_vals = zeros(1, size(EEG_matrix,2));

% ===== COMPUTE MEANS =====
for i = 1:2
    idx = (photo_use == photo_vals(i));
    heatmap_data(i,:) = mean(EEG_matrix(idx,:), 1, 'omitnan');
end

% ===== STATS PER FEATURE =====
for j = 1:size(EEG_matrix,2)
    
    eeg = EEG_matrix(:,j);

    % --- CORRELATION ---
    if std(eeg) > 0
        r_vals(j) = corr(eeg, photo_use, 'Rows','complete');
    else
        r_vals(j) = NaN;
    end
    
    % --- ANOVA ---
    group = photo_use;
    [p, tbl, ~] = anova1(eeg, group, 'off');
    
    SS_between = tbl{2,2};
    SS_total   = SS_between + tbl{3,2};
    
    eta2_vals(j) = SS_between / SS_total;
    p_vals(j) = p;
end

% ===== PLOT =====
imagesc(heatmap_data);
colorbar;
title('Photo vs EEG Features');

xticks(1:length(feature_names));
xticklabels(feature_names);
yticks([1 2]);
yticklabels({'Oppose','Support'});

hold on;

% ===== ADD TEXT =====
for i = 1:2
    for j = 1:length(feature_names)
        
        r = r_vals(j);
        eta = eta2_vals(j);
        p = p_vals(j);
        
        % significance stars
        if p < 0.001
            sig = '***';
        elseif p < 0.01
            sig = '**';
        elseif p < 0.05
            sig = '*';
        else
            sig = '';
        end
        
        text(j, i, sprintf('r=%.2f\nη^2=%.2f%s', r, eta, sig),...
            'HorizontalAlignment','center',...
            'Color','k',...
            'FontSize',9,...
            'FontWeight','bold');
    end
end

%% ===== (B2) SOCIAL vs EEG: MEAN + CORR + ANOVA =====
figure;

social_use = social(1:end-1);

EEG_matrix = [
    EEG_use.P300,...
    EEG_use.N200,...
    EEG_use.entropy,...
    EEG_use.theta,...
    EEG_use.alpha
];

feature_names = {'P300','N200','Entropy','Theta','Alpha'};
social_vals = [0, 0.5, 1];

heatmap_data = zeros(3, size(EEG_matrix,2));
r_vals = zeros(1, size(EEG_matrix,2));
eta2_vals = zeros(1, size(EEG_matrix,2));
p_vals = zeros(1, size(EEG_matrix,2));

% ===== COMPUTE MEANS =====
for i = 1:3
    idx = (social_use == social_vals(i));
    heatmap_data(i,:) = mean(EEG_matrix(idx,:), 1, 'omitnan');
end

% ===== STATS PER FEATURE =====
for j = 1:size(EEG_matrix,2)
    
    eeg = EEG_matrix(:,j);

    % --- CORRELATION ---
    if std(eeg) > 0 && std(social_use) > 0
        r_vals(j) = corr(eeg, social_use, 'Rows','complete');
    else
        r_vals(j) = NaN;
    end
    
    % --- ANOVA ---
    group = social_use;
    [p, tbl, ~] = anova1(eeg, group, 'off');
    
    SS_between = tbl{2,2};
    SS_total   = SS_between + tbl{3,2};
    
    eta2_vals(j) = SS_between / SS_total;
    p_vals(j) = p;
end

% ===== PLOT =====
imagesc(heatmap_data);
colorbar;
title('Social Influence vs EEG Features');

xticks(1:length(feature_names));
xticklabels(feature_names);
yticks(1:3);
yticklabels({'Low','Medium','High'});

hold on;

% ===== ADD TEXT =====
for i = 1:3
    for j = 1:length(feature_names)
        
        r = r_vals(j);
        eta = eta2_vals(j);
        p = p_vals(j);
        
        % significance stars
        if p < 0.001
            sig = '***';
        elseif p < 0.01
            sig = '**';
        elseif p < 0.05
            sig = '*';
        else
            sig = '';
        end
        
        text(j, i, sprintf('r=%.2f\nη^2=%.2f%s', r, eta, sig),...
            'HorizontalAlignment','center',...
            'Color','k',...
            'FontSize',9,...
            'FontWeight','bold');
    end
end

%% ===== (C) CORRELATION MATRIX (UPDATED) =====
figure;

corr_data = [
    photo(1:end-1),...
    social(1:end-1),...
    dR,...
    dR_pred,...
    EEG_use.P300,...
    EEG_use.N200,...
    EEG_use.entropy,...
    EEG_use.theta,...
    EEG_use.alpha
];

labels = {'Photo','Social','Actual dR','Pred dR',...
          'P300','N200','Entropy','Theta','Alpha'};

corr_mat = corr(corr_data, 'Rows','complete');

imagesc(corr_mat);
colorbar;
title('Neural + Behavioral Correlations');

set(gca,'XTick',1:length(labels),'XTickLabel',labels);
set(gca,'YTick',1:length(labels),'YTickLabel',labels);

% ===== ADD CORRELATION VALUES =====
for i = 1:size(corr_mat,1)
    for j = 1:size(corr_mat,2)
        text(j, i, sprintf('%.2f', corr_mat(i,j)),...
            'HorizontalAlignment','center',...
            'Color','w');
    end
end

%% =========================================
% 8. FUNCTIONS
% =========================================

function L = loss_update(x, R_prev, E, dR, EEG)

    [dR_pred, ~] = forward_model(x, R_prev, E, EEG);
    
    % MSE loss
    L = mean((dR_pred - dR).^2);
end

function [dR_pred, alpha_t] = forward_model(x, R_prev, E, EEG)

    % unpack
    a0 = x(1); a1 = x(2); a2 = x(3);
    c0 = x(4); c1 = x(5);
    b0 = x(6); b1 = x(7);

    % ===== EEG → PARAMETERS =====
    alpha_t = a0 + a1*EEG.P300 + a2*EEG.entropy;
    eta_t   = c0 + c1*EEG.theta;
    tau_t   = b0 + b1*EEG.N200;

    % constraints
    alpha_t = max(0, min(1, alpha_t));
    eta_t   = max(0, eta_t);
    tau_t   = max(0.1, tau_t);

    % ===== BELIEF UPDATE =====
    dR_pred = alpha_t .* (E - R_prev) + 0.1 * eta_t;
    dR_pred = max(min(dR_pred,1), -1);

end
