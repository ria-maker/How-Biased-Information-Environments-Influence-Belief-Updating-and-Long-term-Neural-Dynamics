%% ============================================================
% PART 1B: 3x2 FACTORIAL FORECASTING (ADULT BASELINE)
% ============================================================
clear; clc; close all;

% --- Publication Aesthetics ---
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontSize', 11);
set(groot, 'defaultLineLineWidth', 1.5);

%% ============================================================
% 1. SIMULATION PARAMETERS (3x2 FACTORIAL DESIGN)
% ============================================================
w1 = 0.139;          % Visual / Digital Media Weight
w2 = 0.861;          % Social Consensus Weight
alpha_lab = 0.157;   % Raw Adult Baseline from EEG optimization

% --- NEW: Timescale Translation ---
epsilon = 0.01;                 % Long-term scaling factor (1/100)
alpha = alpha_lab * epsilon; % Realistic real-world susceptibility

N_agents = 100;      
T_steps  = 300;      
B_init   = 0.5;      

% The 3x2 Conditions
soc_vals = [0.2, 0.5, 0.8];
soc_labels = {'Low', 'Med', 'High'};

dig_vals = [0.0, 1.0]; % Strictly binary: 0% Support (Oppose) vs 100% Support
dig_labels = {'Oppose', 'Support'};

% Data Storage: 3x2 Struct Array
results = repmat(struct('B_traj_mean', [], 'B_traj_std', [], ...
                        'Vol_mean', [], 'Final_Beliefs', []), 3, 2);

rng(42); 

%% ============================================================
% 2. RUN 3x2 FACTORIAL SIMULATION
% ============================================================
for i_soc = 1:3
    for j_dig = 1:2
        p_soc = soc_vals(i_soc);
        p_dig = dig_vals(j_dig);
        
        B_agents = B_init * ones(N_agents, 1);
        B_history = zeros(N_agents, T_steps);
        B_history(:, 1) = B_agents;
        
        Vol_history = zeros(N_agents, T_steps); % Tracks |dB|
        
        for t = 1:(T_steps - 1)
            % 1. Generate Environment E_t for all agents
            photo_t  = double(rand(N_agents, 1) < p_dig);
            social_t = p_soc + randn(N_agents, 1) * 0.1; % Peer noise
            social_t = max(0, min(1, social_t));         % Bound to [0,1]
            
            E_t = (w1 * photo_t) + (w2 * social_t);
            
            % 2. Update Beliefs
            dB = alpha .* (E_t - B_agents);
            B_agents = B_agents + dB;
            
            % 3. Store Data
            B_history(:, t+1) = B_agents;
            Vol_history(:, t+1) = abs(dB); % Volatility is the update magnitude
        end
        
        % Save aggregated statistics for this condition
        results(i_soc, j_dig).B_traj_mean = mean(B_history, 1);
        results(i_soc, j_dig).B_traj_std  = std(B_history, 0, 1);
        results(i_soc, j_dig).Vol_mean    = mean(mean(Vol_history, 1)); % Temporal mean of pop mean
        results(i_soc, j_dig).Final_Beliefs = B_history(:, end);
    end
end

%% ============================================================
% 3. VISUALIZATION PACKAGE
% ============================================================

%% --- FIGURE 1: 3x2 BELIEF TRAJECTORY GRID ---
figure('Position', [50, 50, 800, 800]);
tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
x_time = 1:T_steps;
C_line = [0.12 0.35 0.60]; % Deep blue

for i_soc = 1:3
    for j_dig = 1:2
        nexttile; hold on; grid on;
        
        mu  = results(i_soc, j_dig).B_traj_mean;
        sig = results(i_soc, j_dig).B_traj_std;
        
        % Shaded Error Bounds (+/- 1 Std Dev)
        fill([x_time, fliplr(x_time)], [mu + sig, fliplr(mu - sig)], ...
             C_line, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        
        % Mean Line
        plot(x_time, mu, 'Color', C_line, 'LineWidth', 2);
        
        % Neutral Reference Line
        yline(0.5, 'k--', 'LineWidth', 1);
        
        ylim([0.1 0.9]);
        title(sprintf('Social: %s | Digital: %s', soc_labels{i_soc}, dig_labels{j_dig}), ...
            'FontSize', 10);
        
        if j_dig == 1; ylabel('Belief State (B_t)'); end
        if i_soc == 3; xlabel('Exposures (t)'); end
        
        set(gca, 'Box', 'off', 'TickDir', 'out');
    end
end
sgtitle('Temporal Belief Dynamics Across 6 Interacting Environments', 'FontSize', 16, 'FontWeight', 'bold');

%% --- FIGURE 2: EQUILIBRIUM HEATMAP (FINAL STATES) ---
figure('Position', [150, 150, 500, 500]);
final_means_matrix = zeros(3,2);

for i = 1:3
    for j = 1:2
        final_means_matrix(i,j) = mean(results(i, j).Final_Beliefs);
    end
end

imagesc(final_means_matrix);
colormap(parula);
set(gca, 'CLim', [0.2 0.8]); % Anchor color limits to biological bounds
cb = colorbar; 
cb.Label.String = 'Mean Final Equilibrium Belief';

% Text overlay
for i = 1:3
    for j = 1:2
        val = final_means_matrix(i,j);
        txt_color = 'k';
        if val < 0.3 || val > 0.7; txt_color = 'w'; end
        text(j, i, sprintf('%.3f', val), 'HorizontalAlignment', 'center', ...
             'Color', txt_color, 'FontSize', 14, 'FontWeight', 'bold');
    end
end

xticks(1:2); xticklabels(dig_labels); xlabel('Digital/Visual Feed (w_1 = 0.116)', 'FontWeight', 'bold');
yticks(1:3); yticklabels(soc_labels); ylabel('Social Consensus Bias (w_2 = 0.884)', 'FontWeight', 'bold');
title('Final System Entrenchment', 'FontSize', 14);
set(gca, 'TickDir', 'out');

%% --- FIGURE 3: MAIN EFFECTS & SIGNIFICANCE (SOCIAL BOXPLOT) ---
figure('Position', [250, 250, 700, 500]); hold on;

% Pool the 2 digital conditions per social condition
grp_low  = [results(1,1).Final_Beliefs; results(1,2).Final_Beliefs];
grp_med  = [results(2,1).Final_Beliefs; results(2,2).Final_Beliefs];
grp_high = [results(3,1).Final_Beliefs; results(3,2).Final_Beliefs];

all_data = [grp_low, grp_med, grp_high];
boxplot(all_data, 'Labels', soc_labels, 'Colors', 'k', 'Symbol', 'o');

% Overlay jittered scatter (Now using N_agents * 2)
x_jit = randn(N_agents*2, 1) * 0.05;
scatter(ones(N_agents*2,1) + x_jit, grp_low, 10, [0.4 0.6 0.8], 'filled', 'MarkerFaceAlpha', 0.5);
scatter(2*ones(N_agents*2,1) + x_jit, grp_med, 10, [0.4 0.6 0.8], 'filled', 'MarkerFaceAlpha', 0.5);
scatter(3*ones(N_agents*2,1) + x_jit, grp_high, 10, [0.4 0.6 0.8], 'filled', 'MarkerFaceAlpha', 0.5);

% Stats
[~, p12] = ttest2(grp_low, grp_med);
[~, p23] = ttest2(grp_med, grp_high);
[~, p13] = ttest2(grp_low, grp_high);

% Significance Brackets
y_max = max(all_data(:));
draw_sig_bracket(1, 2, y_max + 0.02, p12);
draw_sig_bracket(2, 3, y_max + 0.02, p23);
draw_sig_bracket(1, 3, y_max + 0.07, p13);

ylim([0.1 y_max + 0.15]);
xlabel('Social Environment Bias Main Effect');
ylabel('Final Entrenched Belief Distribution');
title('Statistical Validation of Social Polarization', 'FontSize', 14);
set(gca, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1);

%% --- FIGURE 4: COGNITIVE VOLATILITY BAR CHART ---
figure('Position', [350, 350, 500, 400]);

% Correctly reshape the 3x2 Vol_mean fields into a 3x2 matrix
vol_data = reshape([results.Vol_mean], 3, 2); 
b = bar(vol_data, 'grouped', 'EdgeColor', 'k', 'LineWidth', 1);

% Custom Colors for the 2 Digital conditions
b(1).FaceColor = [0.8 0.8 0.8]; % Light Gray (Oppose)
b(2).FaceColor = [0.2 0.2 0.2]; % Dark Gray (Support)

xticks(1:3); xticklabels(soc_labels);
xlabel('Social Environment Bias', 'FontWeight', 'bold');
ylabel('Mean Cognitive Volatility ($|dB|$)', 'Interpreter', 'latex', 'FontWeight', 'bold');
title('Information Volatility Across Interacting Environments', 'FontSize', 14);
legend(b, dig_labels, 'Location', 'northeast', 'Box', 'off');
set(gca, 'Box', 'off', 'TickDir', 'out');

%% --- FIGURE 3B: MAIN EFFECTS & SIGNIFICANCE (BINARY DIGITAL BIAS) ---
figure('Position', [300, 300, 500, 500]); hold on;

% Pool the 3 social conditions per digital condition
grp_vis_oppose  = [results(1,1).Final_Beliefs; results(2,1).Final_Beliefs; results(3,1).Final_Beliefs];
grp_vis_support = [results(1,2).Final_Beliefs; results(2,2).Final_Beliefs; results(3,2).Final_Beliefs];

all_vis_data = [grp_vis_oppose, grp_vis_support];
boxplot(all_vis_data, 'Labels', dig_labels, 'Colors', 'k', 'Symbol', 'o');

% Overlay jittered scatter (Using N_agents * 3)
x_jit = randn(N_agents*3, 1) * 0.05;
C_scatter = [0.5 0.5 0.5]; 
scatter(ones(N_agents*3,1) + x_jit, grp_vis_oppose, 10, C_scatter, 'filled', 'MarkerFaceAlpha', 0.4);
scatter(2*ones(N_agents*3,1) + x_jit, grp_vis_support, 10, C_scatter, 'filled', 'MarkerFaceAlpha', 0.4);

% Stats
[~, p_val_v] = ttest2(grp_vis_oppose, grp_vis_support);

% Significance Bracket
y_max_v = max(all_vis_data(:));
plot([1, 1, 2, 2], [y_max_v+0.02, y_max_v+0.03, y_max_v+0.03, y_max_v+0.02], '-k', 'LineWidth', 1.2);

if p_val_v < 0.001; stars = '***';
elseif p_val_v < 0.01; stars = '**';
elseif p_val_v < 0.05; stars = '*';
else; stars = 'ns'; end
text(1.5, y_max_v + 0.04, stars, 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');

ylim([0.1 1.05]); 
xlabel('Digital Media Feed Content');
ylabel('Final Entrenched Belief Distribution');
title('Isolated Digital Media Effect (w_1 = 0.116)', 'FontSize', 14);
set(gca, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1);

%% ============================================================
% 4. HELPER FUNCTIONS
% ============================================================
function draw_sig_bracket(x1, x2, y, p_val)
    plot([x1, x1, x2, x2], [y-0.01, y, y, y-0.01], '-k', 'LineWidth', 1.2);
    if p_val < 0.001; stars = '***';
    elseif p_val < 0.01; stars = '**';
    elseif p_val < 0.05; stars = '*';
    else; stars = 'ns'; end
    text((x1+x2)/2, y + 0.01, stars, 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
end
