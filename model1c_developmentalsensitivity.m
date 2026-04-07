%% ============================================================
% PART 1C: DEVELOPMENTAL SENSITIVITY ANALYSIS 
% ============================================================
clear; clc; close all;

% --- Publication Aesthetics ---
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultLineLineWidth', 2.0);

%% ============================================================
% 1. PARAMETERS & INITIALIZATION
% ============================================================
w1 = 0.139;          % Visual / Digital Media Weight
w2 = 0.861;          % Social Consensus Weight
alpha_lab = 0.157;   % Raw Adult Baseline from EEG optimization

% --- NEW: Timescale Translation ---
epsilon = 0.01;                 % Long-term scaling factor (1/100)
alpha_adult = alpha_lab * epsilon; % Realistic real-world susceptibility

N_agents = 100;      % Population size for variance
T_steps  = 400;      % Simulation length
B_init   = 0.5;      % Neutral starting belief

% The Independent Variables
gamma_sweep = linspace(1.0, 6.0, 50); % Sweep from Adult(1x) to highly plastic Child(6x)
pA_levels   = [0.65, 0.80, 0.95];     % Mild, Moderate, and Severe Echo Chambers
pA_labels   = {'Mild (p_A = 0.65)', 'Moderate (p_A = 0.80)', 'Severe (p_A = 0.95)'};

% Data Storage
eq_means = zeros(length(pA_levels), length(gamma_sweep));
eq_stds  = zeros(length(pA_levels), length(gamma_sweep));
vol_means = zeros(length(pA_levels), length(gamma_sweep));

rng(42); 

%% ============================================================
% 2. RUN SENSITIVITY SWEEP
% ============================================================
for i = 1:length(pA_levels)
    pA = pA_levels(i);
    
    for j = 1:length(gamma_sweep)
        gamma = gamma_sweep(j);
        alpha_current = min(1.0, alpha_adult * gamma);
        
        B_agents = B_init * ones(N_agents, 1);
        vol_track = zeros(N_agents, T_steps);
        
        for t = 1:T_steps
            % Environment Generation
            photo_t = double(rand(N_agents, 1) < pA);
            social_t = max(0, min(1, pA + randn(N_agents, 1) * 0.1));
            E_t = (w1 * photo_t) + (w2 * social_t);
            
            % Update Rule
            dB = alpha_current .* (E_t - B_agents);
            B_agents = B_agents + dB;
            
            vol_track(:, t) = abs(dB);
        end
        
        % Store statistics from the final 50 steps (Equilibrium)
        eq_means(i, j) = mean(B_agents);
        eq_stds(i, j)  = std(B_agents);
        vol_means(i, j) = mean(vol_track(:, end-50:end), 'all');
    end
end

%% ============================================================
% 3. VISUALIZATION PACKAGE
% ============================================================
% High-contrast color palette for the 3 bias levels
C_lines = [0.4660, 0.6740, 0.1880;  % Mild (Green)
           0.0000, 0.4470, 0.7410;  % Mod (Blue)
           0.8500, 0.3250, 0.0980]; % Severe (Red)

%% --- FIGURE 1: SENSITIVITY CURVE (ENTRENCHMENT vs GAMMA) ---
figure('Position', [100, 100, 700, 500]); hold on; grid on;

for i = 1:length(pA_levels)
    mu = eq_means(i, :);
    sig = eq_stds(i, :);
    
    % Shaded Variance 
    fill([gamma_sweep, fliplr(gamma_sweep)], ...
         [mu + sig, fliplr(mu - sig)], ...
         C_lines(i,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    % Mean Line
    plot(gamma_sweep, mu, 'Color', C_lines(i,:), 'DisplayName', pA_labels{i});
    
    % Asymptote target line (Theoretical maximum bias)
    yline(pA_levels(i), '--', 'Color', C_lines(i,:), 'LineWidth', 1, 'HandleVisibility', 'off');
end

% Formatting
xlabel('Developmental Plasticity Scale ($\gamma$)', 'Interpreter', 'latex', 'FontWeight', 'bold');
ylabel('Final Entrenched Belief ($B_{eq}$)', 'Interpreter', 'latex', 'FontWeight', 'bold');
title('Sensitivity of Polarization to Neurodevelopmental Stage', 'FontSize', 15);
legend('Location', 'northwest', 'Box', 'off');

% Annotations to highlight the "Troublesome" leap
xline(1.0, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');
text(1.05, 0.55, 'Adult Baseline ($\gamma=1$)', 'Interpreter', 'latex', 'Rotation', 90);

xlim([1.0 6.0]);
set(gca, 'Box', 'off', 'TickDir', 'out');

%% --- FIGURE 2: COGNITIVE VOLATILITY (NOISE vs GAMMA) ---
figure('Position', [150, 150, 700, 500]); hold on; grid on;

for i = 1:length(pA_levels)
    plot(gamma_sweep, vol_means(i, :), 'Color', C_lines(i,:), 'DisplayName', pA_labels{i});
end

xlabel('Developmental Plasticity Scale ($\gamma$)', 'Interpreter', 'latex', 'FontWeight', 'bold');
ylabel('Mean Cognitive Volatility ($|dB|$)', 'Interpreter', 'latex', 'FontWeight', 'bold');
title('Susceptibility to Algorithmic Noise vs. Plasticity', 'FontSize', 15);
legend('Location', 'northwest', 'Box', 'off');

xline(1.0, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xlim([1.0 6.0]);
set(gca, 'Box', 'off', 'TickDir', 'out');

%% --- FIGURE 3: THE "SNAP" EFFECT (TIME SERIES AT GAMMA=4.0) ---
% This figure isolates one scenario to explicitly show WHY the child is troublesome
gamma_target = 4.0;
pA_target = 0.85; 
alpha_ch_target = min(1.0, alpha_adult * gamma_target);

B_ad = B_init * ones(N_agents, 1); B_ch = B_init * ones(N_agents, 1);
B_ad_hist = zeros(N_agents, 150); B_ch_hist = zeros(N_agents, 150);

for t = 1:150
    photo_t = double(rand(N_agents, 1) < pA_target);
    social_t = max(0, min(1, pA_target + randn(N_agents, 1) * 0.1));
    E_t = (w1 * photo_t) + (w2 * social_t);
    
    B_ad = B_ad + alpha_adult .* (E_t - B_ad);
    B_ch = B_ch + alpha_ch_target .* (E_t - B_ch);
    
    B_ad_hist(:, t) = B_ad; B_ch_hist(:, t) = B_ch;
end

figure('Position', [200, 200, 700, 450]); hold on; grid on;
plot(1:150, mean(B_ad_hist), 'Color', [0.3 0.3 0.3], 'LineWidth', 2, 'DisplayName', 'Adult (\gamma=1)');
plot(1:150, mean(B_ch_hist), 'Color', [0.85 0.33 0.10], 'LineWidth', 2, 'DisplayName', 'Child (\gamma=4)');
yline(pA_target, 'k--', 'DisplayName', 'Algorithmic Goal (p_A=0.85)');

xlabel('Algorithmic Exposures ($t$)', 'Interpreter', 'latex');
ylabel('Mean Belief State ($B_t$)', 'Interpreter', 'latex');
title('The Entrenchment Snap: Adult vs. Highly Plastic Child', 'FontSize', 15);
legend('Location', 'southeast', 'Box', 'off');
ylim([0.4 1.0]); xlim([1 150]);
set(gca, 'Box', 'off', 'TickDir', 'out');

%% ============================================================
% 4. RUN CONVERGENCE SWEEP
% ============================================================
for i = 1:length(gamma_sweep)
    gamma = gamma_sweep(i);
    alpha_current = min(1.0, alpha_adult * gamma);
    
    B_agents = B_init * ones(N_agents, 1);
    crossed_threshold_at = T_max; 
    
    for t = 1:T_max
        photo_t = double(rand(N_agents, 1) < pA_target);
        social_t = max(0, min(1, pA_target + randn(N_agents, 1) * 0.1));
        E_t = (w1 * photo_t) + (w2 * social_t);
        
        B_agents = B_agents + alpha_current .* (E_t - B_agents);
        
        if mean(B_agents) >= B_threshold
            crossed_threshold_at = t;
            break; 
        end
    end
    time_to_entrench(i) = crossed_threshold_at;
end

%% ============================================================
% 5. CALCULATE THE "TIPPING POINT" (Kneedle Algorithm)
% ============================================================
% To find the elbow of an L-shaped decay curve, normalize the axes 
% to [0,1] and find the point furthest from the straight line connecting 
% the first and last points of the curve.

g_norm = (gamma_sweep - min(gamma_sweep)) / (max(gamma_sweep) - min(gamma_sweep));
t_norm = (time_to_entrench - min(time_to_entrench)) / (max(time_to_entrench) - min(time_to_entrench));

p1 = [g_norm(1), t_norm(1)];
p2 = [g_norm(end), t_norm(end)];
dist = zeros(size(g_norm));

for k = 1:length(g_norm)
    p0 = [g_norm(k), t_norm(k)];
    % Perpendicular distance from point p0 to line p1-p2
    dist(k) = abs((p2(1)-p1(1))*(p1(2)-p0(2)) - (p1(1)-p0(1))*(p2(2)-p1(2))) / norm(p2-p1);
end

[~, knee_idx] = max(dist);
gamma_tipping = gamma_sweep(knee_idx);
t_tipping = time_to_entrench(knee_idx);

fprintf('--- ANALYSIS COMPLETE ---\n');
fprintf('Adult Baseline (Gamma=1.0): %d exposures\n', time_to_entrench(1));
fprintf('Calculated Tipping Point (Gamma=%.2f): %d exposures\n', gamma_tipping, t_tipping);

%% ============================================================
% 6. VISUALIZATION 
% ============================================================
C_adult = [0.12 0.35 0.60]; % Deep Blue
C_tip   = [0.85 0.20 0.20]; % Crimson Red for the tipping point

figure('Position', [150, 150, 800, 500]); hold on; grid on;

% 1. Plot the Vulnerability Zone background shading
fill([gamma_tipping, max(gamma_sweep), max(gamma_sweep), gamma_tipping], ...
     [0, 0, max(time_to_entrench), max(time_to_entrench)], ...
     [0.95 0.85 0.85], 'EdgeColor', 'none', 'HandleVisibility', 'off');

% 2. Plot the main decay curve
plot(gamma_sweep, time_to_entrench, 'Color', [0.3 0.3 0.3], 'LineWidth', 3, 'DisplayName', 'Exposure Requirement');

% 3. Highlight the Adult Baseline
scatter(1.0, time_to_entrench(1), 120, C_adult, 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Adult Baseline (\gamma=1)');

% 4. Highlight the Tipping Point
scatter(gamma_tipping, t_tipping, 150, C_tip, 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', sprintf('Tipping Point (\\gamma=%.2f)', gamma_tipping));

% Draw lines for Adult
plot([1.0 1.0], [0 time_to_entrench(1)], ':', 'Color', C_adult, 'HandleVisibility', 'off');
plot([0 1.0], [time_to_entrench(1) time_to_entrench(1)], ':', 'Color', C_adult, 'HandleVisibility', 'off');

% Draw lines for Tipping Point
plot([gamma_tipping gamma_tipping], [0 t_tipping], '--', 'Color', C_tip, 'HandleVisibility', 'off');
plot([0 gamma_tipping], [t_tipping t_tipping], '--', 'Color', C_tip, 'HandleVisibility', 'off');

% Annotations
text(1.2, time_to_entrench(1), sprintf(' t = %d exposures', time_to_entrench(1)), 'Color', C_adult, 'FontWeight', 'bold');
text(gamma_tipping + 0.2, t_tipping + 100, sprintf(' Critical Drop\n t = %d exposures', t_tipping), 'Color', C_tip, 'FontWeight', 'bold');
text((gamma_tipping + max(gamma_sweep))/2, max(time_to_entrench)*0.8, 'VULNERABILITY ZONE', 'Color', [0.6 0.2 0.2], 'FontSize', 16, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Formatting
xlabel('Developmental Plasticity Scale ($\gamma$)', 'Interpreter', 'latex', 'FontWeight', 'bold');
ylabel('Exposures Required to Entrench ($t$)', 'Interpreter', 'latex', 'FontWeight', 'bold');
title(sprintf('Time-to-Entrenchment (Reaching %d%% of Target Bias)', threshold_pct*100), 'FontSize', 15);
legend('Location', 'northeast', 'Box', 'off');

xlim([1.0 max(gamma_sweep)]);
ylim([0 max(time_to_entrench) * 1.1]);
set(gca, 'Box', 'off', 'TickDir', 'out');
