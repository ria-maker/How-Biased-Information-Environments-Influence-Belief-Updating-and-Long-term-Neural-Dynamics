%% =========================================
% EEG PROCESSING PIPELINE
% =========================================
clear; clc; close all;

%% -------- 1. EEGLAB SETUP --------
eeglab_path = ''; % add path
addpath(genpath(eeglab_path));
eeglab; close;

%% -------- 2. DATA PATH --------
data_root = ''; % add root
subjects = dir(fullfile(data_root, 'sub-*'));
nSub = length(subjects);
assert(nSub > 0, 'No subjects found.');
fprintf('Found %d subjects\n', nSub);

%% =========================================
% SUBJECTS
% =========================================
subjects_to_use = 53:80;
for sub = subjects_to_use
    %sub = 7; % 1:nSub
    subj_path = fullfile(subjects(sub).folder, subjects(sub).name);
    vhdr_file = dir(fullfile(subj_path, '**', 'eeg', '*.vhdr'));
    
    if isempty(vhdr_file)
        fprintf('No EEG file found\n');
        continue;
    end
    
    EEG = pop_loadbv(vhdr_file(1).folder, vhdr_file(1).name);
    EEG = pop_chanedit(EEG, 'lookup','standard-10-5-cap385.elp');
    EEG = eeg_checkset(EEG);
    
    % -------- 2. DOWNSAMPLE --------
    EEG = pop_resample(EEG, 250);
    
    % -------- 3. FILTER --------
    EEG = pop_eegfiltnew(EEG, 1, 40);         % bandpass
    EEG = pop_eegfiltnew(EEG, 48, 52, [], 1); % notch
    
    % -------- 4. REMOVE BAD CHANNELS --------
    bad_ch = {'TP10','FT10','T8'}; % based on impedance
    EEG = pop_select(EEG, 'nochannel', bad_ch);
    
    % -------- 5. RE-REFERENCE --------
    EEG = pop_reref(EEG, []);
    
    %% -------- 6. ICA --------
    EEG = pop_runica(EEG, 'extended',1,'stop',1e-7);
    % Label components
    EEG = pop_iclabel(EEG, 'default');
    EEG = pop_icflag(EEG, [
        NaN NaN;    % Brain
        0.9 1;      % Muscle
        0.9 1;      % Eye
        0.9 1;      % Heart
        0.9 1;      % Line noise
        0.9 1;      % Channel noise
        NaN NaN]);
        
    badICs = find(EEG.reject.gcompreject);
    if ~isempty(badICs)
        EEG = pop_subcomp(EEG, badICs, 0);
    else
        warning('No ICs flagged for removal');
    end
    
    % -------- 7. EPOCHING --------
    % epoch around stimulus onset
    EEG = pop_epoch(EEG, {'PhotoSupport','PhotoOppose'}, [-0.2 1]);
    EEG = eeg_checkset(EEG);
    
    % -------- 8. BASELINE --------
    EEG = pop_rmbase(EEG, [-200 0]);
    
    %% =========================================
    % CORRECT MULTI-EVENT TRIAL LABELING 
    % =========================================
    nTrials = EEG.trials;
    moral  = strings(nTrials,1);
    social = strings(nTrials,1);
    photo  = strings(nTrials,1);
    response = strings(nTrials,1);
    
    for t = 1:nTrials
        raw_i = EEG.epoch(t).eventurevent;
        if iscell(raw_i)
            num_i = [raw_i{:}]; % Extract numbers from cell
        else
            num_i = raw_i;     
        end
        
        if isempty(num_i)
            continue;
        end
        
        % Grab the first urevent index in this epoch
        i = num_i(1); 
        
        % --------------------------------------
        % LOOK FORWARD IN TRIAL 
        % --------------------------------------
        window = EEG.urevent(i : min(i+10, length(EEG.urevent)));
        wtypes = string({window.type});
        
        % ---------------- MORAL ----------------
        if any(contains(wtypes, "Moral_Support"))
            moral(t) = "Moral_Support";
        elseif any(contains(wtypes, "Moral_Oppose"))
            moral(t) = "Moral_Oppose";
        elseif any(contains(wtypes, "Nonmoral_Support"))
            moral(t) = "Nonmoral_Support";
        elseif any(contains(wtypes, "Nonmoral_Oppose"))
            moral(t) = "Nonmoral_Oppose";
        elseif any(contains(wtypes, "Nonmoral_Neutral"))
            moral(t) = "Nonmoral_Neutral";
        else
            moral(t) = "unknown";
        end
        
        % ---------------- SOCIAL ----------------
        if any(contains(wtypes, "SocLow"))
            social(t) = "Low";
        elseif any(contains(wtypes, "SocMed"))
            social(t) = "Medium";
        elseif any(contains(wtypes, "SocHigh"))
            social(t) = "High";
        else
            social(t) = "unknown";
        end
        
        % ---------------- PHOTO ----------------
        if any(contains(wtypes, "PhotoSupport"))
            photo(t) = "Support";
        elseif any(contains(wtypes, "PhotoOppose"))
            photo(t) = "Oppose";
        else
            photo(t) = "unknown";
        end
        
        % ---------------- RESPONSES ----------------
        if any(contains(wtypes, "Resp1"))
            response(t) = "Resp1";
        elseif any(contains(wtypes, "Resp2"))
            response(t) = "Resp2";
        elseif any(contains(wtypes, "Resp3"))
            response(t) = "Resp3";
        elseif any(contains(wtypes, "Resp4"))
            response(t) = "Resp4";
        elseif any(contains(wtypes, "Resp5"))
            response(t) = "Resp5";
        else
            response(t) = "unknown";
        end
    end
    
    %% =========================================
    % FEATURE EXTRACTION 
    % =========================================
    % Ensure consistency
    nTrials = EEG.trials;
    nChans  = size(EEG.data,1);
    nTimes  = size(EEG.data,2);
    times = EEG.times;  
    
    %% ---------------- ERP FEATURES ----------------
    P300 = zeros(nTrials,1);
    N200 = zeros(nTrials,1);
    
    % Define electrode groups safely
    chan_labels = {EEG.chanlocs.labels};
    parietal = find(ismember(chan_labels, {'Pz','P3','P4'}));
    frontal  = find(ismember(chan_labels, {'Fz','FCz'}));
    
    % Sanity check
    if isempty(parietal) || isempty(frontal)
        error('Missing required electrodes (Pz/P3/P4 or Fz/FCz). Check chanlocs.');
    end
    
    for t = 1:nTrials
        trial_data = EEG.data(:,:,t);  % chans × time
        
        % --- P300 (300–500 ms, parietal) ---
        idxP300 = (times >= 300 & times <= 500);
        P300(t) = mean(trial_data(parietal, idxP300), 'all');
        
        % --- N200 (200–300 ms, frontal) ---
        idxN200 = (times >= 200 & times <= 300);
        N200(t) = mean(trial_data(frontal, idxN200), 'all');
    end
    
    %% ---------------- BAND POWER FEATURES ----------------
    % ---------------- BAND DEFINITION ----------------
    bands.delta = [1 4];
    bands.theta = [4 8];
    bands.alpha = [8 12];
    bands.beta  = [13 30];
    bands.gamma = [30 45];
    
    % Define valid time window
    win = (times >= 0 & times <= 800);
    
    % Preallocate
    band_names = fieldnames(bands);
    nBands = length(band_names);
    band_power = zeros(nBands, nTrials);
    
    for b = 1:nBands
        fr = bands.(band_names{b});
        
        % Filter once per band
        EEG_filt = pop_eegfiltnew(EEG, fr(1), fr(2));
        data_filt = EEG_filt.data;  % chans × time × trials
        
        for t = 1:nTrials
            x = data_filt(:, win, t);
            % Band power estimate (RMS power proxy)
            band_power(b,t) = mean(x(:).^2);
        end
    end
    
    %% ---------------- ENTROPY ----------------
    samp_entropy = zeros(nTrials,1);
    for t = 1:nTrials
        sig = EEG.data(:, :, t);
        sig = sig(:);
        sig = zscore(sig); % stable normalization
        edges = linspace(-3,3,100);
        p = histcounts(sig, edges, 'Normalization','probability');
        p(p==0) = [];
        samp_entropy(t) = -sum(p .* log2(p));
    end
    
    %% =========================================
    % FINAL TRIAL TABLE
    % =========================================
    band_names = fieldnames(bands);
    nBands = length(band_names);
    TrialTable = table();
    
    % --- BASIC LABELS ---
    TrialTable.Trial = (1:nTrials)';
    TrialTable.Moral  = categorical(moral);
    TrialTable.Social = categorical(social);
    TrialTable.Photo = categorical(photo);
    
    % --- RESPONSE LABELS ---
    response_num = zeros(nTrials,1);
    response_num(response == "Resp1") = 1;
    response_num(response == "Resp2") = 2;
    response_num(response == "Resp3") = 3;
    response_num(response == "Resp4") = 4;
    response_num(response == "Resp5") = 5;
    TrialTable.Response = categorical(response_num);
    
    % --- FEATURES ---
    TrialTable.P300 = P300;
    TrialTable.N200 = N200;
    for b = 1:nBands
        varName = band_names{b};
        TrialTable.(varName) = band_power(b,:)';
    end
    TrialTable.Entropy = samp_entropy;
    
    % --- NUMERIC ENCODINGS ---
    moral_num = zeros(nTrials,1);
    moral_num(moral == "Moral_Support") = 1;
    moral_num(moral == "Moral_Oppose")  = -1;
    moral_num(contains(moral,"Nonmoral")) = 0;
    TrialTable.MoralNum = moral_num;
    
    social_num = zeros(nTrials,1);
    social_num(social == "Low")    = 1;
    social_num(social == "Medium") = 2;
    social_num(social == "High")   = 3;
    TrialTable.SocialNum = social_num;
    
    photo_num = zeros(nTrials,1);
    photo_num(photo == "Support") = 1;
    photo_num(photo == "Oppose")  = 0;
    TrialTable.PhotoNum = photo_num;
    
    disp(TrialTable);
    
    %% =========================================
    % Figure Plots 
    % =========================================
    
    %% ---------------- P300 / N200 PLOTS ----------------
    figure('Color', 'w', 'Position', [50 50 1200 800]); 
    % --- P300 ---
    subplot(2,3,1)
    plot_condition(P300, moral_num, [0 1], {'Non-Moral','Moral'}, 'P300: Moral','Amplitude')
    subplot(2,3,2)
    plot_condition(P300, social_num, [1 2 3], {'Low','Medium','High'}, 'P300: Social','Amplitude')
    subplot(2,3,3)
    plot_condition(P300, photo_num, [0 1], {'Oppose','Support'}, 'P300: Photo','Amplitude')
    % --- N200 ---
    subplot(2,3,4)
    plot_condition(N200, moral_num, [0 1], {'Non-Moral','Moral'}, 'N200: Moral','Amplitude')
    subplot(2,3,5)
    plot_condition(N200, social_num, [1 2 3], {'Low','Medium','High'}, 'N200: Social','Amplitude')
    subplot(2,3,6)
    plot_condition(N200, photo_num, [0 1], {'Oppose','Support'}, 'N200: Photo','Amplitude')
    sgtitle('EEG Feature Exploration Across Conditions', 'FontSize', 16, 'FontWeight', 'bold');
    
    %% ---------------- BAND POWER PLOTS ----------------
    figure('Color', 'w', 'Position', [100 100 1200 1000]); 
    for b = 1:5
        band_data = band_power(b,:);
        
        % --- COLUMN 1: MORAL ---
        subplot(5,4,(b-1)*4 + 1); hold on;
        conds = [0 1];
        means = zeros(1,2);
        errors = zeros(1,2);
        for i = 1:2
            idx = moral_num == conds(i);
            means(i) = mean(band_data(idx));
            errors(i) = std(band_data(idx))/sqrt(sum(idx));
        end
        bar(conds, means, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k');
        errorbar(conds, means, errors, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
        xticks([0 1]);
        xticklabels({'Non','Moral'});
        ylabel('Band Power (\muV^2)', 'Interpreter', 'tex', 'FontWeight', 'bold');
        title([band_names{b} ' - Moral'], 'Interpreter', 'none');
        set(gca, 'Box', 'off', 'TickDir', 'out', 'YGrid', 'on', 'GridAlpha', 0.3);

        % --- COLUMN 2: SOCIAL ---
        subplot(5,4,(b-1)*4 + 2); hold on;
        conds = [1 2 3];
        means = zeros(1,3);
        errors = zeros(1,3);
        for i = 1:3
            idx = social_num == conds(i);
            means(i) = mean(band_data(idx));
            errors(i) = std(band_data(idx))/sqrt(sum(idx));
        end
        bar(conds, means, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k');
        errorbar(conds, means, errors, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
        xticks([1 2 3]);
        xticklabels({'Low','Med','High'});
        ylabel('Band Power (\muV^2)', 'Interpreter', 'tex', 'FontWeight', 'bold');
        title([band_names{b} ' - Social'], 'Interpreter', 'none');
        set(gca, 'Box', 'off', 'TickDir', 'out', 'YGrid', 'on', 'GridAlpha', 0.3);

        % --- COLUMN 3: PHOTO ---
        subplot(5,4,(b-1)*4 + 3); hold on;
        conds = ["Support","Oppose"];
        means = zeros(1,2);
        errors = zeros(1,2);
        for i = 1:2
            idx = photo == conds(i);
            means(i) = mean(band_data(idx));
            errors(i) = std(band_data(idx))/sqrt(sum(idx));
        end
        bar(1:2, means, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k');
        errorbar(1:2, means, errors, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
        xticks([1 2]);
        xticklabels({'Support','Oppose'});
        ylabel('Band Power (\muV^2)', 'Interpreter', 'tex', 'FontWeight', 'bold');
        title([band_names{b} ' - Photo'], 'Interpreter', 'none');
        set(gca, 'Box', 'off', 'TickDir', 'out', 'YGrid', 'on', 'GridAlpha', 0.3);

        % --- COLUMN 4: INTERACTION (HEATMAP) ---
        subplot(5,4,(b-1)*4 + 4); hold on;
        means = zeros(2,3);
        for m = 0:1
            for s = 1:3
                idx = (moral_num == m) & (social_num == s);
                if sum(idx) > 0
                    means(m+1,s) = mean(band_data(idx));
                else
                    means(m+1,s) = NaN;
                end
            end
        end
        
        imagesc(means);
        colorbar;
        title([band_names{b} ' Interaction'], 'Interpreter', 'none');
        xlabel('Social', 'FontWeight', 'bold'); 
        ylabel('Moral', 'FontWeight', 'bold');
        set(gca, 'YTick', 1:2, 'YTickLabel', {'Non', 'Moral'}, ...
                 'XTick', 1:3, 'XTickLabel', {'Low', 'Med', 'High'}, ...
                 'TickDir', 'out');
             
        % Overlay numbers on heatmap
        [rows, cols] = size(means);
        global_mean = nanmean(means(:)); 
        for r = 1:rows
            for c = 1:cols
                val = means(r,c);
                if ~isnan(val)
                    if val < global_mean
                        t_color = 'w';
                    else
                        t_color = 'k';
                    end
                    text(c, r, sprintf('%.1f', val), 'Color', t_color, ...
                        'HorizontalAlignment', 'center', ...
                        'VerticalAlignment', 'middle', ...
                        'FontSize', 10, 'FontWeight', 'bold');
                end
            end
        end
    end
    sgtitle('Band Power Exploration Across Conditions', 'FontSize', 16, 'FontWeight', 'bold');
    
    %% ---------------- ENTROPY PLOTS ----------------
    figure('Color', 'w', 'Position', [100 300 1000 400]);
    subplot(1,3,1)
    plot_condition(samp_entropy, moral_num, [0 1], {'Non-Moral','Moral'}, 'Entropy: Moral','Entropy')
    subplot(1,3,2)
    plot_condition(samp_entropy, social_num, [1 2 3], {'Low','Medium','High'}, 'Entropy: Social','Entropy')
    subplot(1,3,3)
    plot_condition(samp_entropy, photo_num, [0 1], {'Oppose','Support'}, 'Entropy: Photo','Entropy')
    sgtitle('Neural Signal Complexity (Entropy) Across Conditions', 'FontSize', 16, 'FontWeight', 'bold');
    
    %% =========================================
    % SAVE
    % =========================================
    save_folder = fullfile(subj_path, 'processed_EEG');
    if ~exist(save_folder,'dir')
        mkdir(save_folder);
    end
    save(fullfile(save_folder,'EEG_features.mat'), 'TrialTable');
    fprintf('Saved subject %d\n', sub);
    disp('ALL DONE');
end

%% =========================================
% HELPER FUNCTIONS
% =========================================

function plot_condition(data, cond, conds, labels, title_str, ylabel_str)
    sem = @(x) std(x)/sqrt(max(1,length(x)));
    means = zeros(1,length(conds));
    errors = zeros(1,length(conds));
    
    for i = 1:length(conds)
        idx = cond == conds(i);
        means(i) = mean(data(idx));
        errors(i) = sem(data(idx));
    end
    
    bar(conds, means, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k'); hold on;
    errorbar(conds, means, errors, 'k.', 'LineWidth', 1.5, 'CapSize', 8);
    xticks(conds);
    xticklabels(labels);
    title(title_str, 'Interpreter', 'tex');
    ylabel(ylabel_str, 'Interpreter', 'tex', 'FontWeight', 'bold');
    
    % Publication-style formatting
    set(gca, 'Box', 'off', 'TickDir', 'out', 'YGrid', 'on', 'GridAlpha', 0.3);
    
    % --- Significance Bracket Example ---
    % y_max = max(means + errors) * 1.1; 
    % draw_sig_bracket(conds(1), conds(end), y_max, '*');
end

function draw_sig_bracket(x1, x2, y, star_str)
    hold on;
    bracket_h = y * 0.05; 
    plot([x1, x1, x2, x2], [y-bracket_h, y, y, y-bracket_h], '-k', 'LineWidth', 1.2);
    text(mean([x1, x2]), y + (bracket_h * 0.5), star_str, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 16, 'Color', 'k');
end
