clc;
clear;
% --- NİNAPRO DB5 (8 KANAL) ÖZNİTELİK ÇIKARICI - 7 YENİ HAREKET ---
% Exercise B (1,2,5,6,7) ve Exercise C (5,14) hareketlerini işler.

% 1. DENEKLERİ TANIMLA
subjects = {'S2', 'S5'}; % Çalışmak istediğin denekler

% 2. HAREKET VE ETİKET TANIMLAMALARI
% Python'da 0'dan 6'ya kadar etiketlendiriyoruz.
features_info = {
    {'B_1',  'E2', 1,  0}, % Ex B - 1
    {'B_2',  'E2', 2,  1}, % Ex B - 2
    {'B_5',  'E2', 5,  2}, % Ex B - 5
    {'B_6',  'E2', 6,  3}, % Ex B - 6
    {'B_7',  'E2', 7,  4}, % Ex B - 7
    {'C_5',  'E3', 5,  5}, % Ex C - 5
    {'C_14', 'E3', 14, 6}  % Ex C - 14
};

final_dataset = [];
disp('7 Hareket için 8 Kanallı veri işleme başlatılıyor...');

for s = 1:length(subjects)
    subj = subjects{s};
    disp(['--- İşlenen Denek: ', subj, ' ---']);
    
    % Bu sefer E2 ve E3 dosyalarına ihtiyacımız var
    file_E2 = [subj, '_E2_A1.mat'];
    file_E3 = [subj, '_E3_A1.mat'];
    
    try
        data_E2 = load(file_E2, 'emg', 'restimulus', 'repetition');
        data_E3 = load(file_E3, 'emg', 'restimulus', 'repetition');
    catch ME
        warning(['Dosya yüklenemedi: ', subj, '. Atlanıyor... Lütfen klasörde E2 ve E3 dosyalarının olduğundan emin ol.']);
        continue;
    end
    
    for f = 1:length(features_info)
        feat_name = features_info{f}{1};
        file_type = features_info{f}{2};
        mov_id    = features_info{f}{3};
        label     = features_info{f}{4};
        
        % Doğru dosya tipini (E2 veya E3) seçme
        if strcmp(file_type, 'E2')
            emg = data_E2.emg; stim = data_E2.restimulus; rep = data_E2.repetition;
        else
            emg = data_E3.emg; stim = data_E3.restimulus; rep = data_E3.repetition;
        end
        
        % O harekete ait pencereleri 8 kanallı olarak çıkar
        windows_8ch = extract_trials_8ch(emg, stim, rep, mov_id);
        
        num_windows = size(windows_8ch, 1);
        if num_windows == 0
            continue;
        end
        
        row_features = zeros(num_windows, 32); 
        
        % 8 Kanalın her biri için öznitelikleri hesapla
        for ch = 1:8
            ch_data = reshape(windows_8ch(:, :, ch), num_windows, size(windows_8ch, 2));
            
            rms_val = sqrt(mean(ch_data.^2, 2));
            mav_val = mean(abs(ch_data), 2);
            wl_val  = sum(abs(diff(ch_data, 1, 2)), 2);
            diff_ch = diff(ch_data, 1, 2);
            ssc_val = sum((diff_ch(:, 1:end-1) .* diff_ch(:, 2:end)) < 0, 2);
            
            col_idx = (ch-1)*4 + 1;
            row_features(:, col_idx:col_idx+3) = [rms_val, mav_val, wl_val, ssc_val];
        end
        
        labels_col = repmat(label, num_windows, 1);
        final_dataset = [final_dataset; [row_features, labels_col]];
    end
end

% CSV BAŞLIKLARINI OLUŞTURMA
col_names = {};
for ch = 1:8
    col_names = [col_names, {sprintf('RMS_ch%d', ch), sprintf('MAV_ch%d', ch), ...
                             sprintf('WL_ch%d', ch), sprintf('SSC_ch%d', ch)}];
end
col_names = [col_names, {'Label'}];

% CSV OLARAK KAYDET
emg_table = array2table(final_dataset, 'VariableNames', col_names);
writetable(emg_table, 'emg_features_7moves_8ch.csv');
disp('İşlem tamam! Yeni veri seti "emg_features_7moves_8ch.csv" olarak kaydedildi.');

% --- YARDIMCI FONKSİYON ---
function windows = extract_trials_8ch(emg_data, stim_data, rep_data, target_mov)
    num_reps = max(rep_data);
    window_size = 150; % 150 ms
    step_size = 30;    
    
    windows_list = {};
    
    for rep = 1:num_reps
        idx = find(stim_data == target_mov & rep_data == rep);
        
        if length(idx) > window_size 
            sig = emg_data(idx, 1:8); 
            sig_1000 = resample(sig, 1000, 200);
            
            start_idx = 1;
            while (start_idx + window_size - 1) <= size(sig_1000, 1)
                end_idx = start_idx + window_size - 1;
                windows_list{end+1} = sig_1000(start_idx:end_idx, :);
                start_idx = start_idx + step_size;
            end
        end
    end
    
    num_windows = length(windows_list);
    windows = zeros(num_windows, window_size, 8);
    for i = 1:num_windows
        windows(i, :, :) = windows_list{i};
    end
end