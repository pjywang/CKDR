%%
% JP_simulation_RSES.m
% 
% Simulation script for Relative Shift with Equi-Sparsity (RS-ES) method
% 
% This script generates simulated compositional data using the same methodology
% as in reproducibility/simulations.py to ensure consistency across different
% analysis methods. The data generation process includes:
% 
% 1. Logistic-normal distribution for X data (compositional)
% 2. Two different Y functions (Y1-Y2) representing different continuous response
% 3. Sample sizes: n = 200, 500, 1000
% 4. 100 repetitions for each combination
% 
% Author: Junyoung Park
% Date: 2025
%%

% Python-MATLAB interaction
pe = pyenv(Version="/home/users/pjywang/.conda/envs/DimRed/bin/python3.11");
% pe = pyenv(Version="C:\Users\Junyoung Park\anaconda3\envs\TF\python.exe");

% Recommendation: just use pe=pyenv(Version="your python version");
pe.Version
%%
% Import modules
np = py.importlib.import_module('numpy');
pd = py.importlib.import_module('pandas');
train_test_split = py.importlib.import_module('sklearn.model_selection').train_test_split;
KFold = py.importlib.import_module('sklearn.model_selection').KFold;
%%
% Setup paths dynamically
% This makes the script more robust and portable by actively searching for the project root.

disp('Attempting to find project root...');
current_dir = pwd; % Start from the current working directory
project_root = '';
max_levels = 15; % Safety break to prevent infinite loops
level = 0;

while level < max_levels
    % Check for a landmark file or folder that indicates the project root
    if isfile(fullfile(current_dir, 'README.md')) && isfolder(fullfile(current_dir, 'datasets'))
        project_root = current_dir;
        fprintf('Project root found at: %s\n', project_root);
        break;
    end
    
    % Move up one directory
    parent_dir = fileparts(current_dir);
    
    % If we are at the top of the filesystem, stop.
    if strcmp(parent_dir, current_dir)
        break;
    end
    
    current_dir = parent_dir;
    level = level + 1;
end

if isempty(project_root)
    error(['Could not find the project root. Please make sure you are running this script from within the "Amalgamation" project directory, ' ...
           'or that the "Amalgamation" directory is your current working directory in MATLAB.']);
end


% Add SPG path (necessary functions)
% Assuming the script is in reproducibility/other_methods/Relative-shift
spg_path = fullfile(project_root, 'reproducibility', 'other_methods', 'Relative-shift', 'SPG');
addpath(genpath(spg_path));

% Add project root to path for dataset access
addpath(project_root);

% Add project root to Python path so it can find the reproducibility module
py.sys.path().insert(int32(0), project_root);

% simulation tools
simtools = py.importlib.import_module('reproducibility.simulations');
%%

% Setup simulation parameters
MASTER_SEED = int32(20241225);  % Same as in Python
p = int32(100);                 % Number of variables
Y_noise = 0.1;                  % Noise level
sample_sizes = [200, 500, 1000];  % Sample sizes to test
reps = int32(100);              % Number of repetitions

% Set random seed for reproducibility
rng(2024);

%%
% Generate data for all combinations
fprintf('Starting data generation for RS-ES simulation...\n');

for n_idx = 1:length(sample_sizes)
    n = int32(sample_sizes(n_idx));
    fprintf('\n=== Sample size n = %d ===\n', n);
    
    for Y_func_idx = 1:2
        Y_func_names = {'Y1', 'Y2'};
        Y_func_name = Y_func_names{Y_func_idx};
        
        % Get Y function from Python
        if strcmp(Y_func_name, 'Y1')
            Y_func = simtools.Y1;
        elseif strcmp(Y_func_name, 'Y2')
            Y_func = simtools.Y2;
        end
        
        fprintf('Processing %s...\n', Y_func_name);
        
        % Storage for generated data
        X_data = cell(1, reps);
        Y_data = cell(1, reps);
        X_test_data = cell(1, reps);
        Y_test_data = cell(1, reps);
        
        for rep = 1:reps
            if mod(rep, 20) == 0
                fprintf('  Repetition %d/%d\n', rep, reps);
            end
            
            % Create RandomState with seed based on MASTER_SEED and repetition
            % This mirrors the seed generation in the Python repeat_cv function
            % to ensure the same datasets are generated across different methods
            Y_idx = Y_func_idx;
            if strcmp(Y_func_name, 'Y1')
                target_dim = 2;
            else
                target_dim = 3;
            end
            
            % Seed calculation matching Python version:
            % seed = seed - (n * Y_idx * targ) + i
            seed_offset = int32(double(n) * Y_idx * target_dim);
            rep_seed = MASTER_SEED - seed_offset + int32(rep - 1);
            
            % Create Python RandomState
            RS = np.random.RandomState(rep_seed);
            
            % Generate X data using logistic_normal function
            X = simtools.logistic_normal(n, p, pyargs('cov', 'auto', 'rho', 0.2, ...
                                                      'zero_cut', 0.5, 'seed', RS));
            
            % Generate Y data
            Y = Y_func(X, pyargs('sigma', Y_noise, 'seed', RS));

            % Generate test data
            X_test = simtools.logistic_normal(n, p, pyargs('cov', 'auto', 'rho', 0.2, ...
                                                           'zero_cut', 0.5, 'seed', RS));
            Y_test = Y_func(X_test, pyargs('sigma', Y_noise, 'seed', RS));

            % Convert to MATLAB arrays and store
            X_data{rep} = double(X);
            Y_data{rep} = double(Y);
            X_test_data{rep} = double(X_test);
            Y_test_data{rep} = double(Y_test);
        end
        
        % Save generated data
        save_dir = fullfile(project_root, 'results', 'simulation', 'rs_es_data');
        if ~exist(save_dir, 'dir')
            mkdir(save_dir);
        end
        
        filename = sprintf('data_n%d_%s.mat', n, Y_func_name);
        save_path = fullfile(save_dir, filename);
        
        fprintf('  Saving data to: %s\n', save_path);
        save(save_path, 'X_data', 'Y_data', 'X_test_data', 'Y_test_data');
        
        fprintf('  Data generation completed for %s (n=%d)\n', Y_func_name, n);
    end
end

fprintf('\n=== Data generation completed for all conditions ===\n');

%% RS-ES fitting and storing coefficients + predictions

% Example structure for the analysis loop:
Y_func_names = {'Y1', 'Y2'};  % Define this outside the loop
for n_idx = 1:length(sample_sizes)
    n = sample_sizes(n_idx);
    
    for Y_func_idx = 1:2
        Y_func_name = Y_func_names{Y_func_idx};
        
        % Load the generated data
        filename = sprintf('data_n%d_%s.mat', n, Y_func_name);
        load_path = fullfile(project_root, 'results', 'simulation', 'rs_es_data', filename);
        
        if exist(load_path, 'file')
            fprintf('Ready to process: %s\n', filename);
            % Load data and fit RS-ES model
            loaded_data = load(load_path);
            X_data = loaded_data.X_data;
            Y_data = loaded_data.Y_data;
            X_test_data = loaded_data.X_test_data;
            Y_test_data = loaded_data.Y_test_data;
            
            % Parameter fitting at the first data generation
            X = X_data{1};  % the first repetition
            Y = Y_data{1}';  % the first repetition

            % Parameter fitting
            Tstart=tic;
            W=ones(p,p); % weight matrix for graph (complete graph without edge-specific weights)
            [C, CNorm]=mat2SPGgraph(W);
            option.mu=1e-02;   % Smoothing Parameter, fixed
            % tuning parameter
            option.gammarange = logspace(-5, -2, 30); % (5.5:1.0:30)*1E-4;
            option.fig=1;
            [beta_RS,~,gamma_opt,CV] = cv_SPG_cvrt_py('graph', Y, X, [], C, CNorm, option); 
            T1=toc(Tstart);
            fprintf('  Parameter tuning completed in %.2f seconds\n', T1);
            fprintf('  Optimal gamma: %.6f\n', gamma_opt);
            
            % Storage for fitted coefficients across all repetitions
            beta_all = cell(reps, 1);
            mse_all = zeros(reps, 1);
            
            fprintf('  Fitting RS-ES model for %d repetitions...\n', reps);

            % For each repetition, fit RS-ES and store results
            for rep = 1:reps
                if mod(rep, 20) == 0
                    fprintf('    Fitting repetition %d/%d\n', rep, reps);
                end
                
                X = X_data{rep};
                Y = Y_data{rep}';

                X_test = X_test_data{rep};
                Y_test = Y_test_data{rep}';
                
                % Fit RS-ES model with optimal gamma
                [beta,~,~,~,~] = SPG('graph', Y, X, gamma_opt, 0, C, CNorm, option);

                % Calculate prediction error for the test set
                mse_test = mean((Y_test - X_test * beta).^2);
                
                % Store fitted coefficients and mses
                beta_all{rep} = beta;
                mse_all(rep) = mse_test;
            end
            
            % Save the fitted coefficients
            results_dir = fullfile(project_root, 'results', 'simulation', 'rs_es_results');
            if ~exist(results_dir, 'dir')
                mkdir(results_dir);
            end
            
            results_filename_beta = sprintf('beta_rs_n%d_%s.mat', n, Y_func_name);
            results_path = fullfile(results_dir, results_filename_beta);
            
            % Save coefficients and relevant parameters
            save(results_path, 'beta_all');
            
            results_filename_mse = sprintf('mse_rs_n%d_%s.mat', n, Y_func_name);
            results_path_mse = fullfile(results_dir, results_filename_mse);
            save(results_path_mse, 'mse_all');

            fprintf('  Fitted coefficients saved to: %s\n', results_path);
            fprintf('  MSE results saved to: %s\n', results_path_mse);
            fprintf('  Total MSE for %s (n=%d): %.4f +/- SE %.4f\n', Y_func_name, n, mean(mse_all), std(mse_all) / 10);
            fprintf('  RS-ES fitting completed for %s (n=%d)\n', Y_func_name, n);
        end
    end
end

fprintf('\n=== RS-ES Analysis Completed ===\n');
fprintf('All fitted beta coefficients have been saved to:\n');
fprintf('%s\n', fullfile(project_root, 'results', 'simulation', 'rs_es_results'));
fprintf('\nFiles saved:\n');
for n_idx = 1:length(sample_sizes)
    n = sample_sizes(n_idx);
    for Y_func_idx = 1:2
        Y_func_name = Y_func_names{Y_func_idx};
        fprintf('  - beta_rs_n%d_%s.mat\n', n, Y_func_name);
    end
end
fprintf('\nEach .mat file contains:\n');
fprintf('  - beta_all: cell array of fitted coefficients for 100 repetitions\n');
fprintf('  - gamma_opt: optimal tuning parameter\n');
fprintf('  - CV: cross-validation results\n');
fprintf('  - n, Y_func_name, reps, T1: metadata\n');
