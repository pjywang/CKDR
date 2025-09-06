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
max_levels = 10; % Safety break to prevent infinite loops
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

%%
% Define directory and datasets
directory = fullfile(project_root, 'datasets', 'MLRepo');
name = "ravel";
x_dir = "refseq/taxatable.txt";
y_dir = "task-";
y_names = py.dict(pyargs( ...
    'ravel', py.list({"nugent-score", "ph"}) ...
    ));

% Read counts and Y1 data
counts_path = fullfile(directory, name, x_dir);
y_filename = y_dir + char(y_names{name}{1}) + ".txt";
y_path = fullfile(directory, name, y_filename);

fprintf('Loading counts from: %s\n', counts_path);
fprintf('Loading labels from: %s\n', y_path);

counts = pd.read_csv(counts_path, pyargs('sep', '\t'));
Y1 = pd.read_csv(y_path, pyargs('sep', '\t'));
%%
% Convert counts and Y1 to a MATLAB table
Y1_table = table(Y1);
sample_ids = Y1_table.("#SampleID");
counts_table = table(counts);
%%

% Filter taxa present in less than 5 samples
counts2 = counts_table(:, sample_ids);
taxa_presence = table2array(counts2) > 0;
taxa_sample_count = sum(taxa_presence, 2);
taxa = counts_table.('#OTU ID')(taxa_sample_count >= 5);
counts2 = counts2(taxa_sample_count >= 5, :);
%%
% Convert counts2 and Y1 to numpy arrays
x1 = table2array(counts2).';
y1 = table2array(Y1_table(:, 'Var'));
%%
% Normalize x1
x1 = double(x1);  % Convert to MATLAB double for manipulation
row_sums = sum(x1, 2);
x1 = x1 ./ row_sums;

% Print results
zero_ratio = sum(x1(:) == 0) / numel(x1);
data_shape = size(x1);
y_range_min = min(y1);
y_range_max = max(y1);

fprintf("Zero ratio: %f\n", zero_ratio);
fprintf("Data shape: [%d, %d] with y ranges %f to %f\n", data_shape(1), data_shape(2), y_range_min, y_range_max);
%%

% Test set size 20%
test_size = 0.2;
rng(2024);

%% CV for one split (example use)
% Split the data
res = train_test_split(np.array(x1), np.array(y1), pyargs('test_size', test_size, 'random_state', py.int(20241225), 'stratify', np.array(y1)));
X_train = double(res{1});
X_test = double(res{2});
y_train = double(res{3})';  % Need to transpose
y_test = double(res{4})';

[~, p] = size(X_train);

% CV for parameter selection
Tstart=tic;
W=ones(p,p); % weight matrix for graph (complete graph without edge-specific weights)
[C, CNorm]=mat2SPGgraph(W);
option.mu=1e-02;   % Smoothing Parameter, fixed
% tuning parameter
option.gammarange = logspace(-5, -2, 30); % (5.5:1.0:30)*1E-4;
option.fig=1;
%     option.gammarange=0; % no penalty -> vanilla relative shift model
%     option.gammarange=0.1; % too much penalty -> beta all equal
[beta_RS,~,gamma_opt,CV] = cv_SPG_cvrt_py('graph',y_train, X_train,[], C, CNorm, option); 
T1=toc(Tstart)  
figure(1); clf;
plot(beta_RS,'r')   

%% 100 Repetition

close all; clf;

% params
[~, p] = size(x1);
W=ones(p,p); % weight matrix for graph (complete graph without edge-specific weights)
[C, CNorm]=mat2SPGgraph(W);
option.mu=1e-02;   % Smoothing Parameter, fixed
% tuning parameter
option.gammarange = logspace(-5, -2, 30);
option.fig=1;
option.verbose=false;
%%

% start repetition
T_start=tic;
reps=100;
result_test20 = zeros(reps, 1); % out sample prediction mse
for j=1:reps
    disp(['Running sim #', num2str(j)]);
    res = train_test_split(np.array(x1), np.array(y1), ...
        pyargs('test_size', test_size, 'random_state', py.int(20241224+j), ...
        'stratify', np.array(y1)));
    X_train = double(res{1});
    X_test = double(res{2});
    y_train = double(res{3})'; 
    y_test = double(res{4})';

    % CV for parameter selection
    [beta_RS,~,gamma_opt,CV] = cv_SPG_cvrt_py('graph',y_train, X_train,[], C, CNorm, option);
    
    % refit with optimal gamma
    [beta,~,~,~,~] = SPG('graph', y_train, X_train, gamma_opt, 0, C, CNorm, option);
    result_test20(j, 1) = mean((y_test-X_test*beta).^2);

end
T_rep = toc(T_start) % 205min

disp(['MSE  ', num2str(mean(result_test20)), '+/-', num2str(std(result_test20))]);
close;
save("JP_realdata_100reps.mat", "result_test20");

