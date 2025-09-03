%=======================
% Main Script (example.m)
%=======================
% - Loads data
% - Initializes model
% - Runs EM loop
% - Plots results

codepath = 'C:\code\GLM-HMM-MATLAB';
addpath(genpath(codepath))


% Example usage:
% Simulate or load X (T x M), y (T x 1) binary choices
% Here, we simulate synthetic data for demonstration

params.T = 1000;        % Number of trials
params.M = 4;           % Number of covariates
params.K = 3;           % Number of hidden states
params.max_iter = 100;  % EM iterations
params.tol = 1e-4;      % Convergence tolerance
params.p_correct = 0.85;       % Precision simulation ( from 0.5 to 1)  % Accuracy level for stimulus-driven state
params.pop_weigths = 1;     % pre-populate weigths of the GLMs for each state ( in initialize_model) 
params.display_output = 0;   % boolean 

% random seed for reproducibility
rng(2025)

% Initialize model and simulate data
params = hmmbhv_initialize_model_b( params);
[y, z, X] = simulate_glmhmm_b(params);

% params.A = 0.90 * eye(K) + 0.1 / (K-1) * (ones(K) - eye(K));
% params.A = params.A ./ sum(params.A, 2);  % row-normalize

% Run EM
[params, ll_hist] = run_em_b(X,  y,  params);

% Display final log-likelihood
fprintf('Final Log-likelihood: %.6f\n', ll_hist(end));

% Plot inferred states (soft assignments)
% P = glm_probabilities(X, params.W);
P = glm_likelihood_b(X, params.W,y);
[gamma, ~, ~] = forward_backward_b(P, params.A, params.pi);
plot_results_hmmbhv_b(gamma, y, z);


