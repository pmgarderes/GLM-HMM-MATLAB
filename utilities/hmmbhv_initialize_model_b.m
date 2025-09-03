function params = hmmbhv_initialize_model_b(params)
% Inputs: K (number of states), M (number of features), params for 
% Outputs: params struct with fields:
%   - pi (K x 1 initial state distribution)
%   - A (K x K transition matrix)
%   - W (M x K GLM weights)
%   - hyperparams (struct with sigma2, alpha)

    K = params.K;
    M = params.M;
    % Initialize pi uniformly
    params.pi = ones(K, 1) / K;

    % Initialize transition matrix with high self-transition probability
    base_trans = 0.95;
    noise = 0.01 * randn(K, K);
    A = base_trans * eye(K) + noise;
    A = max(A,0.01); % no negative probabilities
    A = A ./ sum(A, 2); % normalize rows to sum to 1
    params.A = A;

    % Initialize GLM weights with small Gaussian noise
%     rng(2025) % seed random to make it repeatable 
    params.W = randn(M, K) * 0.1;
        
    if params.pop_weigths 
        params.W(:,1) = [2; 0.1; 0.1; 0.1];   % strong stimulus-driven
        params.W(:,2) = [0.1; -2; 0.1; 0.1];  % left bias
        params.W(:,3) = [0.1; 2; 0.1; 0.1];  % right bias
    end

    % Set hyperparameters
    params.hyperparams.sigma2 = 2.0; % variance for Gaussian prior
    params.hyperparams.alpha = 2.0;  % concentration for Dirichlet prior
