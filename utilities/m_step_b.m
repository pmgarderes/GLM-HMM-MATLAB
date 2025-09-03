function params = m_step_b(X, y, gamma, xi, params)

% Inputs: X, y, gamma, xi, params
% Outputs: updated params
% - Updates:
%   - pi (from gamma(1,:))
%   - A (normalized xi)
%   - W (optimize via BFGS/fminunc with Gaussian prior)

    [T, K] = size(gamma);
    M = size(X, 2);

    % Update initial state distribution
    params.pi = gamma(1, :)';
    params.pi = params.pi / sum(params.pi);

    % Update transition matrix A
    A_new = zeros(K, K);
    for i = 1:K
        for j = 1:K
            A_new(i, j) = sum(squeeze(xi(:, i, j)));
        end
        A_new(i, :) = A_new(i, :) / sum(A_new(i, :) + 1e-12);
    end
    params.A = A_new;

    % Update GLM weights W
    W_new = zeros(M, K);
    options = optimoptions('fminunc', 'GradObj', 'on', 'Display', 'off');

    for k = 1:K
        w_init = params.W(:, k);
        g_k = gamma(:, k);
        objective = @(w) neg_ecll_b(w, X, y, g_k, params.hyperparams.sigma2);
        W_new(:, k) = fminunc(objective, w_init, options);
    end
    params.W = W_new;
end