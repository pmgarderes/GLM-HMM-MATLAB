function [params, ll_hist] = run_em_b(X, y,  params)

% Inputs: X, y, K, max_iter, tolerance
% Output: fitted params, log_likelihood history
% - Loops E-step and M-step until convergence


    T = params.T; 
    M = params.M; 
    K = params.K;
    params = hmmbhv_initialize_model_b(params);
    % Eit : make state transition more flexible
    epsilon = 0.1;
    params.A = (1 - epsilon) * eye(K) + epsilon / (K - 1) * (ones(K) - eye(K));
    params.A = params.A ./ sum(params.A, 2);  % row normalize

    ll_hist = zeros(params.max_iter, 1);
    prev_ll = -inf;

    for iter = 1:params.max_iter
        % E-step
        P = glm_likelihood_b(X, params.W, y);
        if params.K>1
            [gamma, xi, logL] = forward_backward_b(P, params.A, params.pi);
        else
            gamma = ones(T,1);  xi = ones(T,1); logL= -1;
        end

        % M-step
        params = m_step_b(X, y, gamma, xi, params);

        % Log-likelihood tracking
        ll_hist(iter) = logL;
        if params.display_output
            fprintf('Iter %d: Log-likelihood = %.6f\n', iter, logL);
        end

        % Check convergence
        if abs(logL - prev_ll) < params.tol
            ll_hist = ll_hist(1:iter);
            break;
        end
        prev_ll = logL;
    end
end