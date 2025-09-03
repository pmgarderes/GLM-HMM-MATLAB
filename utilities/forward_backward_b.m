function [gamma, xi, log_likelihood] = forward_backward_b(P, A, pi)

% Inputs: P (T x K), A (K x K), pi (K x 1)
% Outputs:
%   - gamma (T x K) posterior state probabilities
%   - xi (T-1 x K x K) posterior transitions
%   - log_likelihood (scalar)

    [T, K] = size(P);
    alpha = zeros(T, K);
    beta = zeros(T, K);
    scale = zeros(T, 1); % for log-likelihood

    % Forward pass
    alpha(1, :) = pi' .* P(1, :);
    scale(1) = sum(alpha(1, :));
    alpha(1, :) = alpha(1, :) / scale(1);

    for t = 2:T
        alpha(t, :) = (alpha(t-1, :) * A) .* P(t, :);
        scale(t) = sum(alpha(t, :));
        alpha(t, :) = alpha(t, :) / scale(t);
    end

    % Backward pass
    beta(T, :) = 1 / scale(T);
    for t = T-1:-1:1
        beta(t, :) = (beta(t+1, :) .* P(t+1, :)) * A';
        beta(t, :) = beta(t, :) / scale(t);
    end

    % Compute gamma
    gamma = alpha .* beta;
    gamma = gamma ./ sum(gamma, 2);

    % Compute xi
    xi = zeros(T-1, K, K);
    for t = 1:T-1
        denom = (alpha(t, :) * A) .* P(t+1, :) * beta(t+1, :)';
        for i = 1:K
            for j = 1:K
                xi(t, i, j) = alpha(t, i) * A(i, j) * P(t+1, j) * beta(t+1, j);
            end
        end
        xi(t, :, :) = xi(t, :, :) / sum(sum(xi(t, :, :)));
    end

    % Log-likelihood
    log_likelihood = sum(log(scale + 1e-12));
end