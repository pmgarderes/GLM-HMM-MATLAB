function P = glm_likelihood_b(X, W, y)
% Inputs: X (T x M), W (M x K)
% Output: P (T x K) matrix of Bernoulli GLM probabilities

    % Compute P(t,k) = likelihood of y(t) under state k's GLM
    T = size(X,1); K = size(W,2);
    P = zeros(T, K);

    for k = 1:K
        logits = X * W(:,k);
        p_k = 1 ./ (1 + exp(-logits));
        P(:,k) = p_k.^y .* (1 - p_k).^(1 - y);  % Bernoulli likelihood
    end

    % Numerical stability
    epsilon = 1e-9;
    P = max(P, epsilon);
end