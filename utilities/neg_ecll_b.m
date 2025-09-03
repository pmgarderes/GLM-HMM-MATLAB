function [nll, grad] = neg_ecll_b(w, X, y, gamma_k, sigma2)
    % Negative expected complete-data log-likelihood and gradient
    % for a single state's GLM weights
    T = length(y);
    logits = X * w;
    p = 1 ./ (1 + exp(-logits));
    p = max(min(p, 1 - 1e-9), 1e-9);  % Numerical stability

    % NLL
    ll = gamma_k .* (y .* log(p) + (1 - y) .* log(1 - p));
    nll = -sum(ll) + 0.5 * sum(w.^2) / sigma2;

    % Gradient
    grad_ll = X' * ((p - y) .* gamma_k);
    grad = grad_ll + w / sigma2;
end