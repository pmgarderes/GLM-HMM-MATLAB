function [y, z, X] = simulate_glmhmm_b(params)
    % Simulate GLM-HMM data with parameterizable accuracy
    % States:
    %   1 = stimulus-driven (90% correct)
    %   2 = biased left
    %   3 = biased right

    p_correct = params.p_correct;
    T = params.T; 
    z = zeros(T, 1);
    y = zeros(T, 1);
    

    stim = randi([0 1], T, 1)*2-1 ;%X(:,1).*X(:,1).*(randi([0 1], T, 1)*2-1); % linspace(-2, 2, T)' + 0.2 * randn(T, 1);  % stimulus
    bias = ones(T, 1);                               % bias term

    
    % Sample latent states
    z(1) = randsample(3, 1, true, params.pi);
    for t = 2:T
        z(t) = randsample(3, 1, true, params.A(z(t-1), :));
    end

    % Simulate choices
    for t = 1:T
        x_stim = stim(t);
        switch z(t)
            case 1 % stimulus-driven
                p = p_correct * double(x_stim > 0) + (1 - p_correct) * double(x_stim <= 0);
            case 2 % biased left
                p = 1-p_correct;
            case 3 % biased right
                p = p_correct;
        end
        y(t) = rand < p;
    end
%     y(y==0) = -1; 
    % Compute history covariates
    prev_choice = [0; 2*y(1:end-1)-1];
%     prev_choice = max(prev_choice,0);
    win_stay = [0; (2*y(1:end-1)-1) .* (2*(y(1:end-1)==(stim(1:end-1)>0))-1)];

    % Form X with 4 covariates: stimulus, bias, prev_choice, win_stay
    X = [stim, bias, prev_choice, win_stay];
end