function [y,  X] = populateX_glmhmm_b(stim,y)

    T = size(stim,1);
    bias = ones(T, 1);                               % bias term
    % Compute history covariates
    prev_choice = [0; 2*y(1:end-1)-1];
    win_stay = [0; (2*y(1:end-1)-1) .* (2*(y(1:end-1)==(stim(1:end-1)>0))-1)];

    % Form X with 4 covariates: stimulus, bias, prev_choice, win_stay
    X = [stim, bias, prev_choice, win_stay];