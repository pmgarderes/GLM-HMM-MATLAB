function Results = Fit_GLMHMM_GNGdata(Stim,y, params)


[y,  X] = populateX_glmhmm_b(Stim,y);

params.T = size(Stim,1);
params.M = size(X,2);


params = hmmbhv_initialize_model_b( params);


% Run EM
[params, ll_hist] = run_em_b(X,  y,  params);

% Display final log-likelihood
fprintf('Final Log-likelihood: %.6f\n', ll_hist(end));

% Plot inferred states (soft assignments)
P = glm_likelihood_b(X, params.W,y);
[gamma, ~, ~] = forward_backward_b(P, params.A, params.pi);

Results.gamma = gamma;
Results.params = params;
Results.P = P;

% make some prediciton from states
% [~, dom_state] = max(gamma');
if size(gamma,2)>1;         [val, dom_state] = max(gamma');
else; dom_state = ones(size(gamma,1),1); end
PredictedBhv = zeros(params.T,params.K);
Fpred = nan(params.T,1);
for k = 1:params.K
    % Animal perfromance in the state
    Correct = (y==1 & Stim>0) | (y==0 & Stim<=0);
    Results.performance(k) = mean( Correct(dom_state==k));
    % state prediciton of the behavior
    logits = params.W(:,k)'*X';
    PredictedBhv(:,k) = 1 ./ (1 + exp(-logits));
    PredictedBhv(:,k) = PredictedBhv(:,k)>0.5; %mean(PredictedBhv(:,k)) ;
    Fpred(dom_state==k) = PredictedBhv(dom_state==k,k) ;
    Results.accuracy(k) = mean(PredictedBhv(dom_state==k,k)==y(dom_state==k));
    Results.occupancy(k) = sum(dom_state==k)/length(dom_state);
end
Results.Fpred = mean(Fpred==y);
Results.dom_state = dom_state;


if params.display_output
    plot_results_hmmbhv_b(gamma, y, nan(params.T,1));

    figure
    subplot(2,2,1);
    imagesc(Results.params.A)
    title('P(change)')  ;   xlabel('state (from) ');      ylabel('state (to)')

    subplot(2,2,2) ;
    imagesc(Results.params.W')
    hold on; title('State Weights')
    xlabel('Covariates')
    ylabel('model')
    colorbar

    subplot(2,2,3) ;  % animals perfromance during different states
    bar(100*Results.performance)
    hold on; title('task performance')
    xlabel('state')
    ylabel('performance (%)')

    subplot(2,2,4) ;  % animals perfromance during different states
    bar(100*Results.accuracy)
    hold on; title('prediciton accuracy')
    xlabel('state')
    ylabel('performance (%)')
end

