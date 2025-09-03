function params = Characterize_GLMHMM_Decompositoin(params, Fullbhv)

    
% params.GLMHMM.M = 4;           % Number of covariates
% params.GLMHMM.K = 3;           % Number of hidden states
% params.GLMHMM.max_iter = 100;  % EM iterations
% params.GLMHMM.tol = 1e-4;      % Convergence tolerance
% params.GLMHMM.p_correct = 0.85;       % Precision simulation ( from 0.5 to 1)  % Accuracy level for stimulus-driven state
% params.GLMHMM.pop_weigths = 1;     % pre-populate weigths of each state's GLM (3 state demo simulation) 
% params.GLMHMM.display_output = 1;   % boolean 

params.GLMHMM.K = 4;           % Number of hidden states
params.GLMHMM.max_iter = 100;  % EM iterations
params.GLMHMM.tol = 1e-3;      % Convergence tolerance
params.GLMHMM.pop_weigths = 1;     % boolean pre-populate weigths of each state's GLM (3 state demo simulation) 
params.GLMHMM.display_output = 1;   % boolean 
% generate the covariates 
Stim = Fullbhv.StimIntensity.*sqrt(Fullbhv.nWhiskers); 
Stim(isnan(Stim))=-1; %Stim = zscore(Stim);
y = ismember(Fullbhv.TrOutcome ,[1,3]);
% fit the model with an "optimal K =5 states . 
Results = Fit_GLMHMM_GNGdata(Stim,y, params.GLMHMM); 

% iterate across active sessions
 [uniqSess, ~, Csess]  = unique(Fullbhv.cexp);
 params.GLMHMM.K = 3;   
 Sesscnt = 0;  
 clear KsPerf
for sess =1 :length(uniqSess)
    vecSess = find(Csess==sess);
    if (sum(Fullbhv.TrOutcome(vecSess)==1)>10) % at least 10 hits
        Sesscnt = Sesscnt+1;
        Stim = Fullbhv.StimIntensity(vecSess); % .*sqrt(Fullbhv.nWhiskers(vecSess));
        Stim(isnan(Stim))=-1; %Stim = zscore(Stim);
        y = ismember(Fullbhv.TrOutcome(vecSess),[1,3]); 
        for kk = 1:8
            params.GLMHMM.K = kk;
            params.GLMHMM.display_output = 0;   % boolean
            params.GLMHMM.pop_weigths = 0 ;
            Results = Fit_GLMHMM_GNGdata(Stim,y, params.GLMHMM);
            KsPerf(kk,Sesscnt) = Results.Fpred;
            Aperf(sess) =(sum(Fullbhv.TrOutcome(vecSess)==1));
            
        end
    end
end

plotShadedError( 100*KsPerf)
hold on; title('N state accuracy per session')
xlabel('state')
ylabel('accuracy (%)')

