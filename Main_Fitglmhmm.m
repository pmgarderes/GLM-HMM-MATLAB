% GLM-HMM MATLAB Implementation Skeleton (Ashwood et al.)
%=======================
% Example Script (single_session_example.m)
%=======================
% This example runs GLM-HMM on a single behavioral session

codepath = 'C:\code\GLM-HMM-MATLAB';
addpath(genpath(codepath))


% Assumes user has:
% - stimIntensity: vector of stimulus strength per trial
% - trialOutcome: vector of trial outcomes ( 0 = left; 1 = right) or for  GNG ( 0 = no lick , 1 = lick)

% --- Generate Example Data --- LOAD your data here
T = 500; % number of trials 
stimIntensity = randn(T, 1);            % simulated stimulus
trialOutcome = round(rand(T,1));        % simulated outcomes (hit/miss/FA)



params.GLMHMM.K = 5;
params.GLMHMM.display_output = 0;   % boolean
params.GLMHMM.pop_weigths = 0 ;

params.GLMHMM.max_iter = 100;  % EM iterations
params.GLMHMM.tol = 1e-3;      % Convergence tolerance
params.GLMHMM.pop_weigths = 1;     % boolean pre-populate weigths of each state's GLM (3 state demo simulation) 
params.GLMHMM.display_output = 1;   % boolean 



% Run GLM- HMM ( 2AFC data or GNG ) 
Results = Fit_GLMHMM_GNGdata(stimIntensity,trialOutcome, params.GLMHMM); 
