function [Fullbhv, params] = Gather_SensoryndNonSensory_States(Fullbhv, AllBhv, data, params) 


% params.GLMHMM.K = 4;           % Number of hidden states
params.GLMHMM.max_iter = 100;  % EM iterations
params.GLMHMM.tol = 1e-3;      % Convergence tolerance
params.GLMHMM.pop_weigths = 1;     % boolean pre-populate weigths of each state's GLM (3 state demo simulation) 
params.GLMHMM.display_output = 1;   % boolean 


% Finally, estimate the "stimulus-driven" versus NON "stimulus driven" epochs
[uniqSess, ~, Csess]  = unique(Fullbhv.cexp);
params.GLMHMM.K = 5;
params.GLMHMM.display_output = 0;   % boolean
params.GLMHMM.pop_weigths = 0 ;
Sesscnt = 0;
clear KsPerf
NoStimStateList = [] ;
StimStateList = [] ;
for sess =1 :length(uniqSess)
    vecSess = find(Csess==sess);
    vecArousal  =cat(1,AllBhv(vecSess).yhat);
    vecArousal = smooth(vecArousal,10);
    Stim =  Fullbhv.StimIntensity(vecSess).*sqrt(Fullbhv.nWhiskers(vecSess));
    Stim(isnan(Stim))=-1; %Stim = zscore(Stim);
    y = ismember(Fullbhv.TrOutcome(vecSess),[1,3]);
    % Identify  "stimulus" states
    if (sum(Fullbhv.TrOutcome(vecSess)==1)>10) % at least 10 hits
        Results = Fit_GLMHMM_GNGdata(Stim,y, params.GLMHMM); 
        % "stimulus-driven" states 
        Stimstate = find(Results.performance>0.65);  % we could also use weigth for stim is higher than anything else
        StimTr = ismember(Results.dom_state,   Stimstate);
        StimStateList = [StimStateList, vecSess(StimTr)'];
        for k = 1:params.GLMHMM.K
            Results.StateArousal(k) = nanmean(vecArousal(Results.dom_state==k) );
        end
        AntiStimstate = find(Results.performance<0.5   &  Results.StateArousal>0.075);  % we could also use weigth for stim is higher than anything else
        StimTr = ismember(Results.dom_state,   AntiStimstate);
        NoStimStateList = [NoStimStateList, vecSess(StimTr)'];
%         plot(vecArousal); hold on 
    end
end

% Finally get the vector of latent states 
VecSNSState = zeros(length(AllBhv),1);
VecSNSState(StimStateList) = 1; VecSNSState(NoStimStateList) = -1;
Fullbhv.SNSState =VecSNSState;
    %% Session by session analysis of Sensory_Influence
%      [sensoryInfluence, pLick, didLick, yhatTR, dprimeSeries] = computeSensoryInfluenceFromYhat(data);
for sess =1 :length(uniqSess)
    vecSess = find(Csess==sess);
    cdata.bhv = data.bhv(vecSess);
    Fullbhv= extractFullbhv_fromdata_basic(cdata);
%      [sensoryInfluence, pLick, didLick, yhatTR, dprimeSeries] = computeSensoryInfluenceFromYhat(cdata);
     % Compute a trial by trial estimate
     window_sizes = [3, 5, 10, 20];
     [Newbhv, model] = estimate_sensory_influence_from_behavior_improved(Fullbhv, window_sizes);
    SEnfluence(vecSess) = Newbhv.Sensory_Influence; 
end

% using all as 1 animal 
Fullbhv= extractFullbhv_fromdata_basic(data);
[Newbhv, model] = estimate_sensory_influence_from_behavior_improved(Fullbhv, window_sizes);

Fullbhv.SNSState =VecSNSState;
for tr = 1:length(AllBhv); AllBhv(tr).SNSState = VecSNSState(tr); end
Fullbhv.SEnfluence =SEnfluence';
for tr = 1:length(AllBhv); AllBhv(tr).SEnfluence = SEnfluence(tr); end