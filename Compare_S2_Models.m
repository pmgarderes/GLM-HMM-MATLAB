function Compare_S2_Models(S1_data, S2_data, time_vector)
    % Compare_S2_Models - Compares different models of S2 response against real data.
    %
    % INPUTS:
    %   S1_data      - Matrix of S1 responses (trials x time).
    %   S2_data      - Matrix of S2 responses (trials x time).
    %   time_vector  - Time vector for the responses (1 x N).
    %
    % OUTPUT:
    %   Model Comparison Table (displayed and saved)
    
    % Define available models and their initial parameters
    models = {
        'Linear', @Model_Linear, [];
        'Multiplicative', @Model_Multiplicative, 1.5; % Initial gamma
        'Gain Control', @Model_GainControl, 0.5;      % Initial beta
        'Threshold', @Model_Threshold, 0.2;            % Initial theta
        'Sigmoid', @Model_Sigmoid, [10, 0.2];         % Initial [k, theta]
    };
    
    % Initialize results table
    ModelNames = models(:, 1);
    R2_values = zeros(length(models), 1);
    AIC_values = zeros(length(models), 1);
    OptParams = cell(length(models), 1);
    
    % Optimization Options
    options = optimset('Display', 'off');
    
    % Create Figure for Comparison
    figure;
    subplot(1, 2, 1);
    hold on;
    plot(time_vector, mean(S2_data, 1), 'k', 'LineWidth', 2);
    legend_entries = {'Real S2'};
    
    % Apply each model
    for m = 1:length(models)
        model_name = models{m, 1};
        model_func = models{m, 2};
        init_params = models{m, 3};
        
        fprintf('Fitting model: %s\n', model_name);
        
        % Optimize model parameters
        if isempty(init_params)
            % No optimization for linear model
            S2_pred = model_func(S1_data, time_vector);
            OptParams{m} = [];
        else
            % Optimize model using fminsearch
            [best_params, ~] = fminsearch(@(p) Model_Error(p, model_func, S1_data, S2_data, time_vector), init_params, options);
            OptParams{m} = best_params;
            S2_pred = model_func(S1_data, time_vector, best_params);
        end
        
        % Fit model to S2 data
        [R2, AIC] = FitModel(S2_data, S2_pred);
        R2_values(m) = R2;
        AIC_values(m) = AIC;
        
        % Plotting for visualization
        plot(time_vector, mean(S2_pred, 1), 'LineWidth', 1.5);
        legend_entries{end+1} = sprintf('%s (R²=%.3f)', model_name, R2);
    end
    
    % Plot settings
    title('Model Comparison - S2 Predictions');
    xlabel('Time (ms)');
    ylabel('Amplitude');
    legend(legend_entries);
    hold off;
    
    % Display model comparison
    ModelComparison = table(ModelNames, R2_values, AIC_values, OptParams);
    disp(ModelComparison);
    
    % Highlight Best Model in Second Plot
    subplot(1, 2, 2);
    [~, best_model_idx] = max(R2_values);
    best_model_name = ModelNames{best_model_idx};
    best_S2_pred = models{best_model_idx, 2}(S1_data, time_vector, OptParams{best_model_idx});
    
    % Plot Best Model Fit
    plot(time_vector, mean(S2_data, 1), 'k', 'LineWidth', 2);
    hold on;
    plot(time_vector, mean(best_S2_pred, 1), 'r', 'LineWidth', 2);
    title(sprintf('Best Model: %s (R²=%.3f)', best_model_name, R2_values(best_model_idx)));
    xlabel('Time (ms)');
    ylabel('Amplitude');
    legend('Real S2', best_model_name);
    
    % Save results
    save('Model_Comparison.mat', 'ModelComparison');
end

% Supporting Function: Model Error (for Optimization)
function error_value = Model_Error(params, model_func, S1_data, S2_data, time_vector)
    % Generate S2 prediction using this model
    S2_pred = model_func(S1_data, time_vector, params);
    % Calculate Mean Squared Error
    error_value = mean((S2_data - S2_pred).^2, 'all');
end

% Supporting Function: Model Fitting (Computes R² and AIC)
function [R2, AIC] = FitModel(S2_real, S2_pred)
    % FitModel - Calculates R-squared and AIC for model fit
    %
    % INPUTS:
    %   S2_real - Real S2 data (trials x time)
    %   S2_pred - Predicted S2 data (trials x time)
    %
    % OUTPUTS:
    %   R2      - R-squared of the fit
    %   AIC     - Akaike Information Criterion
    
    % Calculate Mean Squared Error
    mse = mean((S2_real - S2_pred).^2, 'all');
    sst = var(S2_real(:)) * numel(S2_real);
    
    % Calculate R²
    R2 = 1 - (mse / sst);
    
    % Calculate AIC
    n = numel(S2_real);
    k = numel(S2_pred); % Number of parameters
    L = -0.5 * n * log(mse);
    AIC = 2 * k - 2 * L;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MODELS OF S2 INTEGRATION  %%


function S2_pred = Model_Linear(S1_data, time_vector)
    % Model_Linear - Simple Linear Model for S2 Response
    %
    % INPUTS:
    %   S1_data     - Matrix of S1 responses (trials x time).
    %   time_vector - Time vector (1 x N).
    %
    % OUTPUT:
    %   S2_pred     - Predicted S2 response (trials x time).
    
    % Linear model: S2 is directly proportional to S1
    S2_pred = mean(S1_data, 1);
end


function S2_pred = Model_Multiplicative(S1_data, time_vector)
    % Model_Multiplicative - Super-Additive Model for S2 Response
    %
    % INPUTS:
    %   S1_data     - Matrix of S1 responses (trials x time).
    %   time_vector - Time vector (1 x N).
    %
    % OUTPUT:
    %   S2_pred     - Predicted S2 response (trials x time).
    
    % Multiplicative model: Non-linear enhancement
    gamma = 1.5;  % Exponent for non-linearity (tunable)
    S2_pred = mean(S1_data, 1) .^ gamma;
end


function S2_pred = Model_GainControl(S1_data, time_vector)
    % Model_GainControl - Gain Control Model for S2 Response
    %
    % INPUTS:
    %   S1_data     - Matrix of S1 responses (trials x time).
    %   time_vector - Time vector (1 x N).
    %
    % OUTPUT:
    %   S2_pred     - Predicted S2 response (trials x time).
    
    % Gain control model: Divisive normalization
    beta = 0.5; % Gain control factor (tunable)
    S1_mean = mean(S1_data, 1);
    S2_pred = S1_mean ./ (1 + beta * S1_mean);
end



function S2_pred = Model_Threshold(S1_data, time_vector)
    % Model_Threshold - Thresholding Model for S2 Response
    %
    % INPUTS:
    %   S1_data     - Matrix of S1 responses (trials x time).
    %   time_vector - Time vector (1 x N).
    %
    % OUTPUT:
    %   S2_pred     - Predicted S2 response (trials x time).
    
    % Thresholding model: Only strong inputs count
    theta = 0.2; % Threshold (tunable)
    S1_mean = mean(S1_data, 1);
    S2_pred = max(0, S1_mean - theta);
end


function S2_pred = Model_Sigmoid(S1_data, time_vector)
    % Model_Sigmoid - Sigmoid Non-Linearity Model for S2 Response
    %
    % INPUTS:
    %   S1_data     - Matrix of S1 responses (trials x time).
    %   time_vector - Time vector (1 x N).
    %
    % OUTPUT:
    %   S2_pred     - Predicted S2 response (trials x time).
    
    % Sigmoid model: Smooth non-linearity
    k = 10;    % Sigmoid steepness (tunable)
    theta = 0.2; % Sigmoid midpoint (tunable)
    S1_mean = mean(S1_data, 1);
    S2_pred = 1 ./ (1 + exp(-k * (S1_mean - theta)));
end








% % function Compare_S2_Models(S1_data, S2_data, time_vector)
% %     % Compare_S2_Models - Compares different models of S2 response against real data.
% %     %
% %     % INPUTS:
% %     %   S1_data      - Matrix of S1 responses (trials x time).
% %     %   S2_data      - Matrix of S2 responses (trials x time).
% %     %   time_vector  - Time vector for the responses (1 x N).
% %     %
% %     % OUTPUT:
% %     %   Model Comparison Table (displayed and saved)
% %     
% %     % Define available models
% %     models = {
% %         'Linear', @Model_Linear;
% %         'Multiplicative', @Model_Multiplicative;
% %         'Gain Control', @Model_GainControl;
% %         'Threshold', @Model_Threshold;
% %         'Sigmoid', @Model_Sigmoid;
% %     };
% %     
% %     % Initialize results table
% %     ModelNames = models(:, 1);
% %     R2_values = zeros(length(models), 1);
% %     AIC_values = zeros(length(models), 1);
% %     
% %     figure;
% %     hold on;
% %     
% %     % Apply each model
% %     for m = 1:length(models)
% %         model_name = models{m, 1};
% %         model_func = models{m, 2};
% %         
% %         fprintf('Fitting model: %s\n', model_name);
% %         
% %         % Generate S2 prediction using this model
% %         S2_pred = model_func(S1_data, time_vector);
% %         
% %         % Fit model to S2 data
% %         [R2, AIC] = FitModel(S2_data, S2_pred);
% %         R2_values(m) = R2;
% %         AIC_values(m) = AIC;
% %         
% %         % Plotting for visualization
% %         subplot(2, 3, m);
% %         plot(time_vector, mean(S2_data, 1), 'k', 'LineWidth', 1.5); hold on;
% %         plot(time_vector, mean(S2_pred, 1), 'r', 'LineWidth', 1.5);
% %         title(sprintf('%s\nR²=%.3f, AIC=%.2f', model_name, R2, AIC));
% %         xlabel('Time (ms)');
% %         ylabel('Amplitude');
% %     end
% %     
% %     % Display model comparison
% %     ModelComparison = table(ModelNames, R2_values, AIC_values);
% %     disp(ModelComparison);
% %     
% %     % Save results
% %     save('Model_Comparison.mat', 'ModelComparison');
% % end
% % 
% % % Supporting Function: Model Fitting (Computes R² and AIC)
% % function [R2, AIC] = FitModel(S2_real, S2_pred)
% %     % FitModel - Calculates R-squared and AIC for model fit
% %     %
% %     % INPUTS:
% %     %   S2_real - Real S2 data (trials x time)
% %     %   S2_pred - Predicted S2 data (trials x time)
% %     %
% %     % OUTPUTS:
% %     %   R2      - R-squared of the fit
% %     %   AIC     - Akaike Information Criterion
% %     
% %     % Calculate Mean Squared Error
% %     mse = mean((S2_real - S2_pred).^2, 'all');
% %     sst = var(S2_real(:)) * numel(S2_real);
% %     
% %     % Calculate R²
% %     R2 = 1 - (mse / sst);
% %     
% %     % Calculate AIC
% %     n = numel(S2_real);
% %     k = 2; % Two parameters (non-linearity and scaling)
% %     L = -0.5 * n * log(mse);
% %     AIC = 2 * k - 2 * L;
% % end
% % 
% % 
% % 
% % 
