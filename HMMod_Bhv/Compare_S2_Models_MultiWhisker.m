function Compare_S2_Models_MultiWhisker(S1_data, S2_data, time_vector)
    % Compare_S2_Models_MultiWhisker - Compares different multi-whisker models of S2 response.
    %
    % INPUTS:
    %   S1_data      - 3D Matrix of S1 responses (trials x time x whiskers).
    %   S2_data      - Matrix of S2 responses (trials x time).
    %   time_vector  - Time vector for the responses (1 x N).
    %
    % OUTPUT:
    %   Model Comparison Table (displayed and saved)
    
    % Define available multi-whisker models
    models = {
        'Linear Sum', @Model_Linear_Sum, [];
        'Multiplicative Interaction', @Model_Multiplicative_Multi, 1.5; % Initial gamma
        'Gain Control (Normalization)', @Model_GainControl_Multi, 0.5;  % Initial beta
        'Thresholding (Gating)', @Model_Threshold_Multi, 0.2;            % Initial theta
        'Sigmoid Non-Linearity', @Model_Sigmoid_Multi, [10, 0.2];       % Initial [k, theta]
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
    title('Multi-Whisker Model Comparison - S2 Predictions');
    xlabel('Time (ms)');
    ylabel('Amplitude');
    legend(legend_entries);
    hold off;
    
    % Display model comparison
    ModelComparison = table(ModelNames, R2_values, AIC_values, OptParams);
    disp(ModelComparison);
    
    % Save results
    save('Model_Comparison_MultiWhisker.mat', 'ModelComparison');
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
