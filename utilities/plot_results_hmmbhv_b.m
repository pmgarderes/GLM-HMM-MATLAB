function plot_results_hmmbhv_b(gamma, y, z)
    % Plot the posterior state probabilities, binary choices, and true states
    T = size(gamma, 1);
    K = size(gamma, 2);

    figure;
    subplot(3,1,1);
    imagesc(gamma');
    colormap('jet');
%     colorbar;
    xlabel('Trial'); ylabel('State');
    title('Posterior State Probabilities (\gamma)');
    set(gca,'YDir','normal')

    subplot(3,1,2);
    plot(1:T, y, 'k.');
    ylim([-0.2, 1.2]);
    xlabel('Trial'); ylabel('Choice');
    title('Observed Binary Choices');

    subplot(3,1,3);
    if size(gamma,2)>1;         [val, dom_state] = max(gamma');
    else; dom_state = ones(size(gamma,1),1); end
    stairs(1:T, dom_state, 'LineWidth', 1);    
    stairs(1:T, z, 'LineWidth', 1); 
    ylim([0.5, K + 0.5]);
    xlabel('Trial'); ylabel('True State');
    title('True Latent States');
end