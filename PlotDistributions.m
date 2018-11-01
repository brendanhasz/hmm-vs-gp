
%% Plot distributions for GP and HMM models

color = 1/255*[159 201 235];
vpad = 1.2;
lw = 3;
lcol = 0.6*[1 1 1];

% Plots for GP
figure
subplot(4,2,1)
    xx = linspace(0, 10, 300);
    yy = gampdf(1./xx, 3, 1);
    plot(xx, yy, 'LineWidth', lw, 'Color', color);
    hold on
    ylim = [0 max(yy)*vpad];
    plot(xlim, [0 0], 'Color', lcol)
    plot([0 0], ylim, 'Color', lcol)
    title('InvGamma(3, 1)')
    axis off
subplot(4,2,2)
    xx = linspace(0, 4, 300);
    yy = normpdf(xx, 0, 1);
    plot(xx, yy, 'LineWidth', lw, 'Color', color);
    hold on
    ylim = [0 max(yy)*vpad];
    plot(xlim, [0 0], 'Color', lcol)
    plot([0 0], ylim, 'Color', lcol)
    title('HalfNormal(0, 1)')
    axis off
subplot(4,2,3)
    xx = linspace(0, 20, 300);
    yy = normpdf(log(xx), 1, 0.8);
    plot(xx, yy, 'LineWidth', lw, 'Color', color);
    hold on
    ylim = [0 max(yy)*vpad];
    plot(xlim, [0 0], 'Color', lcol)
    plot([0 0], ylim, 'Color', lcol)
    title('LogNormal(1, 0.8)')
    axis off
subplot(4,2,4)
    xx = linspace(-4, 4, 300);
    yy = normpdf(xx, 0, 1);
    plot(xx, yy, 'LineWidth', lw, 'Color', color);
    hold on
    %ylim = [0 max(yy)*vpad];
    plot(xlim, [0 0], 'Color', lcol)
    %plot([0 0], ylim, 'Color', lcol)
    title('Normal(0, 1)')
    axis off

% Plots for HMM
%figure
subplot(4,2,5)
    xx = linspace(0, 1, 300);
    yy = betapdf(xx, 1.2, 1.2);
    plot(xx, yy, 'LineWidth', lw, 'Color', color);
    hold on
    ylim = [0 max(yy)*vpad];
    plot(xlim, [0 0], 'Color', lcol)
    plot([0 0], ylim, 'Color', lcol)
    plot([1 1], ylim, 'Color', lcol)
    title('Beta(1.2, 1.2)')
    axis off
subplot(4,2,6)
    xx = linspace(0, 20, 300);
    yy = gampdf(xx, 7.5, 1);
    plot(xx, yy, 'LineWidth', lw, 'Color', color);
    hold on
    ylim = [0 max(yy)*vpad];
    plot(xlim, [0 0], 'Color', lcol)
    plot([0 0], ylim, 'Color', lcol)
    title('Gamma(7.5, 1)')
    axis off
subplot(4,2,7)
    xx = linspace(0, 1, 300);
    yy = 1./(xx.*(1-xx)).*normpdf(log(xx./(1-xx)), -0.5, 1);
    plot(xx, yy, 'LineWidth', lw, 'Color', color);
    hold on
    ylim = [0 max(yy)*vpad];
    plot(xlim, [0 0], 'Color', lcol)
    plot([0 0], ylim, 'Color', lcol)
    plot([1 1], ylim, 'Color', lcol)
    title('LogitNormal(-0.5, 1)')
    axis off
