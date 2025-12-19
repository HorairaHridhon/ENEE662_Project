clear; clc; close all;

% --- Global Visualization Settings ---
set(0, 'DefaultAxesFontSize', 20);       
set(0, 'DefaultTextFontSize', 20);
set(0, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesLineWidth', 2);
set(0, 'DefaultAxesLabelFontSizeMultiplier', 1.1);
set(0, 'DefaultAxesTitleFontSizeMultiplier', 1.2);

% --- Global parameters ---
lambda0 = 1;
k = 2*pi/lambda0;
Nt = 181;    % Theta integration resolution
Np = 361;    % Phi integration resolution
Npat = 1001; % Pattern resolution for plotting

% --- 1. Square Array Geometry ---
%    16x16 elements, 0.3 lambda spacing
nx = 16; ny = 16;
dx = 0.3 * lambda0; dy = 0.3 * lambda0;
xs = ((0:nx-1) - (nx-1)/2) * dx;
ys = ((0:ny-1) - (ny-1)/2) * dy;
[X,Y] = meshgrid(xs,ys);
pos_sq = [X(:), Y(:)];
N_sq = size(pos_sq,1);

% Square Target & Nulls
theta0_sq = 30;  phi0_sq = 0;
nulls_sq = [ -20  0;   % Theta, Phi
              70  0 ];

% --- 2. Circular Array Geometry ---
%    Single Ring (UCA), 32 elements, Radius 1 lambda
N_c = 32;
rho = 1 * lambda0;
phis_c = (0:N_c-1) * (360/N_c); % Degrees
pos_c = [rho * cosd(phis_c'), rho * sind(phis_c')];

% Circular Target & Nulls
% Note: Circular array usually points at Theta=90 (xy plane)
theta0_c = 90;  phi0_c = 30; 
nulls_c  = [ 90 -150 ]; % Null at Phi = -150 in the xy plane

% --- Build P Matrices (Integration of Manifolds) ---
fprintf('Building P matrix (Square Array)... ');
P_sq = buildP(pos_sq, Nt, Np, k);
fprintf('Done.\n');

fprintf('Building P matrix (Circular Array)... ');
P_c  = buildP(pos_c, Nt, Np, k);
fprintf('Done.\n');

% --- Compute Excitations ---
% 1. Classic (Phase Conjugate / Steering Vector)
a_sq_classic = steering(pos_sq, theta0_sq, phi0_sq, k);
a_c_classic  = steering(pos_c,  theta0_c,  phi0_c,  k);

% 2. QP Optimization (Directivity Max ONLY) 
fprintf('Solving QP Directivity (Square)...\n');
a_sq_qp = solveQP(P_sq, pos_sq, theta0_sq, phi0_sq, [], k);
fprintf('Solving QP Directivity (Circular)...\n');
a_c_qp  = solveQP(P_c,  pos_c,  theta0_c,  phi0_c,  [], k);

% 3. QP Optimization (Directivity + Nulls) 
fprintf('Solving QP Nulls (Square)...\n');
a_sq_null = solveQP(P_sq, pos_sq, theta0_sq, phi0_sq, nulls_sq, k);
fprintf('Solving QP Nulls (Circular)...\n');
a_c_null  = solveQP(P_c,  pos_c,  theta0_c,  phi0_c,  nulls_c,  k);

% --- Plotting Setup ---
classic_color = [0.2 0.6 0.8];
opt_color = [0.8 0.3 0.1];
marker_size = 10;
line_width = 2.5;

%% FIGURE 2: Maximizing Directivity
[th, sq_c0] = patternTheta(pos_sq, a_sq_classic, 0, k, Npat);
[~,  sq_q0] = patternTheta(pos_sq, a_sq_qp,      0, k, Npat);
[ph, c_c0]  = patternPhi(pos_c,  a_c_classic, 90, k, Npat);
[~,  c_q0]  = patternPhi(pos_c,  a_c_qp,      90, k, Npat);

f2 = figure('Color','w','Position',[100 100 1000 400]);

% (a) Square
subplot(1,2,1);
plot(th, sq_c0, '--', 'LineWidth', line_width, 'Color', classic_color); 
hold on;
plot(th, sq_q0, '-', 'LineWidth', line_width, 'Color', opt_color);
grid on; 
ylim([-60 2]); 
xlim([-100 100]);
xlabel('$\theta$ (deg)', 'Interpreter', 'latex');
ylabel('Normalized $|E|$ (dB)', 'Interpreter', 'latex');
title('Square Array', 'FontWeight', 'bold');
legend('W/O Optimization', 'QP-Optimization', 'Location', 'Best', ...
    'Interpreter', 'latex');

% (b) Circular
subplot(1,2,2);
plot(ph, c_c0, '--', 'LineWidth', line_width, 'Color', classic_color); 
hold on;
plot(ph, c_q0, '-', 'LineWidth', line_width, 'Color', opt_color);
grid on; 
ylim([-60 2]); 
xlim([-180 180]);
xlabel('$\phi$ (deg)', 'Interpreter', 'latex');
ylabel('Normalized $|E|$ (dB)', 'Interpreter', 'latex');
title('Circular Array', 'FontWeight', 'bold');
legend('W/O Optimization', 'QP-Optimization', 'Location', 'Best', ...
    'Interpreter', 'latex');


set(f2, 'PaperPositionMode', 'auto'); 
drawnow; 
exportgraphics(f2, 'Figure_2_new.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none');
exportgraphics(f2, 'Figure_2_new.png', 'Resolution', 300);

%% FIGURE 3: Directivity + Nulls
[th2, sq_n] = patternTheta(pos_sq, a_sq_null, 0, k, Npat);
[ph2, c_n]  = patternPhi(pos_c,  a_c_null,  90, k, Npat);


f3 = figure('Color','w','Position',[150 150 1000 400]);

% (a) Square
subplot(1,2,1);
plot(th, sq_c0, '--', 'LineWidth', line_width, 'Color', classic_color); 
hold on;
plot(th2, sq_n, '-', 'LineWidth', line_width, 'Color', opt_color);
xline(-20, 'k-', 'LineWidth', 2.5); 
xline(70, 'k-', 'LineWidth', 2.5); % Null markers
grid on; 
ylim([-60 2]); 
xlim([-100 100]);
xlabel('$\theta$ (deg)', 'Interpreter', 'latex');
ylabel('Normalized $|E|$ (dB)', 'Interpreter', 'latex');
title('Square Array with Nulls', 'FontWeight', 'bold');
legend('W/O Optimization', 'QP-Optimization', 'Location', 'Best', ...
    'Interpreter', 'latex');

% (b) Circular
subplot(1,2,2);
plot(ph, c_c0, '--', 'LineWidth', line_width, 'Color', classic_color); 
hold on;
plot(ph2, c_n, '-', 'LineWidth', line_width, 'Color', opt_color);
xline(-150, 'k-', 'LineWidth', 2.5); % Null marker
grid on; 
ylim([-60 2]); 
xlim([-180 180]);
xlabel('$\phi$ (deg)', 'Interpreter', 'latex');
ylabel('Normalized $|E|$ (dB)', 'Interpreter', 'latex');
title('Circular Array with Null', 'FontWeight', 'bold');
legend('W/O Optimization', 'QP-Optimization', 'Location', 'Best', ...
    'Interpreter', 'latex');


set(f3, 'PaperPositionMode', 'auto');
drawnow;
exportgraphics(f3, 'Figure_3_new.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none');
exportgraphics(f3, 'Figure_3_new.png', 'Resolution', 300);

%% FIGURE 4: Excitations (Normalized)
% We normalize to Peak=1 for visualization compatibility
rng(42); % Fixed seed 
idx_sq = randperm(N_sq, 10);
idx_c  = randperm(N_c,  10);

% Normalize for plot (Standard practice to compare distribution shape)
a_sq_classic_norm = a_sq_classic ./ max(abs(a_sq_classic));
a_sq_null_norm    = a_sq_null    ./ max(abs(a_sq_null));
a_c_classic_norm  = a_c_classic  ./ max(abs(a_c_classic));
a_c_null_norm     = a_c_null     ./ max(abs(a_c_null));

colors = lines(10); 
f4 = figure('Color','w','Position',[200 200 1000 400]);

% (a) Square
subplot(1,2,1); 
hold on;
for i = 1:10
    k_el = idx_sq(i);
    col = colors(i,:);
    % Classic (Diamonds)
    plot(real(a_sq_classic_norm(k_el)), imag(a_sq_classic_norm(k_el)), ...
        'd', 'MarkerSize', marker_size, 'LineWidth', 1.5, ...
        'MarkerFaceColor', col, 'MarkerEdgeColor', 'k');
    % Optimized (Circles)
    plot(real(a_sq_null_norm(k_el)), imag(a_sq_null_norm(k_el)), ...
        'o', 'MarkerSize', marker_size, 'LineWidth', 1.5, ...
        'MarkerFaceColor', col, 'MarkerEdgeColor', 'k');
end
yline(0, 'k--', 'LineWidth', 1.5); 
xline(0, 'k--', 'LineWidth', 1.5);
grid on; 
axis equal;
xlabel('Real Part', 'Interpreter', 'latex');
ylabel('Imaginary Part', 'Interpreter', 'latex');
title('Square Array Excitations', 'FontWeight', 'bold');

% Create dummy legend
h1 = plot(nan, nan, 'd', 'MarkerSize', marker_size, 'LineWidth', 1.5, ...
    'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
h2 = plot(nan, nan, 'o', 'MarkerSize', marker_size, 'LineWidth', 1.5, ...
    'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
legend([h1 h2], {'Classic', 'Optimized'}, 'Location', 'best', ...
    'Interpreter', 'latex');

% (b) Circular
subplot(1,2,2); 
hold on;
for i = 1:10
    k_el = idx_c(i);
    col = colors(i,:);
    plot(real(a_c_classic_norm(k_el)), imag(a_c_classic_norm(k_el)), ...
        'd', 'MarkerSize', marker_size, 'LineWidth', 1.5, ...
        'MarkerFaceColor', col, 'MarkerEdgeColor', 'k');
    plot(real(a_c_null_norm(k_el)), imag(a_c_null_norm(k_el)), ...
        'o', 'MarkerSize', marker_size, 'LineWidth', 1.5, ...
        'MarkerFaceColor', col, 'MarkerEdgeColor', 'k');
end
yline(0, 'k--', 'LineWidth', 1.5); 
xline(0, 'k--', 'LineWidth', 1.5);
grid on; 
axis equal;
xlabel('Real Part', 'Interpreter', 'latex');
ylabel('Imaginary Part', 'Interpreter', 'latex');
title('Circular Array Excitations', 'FontWeight', 'bold');
legend([h1 h2], {'Classic', 'Optimized'}, 'Location', 'best', ...
    'Interpreter', 'latex');


set(f4, 'PaperPositionMode', 'auto');
drawnow;
exportgraphics(f4, 'Figure_4_new.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none');
exportgraphics(f4, 'Figure_4_new.png', 'Resolution', 300);


%% Local Functions -----------------------------
function P = buildP(pos, Nt, Np, k)
    % Numerically integrates the manifold d*d' over the sphere to get P
    N = size(pos,1);
    P = zeros(N,N);
    thetas = linspace(0, 180, Nt);
    phis   = linspace(-180, 180, Np); 
    
    dth = deg2rad(thetas(2)-thetas(1));
    dph = deg2rad(phis(2)-phis(1));
    
    % Pre-compute to speed up slightly
    for i = 1:length(thetas)
        th = thetas(i);
        sin_th = sin(deg2rad(th));
        if sin_th < 1e-6, continue; end % Skip poles to avoid redundancy
        
        w = sin_th * dth * dph; % Solid angle element
        
        for ph = phis
            d = steering(pos, th, ph, k);
            P = P + w * (d * d');
        end
    end
    % Ensure Hermitian symmetry
    P = 0.5*(P + P');
    % Diagonal loading for numerical stability
    P = P + 1e-6 * eye(N); 
end

function d = steering(pos, theta, phi, k)
    % Returns steering vector (column)
    th_r = deg2rad(theta); 
    ph_r = deg2rad(phi);
    
    % Direction cosines
    u = sin(th_r) * cos(ph_r);
    v = sin(th_r) * sin(ph_r);
    
    % Phase shift
    phase = k * (pos(:,1)*u + pos(:,2)*v);
    d = exp(1j * phase); 
end

function a = solveQP(P, pos, theta0, phi0, nulls, k)
    % Solves: min(a'Pa) s.t. Re(d0'a)=1, Im(d0'a)=0, and Nulls=0
    N = size(pos,1);
    
    % Transform to Real-valued QP: x = [Re(a); Im(a)]
    % Cost function a'Pa -> x'Hx
    PR = real(P); 
    PI = imag(P);
    H  = [PR, -PI; 
          PI,  PR];
    H = 0.5*(H + H'); % Ensure symmetry
    
    f = zeros(2*N,1);
    
    % 1. Main Beam Constraint: E(theta0, phi0) = 1 + 0j
    d0 = steering(pos, theta0, phi0, k);
    u0 = real(d0); 
    v0 = imag(d0);
    
    Aeq = [u0', v0'; 
          -v0', u0'];
    beq = [1; 0];
    
    % 2. Null Constraints
    if ~isempty(nulls)
        for i = 1:size(nulls,1)
            dq = steering(pos, nulls(i,1), nulls(i,2), k);
            uq = real(dq); 
            vq = imag(dq);
            
            % E(null) = 0 => Real=0, Imag=0
            Aeq = [Aeq;
                   uq', vq';
                  -vq', uq'];
            beq = [beq; 0; 0];
        end
    end
    
    opts = optimoptions('quadprog','Algorithm','interior-point-convex',...
                        'Display','off');
    
    x = quadprog(H, f, [], [], Aeq, beq, [], [], [], [], opts);
    
    % Reconstruct complex vector
    if isempty(x)
        warning('QP Optimization failed. Returning zeros.');
        a = zeros(N,1);
    else
        a = x(1:N) + 1j*x(N+1:end);
    end
end

function [angles, db_pat] = patternTheta(pos, a, phi_cut, k, n_pts)
    angles = linspace(-90, 90, n_pts);
    pat = zeros(size(angles));
    for i = 1:n_pts
        d = steering(pos, angles(i), phi_cut, k);
        pat(i) = abs(d' * a);
    end
    pat = pat / max(pat);
    db_pat = 20*log10(pat + 1e-10);
end

function [angles, db_pat] = patternPhi(pos, a, theta_cut, k, n_pts)
    angles = linspace(-180, 180, n_pts);
    pat = zeros(size(angles));
    for i = 1:n_pts
        d = steering(pos, theta_cut, angles(i), k);
        pat(i) = abs(d' * a);
    end
    pat = pat / max(pat);
    db_pat = 20*log10(pat + 1e-10);
end
