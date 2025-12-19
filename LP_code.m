
clear; clc; close all;

% --- Global Visualization Settings ---
set(0, 'DefaultAxesFontSize', 20);       
set(0, 'DefaultTextFontSize', 20);
set(0, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesLineWidth', 2);
set(0, 'DefaultAxesLabelFontSizeMultiplier', 1.1);
set(0, 'DefaultAxesTitleFontSizeMultiplier', 1.2);

% Global parameters

lambda0 = 1;
k = 2*pi/lambda0;
Npat = 2001;              % plot resolution
eps_db = 1e-12;

% Mainlobe exclusion half-width (deg) for sidelobe constraint sampling
ML_excl_sq  = 15;         % exclude [theta0-ML_excl, theta0+ML_excl] on phi=0 cut
ML_excl_cir = 15;         % exclude [phi0-ML_excl,  phi0+ML_excl]  on theta=90 cut

% Sidelobe sampling resolution for LP constraints (deg)
dSLL_sq  = 1.0;
dSLL_cir = 1.0;

% Square array geometry (16x16, spacing 0.3 lambda)
nx = 16; ny = 16;
dx = 0.3; dy = 0.3;

xs = ((0:nx-1) - (nx-1)/2) * dx;
ys = ((0:ny-1) - (ny-1)/2) * dy;
[X,Y] = meshgrid(xs,ys);
pos_sq = [X(:), Y(:)];
N_sq = size(pos_sq,1);

theta0_sq = 30;
phi0_sq   = 0;


% Circular array geometry (single ring, 32 elements, radius λ) 
rho = 1;
N_c = 32;
phi_el = (0:N_c-1).' * (2*pi/N_c);
pos_c  = [rho*cos(phi_el), rho*sin(phi_el)];

theta0_c = 90;
phi0_c   = 30;


a_sq_classic = steering(pos_sq, theta0_sq, phi0_sq, k);
a_c_classic  = steering(pos_c,  theta0_c,  phi0_c,  k);


% LP optimization 

% ---- Square: apply SLL constraints on phi=0, theta in [-90,90] excluding mainlobe
theta_grid_sq = -90:dSLL_sq:90;
theta_SLL_sq  = theta_grid_sq( abs(theta_grid_sq - theta0_sq) > ML_excl_sq );
phi_SLL_sq    = phi0_sq * ones(size(theta_SLL_sq));

a_sq_lp = solveLP_SLL_onCut(pos_sq, k, theta0_sq, phi0_sq, theta_SLL_sq, phi_SLL_sq);

% ---- Circular: apply SLL constraints on theta=90, phi in [-180,180] excluding mainlobe
phi_grid_c = -180:dSLL_cir:180;
phi_SLL_c  = phi_grid_c( angdiff_deg(phi_grid_c, phi0_c) > ML_excl_cir );
theta_SLL_c = theta0_c * ones(size(phi_SLL_c));

a_c_lp = solveLP_SLL_onCut(pos_c, k, theta0_c, phi0_c, theta_SLL_c, phi_SLL_c);



% Fig 9(a): Square theta cut at phi=0
[th_plot, sq_class_db] = patternTheta(pos_sq, a_sq_classic, 0, k, Npat, eps_db);
[~,       sq_lp_db]    = patternTheta(pos_sq, a_sq_lp,      0, k, Npat, eps_db);

% Fig 9(b): Circular phi cut at theta=90
[ph_plot, c_class_db]  = patternPhi(pos_c, a_c_classic, 90, k, Npat, eps_db);
[~,       c_lp_db]     = patternPhi(pos_c, a_c_lp,      90, k, Npat, eps_db);


% Define colors 
classic_color = [0.2 0.6 0.8];
opt_color = [0.8 0.3 0.1];
null_color = 'k';
marker_size = 10;
line_width = 2.5;

f9 = figure('Color','w','Position',[100 100 1000 400]);


subplot(1,2,1);
plot(th_plot, sq_class_db,'--','LineWidth',line_width); hold on;
plot(th_plot, sq_lp_db,   '-','LineWidth',line_width);
grid on; ylim([-60 5]); xlim([-90 90]);
xlabel('$\theta$ (deg)', 'Interpreter', 'latex');
ylabel('Normalized $|E|$ (dB)', 'Interpreter', 'latex');
title('Square Array', 'FontWeight', 'bold');
legend('W/O Optimization', 'LP-Optimization', 'Location', 'Best', ...
    'Interpreter', 'latex');


subplot(1,2,2);
plot(ph_plot, c_class_db,'--','LineWidth', line_width); hold on;
plot(ph_plot, c_lp_db,   '-','LineWidth', line_width);
grid on; ylim([-60 5]); xlim([-180 180]);
xlabel('$\phi$ (deg)', 'Interpreter', 'latex');
ylabel('Normalized $|E|$ (dB)', 'Interpreter', 'latex');
title('Circular Array', 'FontWeight', 'bold');
legend('W/O Optimization', 'LP-Optimization', 'Location', 'Best', 'Interpreter', 'latex');

yl = ylim;
leftEdge  = wrapTo180Deg(phi0_c - ML_excl_cir);
rightEdge = wrapTo180Deg(phi0_c + ML_excl_cir);


set(f9, 'PaperPositionMode', 'auto'); 
drawnow; 
exportgraphics(f9, 'Figure_9_new.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none');
exportgraphics(f9, 'Figure_9_new.png', 'Resolution', 300);




%% ---------------- Local functions -----------------------------

function a = solveLP_SLL_onCut(pos, k, theta0, phi0, theta_sll, phi_sll)
    
    % y = [Re(a); Im(a)] in R^{2N}, t in R.
    %
    % Minimize t
    % s.t. ±d_mR^T y <= t and ±d_mI^T y <= t   for all sidelobe samples m
    %      d0R^T y = 1,  d0I^T y = 0

    N = size(pos,1);
    nz = 2*N + 1;     % [y; t]
    tIdx = nz;

    % ---- Objective: min t
    f = zeros(nz,1);
    f(tIdx) = 1;

    % ---- Build inequality constraints A*z <= b
    M = numel(theta_sll);
    A = zeros(4*M, nz);
    b = zeros(4*M, 1);

    for m = 1:M
        d = steering(pos, theta_sll(m), phi_sll(m), k);
        dR = [real(d); imag(d)];      
        dI = [-imag(d); real(d)];     

        r0 = 4*(m-1);

        %  dR^T y - t <= 0
        A(r0+1, 1:2*N) =  dR.';
        A(r0+1, tIdx)  = -1;

        % -dR^T y - t <= 0
        A(r0+2, 1:2*N) = -dR.';
        A(r0+2, tIdx)  = -1;

        %  dI^T y - t <= 0
        A(r0+3, 1:2*N) =  dI.';
        A(r0+3, tIdx)  = -1;

        % -dI^T y - t <= 0
        A(r0+4, 1:2*N) = -dI.';
        A(r0+4, tIdx)  = -1;
    end

    % ---- Equalities: 
    d0 = steering(pos, theta0, phi0, k);
    d0R = [real(d0); imag(d0)];
    d0I = [-imag(d0); real(d0)];

    Aeq = zeros(2, nz);
    beq = zeros(2, 1);

    Aeq(1,1:2*N) = d0R.';   beq(1) = 1;   % d0R^T y = zeta_R = 1
    Aeq(2,1:2*N) = d0I.';   beq(2) = 0;   % d0I^T y = zeta_I = 0

    % ---- Bounds 
    BIG = 1e6;
    lb = -BIG*ones(nz,1);
    ub =  BIG*ones(nz,1);
    lb(tIdx) = 0;           % t >= 0

    opts = optimoptions('linprog','Algorithm','dual-simplex','Display','off');

    z = linprog(f, A, b, Aeq, beq, lb, ub, opts);

    y = z(1:2*N);
    a = y(1:N) + 1j*y(N+1:end);
end

function d = steering(pos, theta, phi, k)
    th = deg2rad(theta);
    ph = deg2rad(phi);
    ux = sin(th)*cos(ph);
    uy = sin(th)*sin(ph);
    phase = k*(pos(:,1)*ux + pos(:,2)*uy);
    d = exp(-1j*phase);
end

function [th,db] = patternTheta(pos,a,phi,k,n,eps_db)
    th = linspace(-90,90,n);
    af = zeros(size(th));
    for i=1:n
        d = steering(pos, th(i), phi, k);
        af(i) = abs(d' * a);
    end
    af = af / max(af);
    db = 20*log10(af + eps_db);
end

function [ph,db] = patternPhi(pos,a,theta,k,n,eps_db)
    ph = linspace(-180,180,n);
    af = zeros(size(ph));
    for i=1:n
        d = steering(pos, theta, ph(i), k);
        af(i) = abs(d' * a);
    end
    af = af / max(af);
    db = 20*log10(af + eps_db);
end

function d = angdiff_deg(a, b)
    % minimal absolute angular difference in degrees, for arrays
    d = abs(mod(a - b + 180, 360) - 180);
end

function x = wrapTo180Deg(x)
    x = mod(x + 180, 360) - 180;
end
