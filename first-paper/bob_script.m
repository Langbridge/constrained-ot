clear all
%%%%%%%%%%%%%%%%%%% Parameters of simulation
M = 10; % number of sources
N = 10; % number of targets
L = 10; % number of bars on error plots
MC = 5; % number of MC runs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Differ = 0*[1:L,MC]; % Array to plot cost
for MC_Count = 1:MC % Number of MC runs
%%%%%%%%%%%%%%%%% Initialise each simulation
% Set source and target
u = rand(1,M);
u = u./sum(u);
S1 = sum(u);
v = rand(1,N);
v = v./sum(v);
S2 = sum(v);
SC = S1/S2;
v = 2*SC*v;
v0 = v;
% T is the transport map.
% C is the cost matrix
C = rand(M,N);
gamma0 = 1.99; % exponent of regularisation term
gamma2 = 1.005; % divergence term
f = -gamma0/gamma2;
gamma = gamma2/gamma0;
K0 = exp(-gamma0*C); % why is this not exp(-C / gamma) ?
% Readjust K to include zeros in the map
    for i = 1:M
        for j = 1:N
        if 2*floor((i+j)/2)-(i+j) == 0
          K0(i,j) = 0;
        end
        end;
    end;    
% Initialise T
    T = K0;
% Keep a copy of T
    Told = T;
%%%%%%%%%%%%%%%% Martins Algorithm
    for i = 1:L  % Alg 2 line 6
        D1 = (diag(inv(diag(sum(T')))*u')); % inv(diag(sum(T')) == 1/sum_n(T') (Alg 1 line 2) !! where is d2j?
        T = D1*T; % Alg 2 line 4
        D2 = (diag(inv(diag(sum(T)))*v0')); % inv(diag(sum(T')) == 1/sum_n(T) (Alg 1 line 3) !! where is d1i?
        D2 = D2^(gamma/(1+gamma)); % Alg 2 line 3
        T = T*D2; % Alg 2 line 4
        MT = D2^(-1/gamma); % Eq 28
        v0 = v0*MT; % Eq 28
        %k = sum(v)/sum(v0);
        %v0 = k*v;
        MD = abs(T-Told);  % Alg 2 line 5
        temp = sum(sum(MD));
        differ(i,MC_Count) = temp;
        %pause
        Told = T;
    end;
MC_Count
end;
%%%%%%%%%%%%%%%%%%%%% Plot convergence
Z = diag(differ(1,:));
%Z = 1;
M1 = differ*inv(Z);
plot([1:L],log(sum(M1,2)/MC_Count),'*')
hold on
plot([1:L],log(sum(M1,2)/MC_Count),':')
xlabel('Iterations of modified algorithm')
ylabel('log of average difference metric')
grid on
title('Monte Carlo Simulations')
figure
hold on
plot([1:L],log(M1),'*')
plot([1:L],log(M1),':')
xlabel('Iterations,l, of modified algorithm')
ylabel('Normalised log of difference metric')
grid on
title('Normalised convergence')
print -deps anthony_correction.eps
%\textcolor{blue}{The y-caption is wrong. It should read 'Normalized log of the difference metric'. The x-caption should be changed to `Iterations, $l$, of modified algorithm'.