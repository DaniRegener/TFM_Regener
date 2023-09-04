%% Plot the MSE as function of different parameters for the 2D test case
% Author:     Daniel Regener Roig
% Supervisor: Arnau Miró Jané
% Date:       04/09/2023
% Program developed for the master's thesis "TFM-220MUAERON- 
% Advanced methods for numerical simulations of turbulent flows"
% ESEIAAT - UPC

% Description: This program uses the data generated by run_some_f_test.sh 
% and stored in library_f/ to observe the stability of different cases 
% upon varying f.


close all
addpath("library_f/")

% These the three blocks presented in the report of the thesis, uncomment
% the one you desire and comment the others
% Please ensure that sch variable is consistent with the names of the 
% files in library_f/. 
% -> A legg stands for the legend and the 
% -> names_X are for the figures. 
% -> poisson4ite is the number of poisson calls that does each scheme per
%    iteration


% BLOCK 1 Adams Bashforth + Leapfrog
% sch = {'AB','LF'};
% poisson4ite = [1,1];
% colors = {[ 0 0 0 ],...
%     [ 0 0 0 ]};
% lines = {':',':'};
% markers = {'+','o'};
% legg = {'AB','LF'};
% name_1 = "dt_lf.pdf";
% name_2 = "poisson_lf.pdf";
% name_3 = "f_lf.pdf";


% BLOCK2 Consistent ERK and LSERK
% sch = {'AB','LF','RK2_midpoint','RK3_SSP',...
%     'RK4_consistent','RK4s6',...
%     'RK4s14'};
% poisson4ite = [1,1,2,3,4,6,14];
% colors = {[ 0 0 0 ],...
%     [ 0 0 0 ],...
%     [0 0.4470 0.7410],...
%     [0.8500 0.3250 0.0980],...
%     [0.9290 0.6940 0.1250],...
%     [0.4940 0.1840 0.5560],...
%     [0.4660 0.6740 0.1880],...
%     [0.3010 0.7450 0.9330],...
%     [0.6350 0.0780 0.1840]};
% lines = {':',':','-','-','-','-','-','-','-'};
% markers = {'+','o','s','s','s','s','s','s','s'};
% legg = {'AB','LF','RK2 midpoint','RK3 SSP','RK4 classic','RK4s6','RK4s14'};
% name_1 = "dt_various_definitive.pdf";
% name_2 = "poisson_various_definitive.pdf";
% name_3 = "f_various_definitive.pdf";


% BLOCK3 Approximated ERK
sch = {'AB','LF','RK2_midpoint','RK2_midpoint_a1','RK3_SSP','RK3_SSP_a1','RK3_SSP_a2','RK4_consistent',...
    'RK4_a1','RK4_classic_a2'};
poisson4ite = [1,1,2,1,3,1,1,4,1,1];
colors = {[ 0 0 0 ],...
    [ 0 0 0 ],...
    [0 0.4470 0.7410],...
    [0 0.4470 0.7410],...
    [0.8500 0.3250 0.0980],...
    [0.8500 0.3250 0.0980],...
    [0.8500 0.3250 0.0980],...
    [0.9290 0.6940 0.1250],...
    [0.9290 0.6940 0.1250],...
    [0.9290 0.6940 0.1250]};
lines = {':',':','-','-.','-','-.','--','-','-.','--'};
markers = {'+','o','s','^','s','^','d','s','^','d'};
legg = {'AB','LF','RK2 midpoint','RK2 midpoint O2','RK3 SSP',...
    'RK3 SSP O2','RK3 SSP OP','RK4 classic','RK4 classic O2','RK4 classic OP'};
name_1 = "dt_approx_definitive.pdf";
name_2 = "poisson_approx_definitive.pdf";
name_3 = "f_approx_definitive.pdf";


for i=1:length(sch) % for each desire scheme
    % Read MSE file
    fileID = fopen(strcat(sch{i},"_MSE.txt"),'r');
    formatSpec = 'MSE:  %f\n';
    error.(sch{i}) = fscanf(fileID,formatSpec);
    fclose(fileID);
    
    % Read f file
    fileID = fopen(strcat(sch{i},"_f.txt"),'r');
    formatSpec = 'f:  %f\n';
    f.(sch{i}) = fscanf(fileID,formatSpec);
    fclose(fileID);
    
    % Read iterations file
    fileID = fopen(strcat(sch{i},"_ites.txt"),'r');
    formatSpec = 'Number of iterations:  %f\n';
    ite.(sch{i}) = fscanf(fileID,formatSpec);
    fclose(fileID);
    
    % Read average dt file
    fileID = fopen(strcat(sch{i},"_avgdt.txt"),'r');
    formatSpec = 'Average dt:  %f\n';
    dt.(sch{i}) = fscanf(fileID,formatSpec);
    fclose(fileID);
    
    % Cut the diverged part
    j = 1;
    while 1
        if error.(sch{i})(j+1) > 10*error.(sch{i})(j)
             j = j +1;
              break;
        elseif j == length(error.(sch{i}))-1
            j = j + 1;
            break;
        end
        j = j + 1;
    end
    error.(sch{i}) = error.(sch{i})(1:j);
    f.(sch{i}) = f.(sch{i})(1:j);
    ite.(sch{i}) = ite.(sch{i})(1:j);
    dt.(sch{i}) = dt.(sch{i})(1:j);
end

% plot the MSE as function of the mean dt
figure
hold on
plot(dt.AB,error.AB,'k:','Marker','+')
plot(dt.LF,error.LF,'k:','Marker','o')
for i=3:length(sch)
    plot(dt.(sch{i}),error.(sch{i}),'color',colors{i},'marker',...
        markers{i},'linestyle',lines{i})
end
grid minor
set(gca,'yscale','log')
xlabel('Mean $\Delta t$','Interpreter','latex')
ylabel('MSE','Interpreter','latex')
legend(legg,'Interpreter','latex','location','northeastoutside')
set(gcf,'Position',[100 100 640 320])
exportgraphics(gca,name_1,'Resolution',300)

% plot the MSE as function of the calls to the solver
figure
hold on
plot(ite.AB,error.AB,'k:','Marker','+')
plot(ite.LF,error.LF,'k:','Marker','o')
for i=3:length(sch)
    plot(ite.(sch{i})*poisson4ite(i),error.(sch{i}),'color',colors{i},...
        'marker',markers{i},'linestyle',lines{i})
end
grid minor
set(gca,'yscale','log')
xlabel('Number of Poisson solvings','Interpreter','latex')
ylabel('MSE','Interpreter','latex')
xlim([0,300])
ylim([2e-8,4e-6])
legend(legg,'Interpreter','latex','location','northeast')
set(gcf,'Position',[100 100 640 320])
exportgraphics(gca,name_2,'Resolution',300)

% plot the MSE as function of f
figure
hold on
plot(f.AB,error.AB,'k:','Marker','+')
plot(f.LF,error.LF,'k:','Marker','o')
for i=3:length(sch)
    plot(f.(sch{i}),error.(sch{i}),'color',colors{i},'marker',markers{i},'linestyle',lines{i})
end
grid minor
set(gca,'yscale','log')
xlabel('Security factor $f$','Interpreter','latex')
ylabel('MSE','Interpreter','latex')
legend(legg,'Interpreter','latex','location','northeastoutside')
set(gcf,'Position',[100 100 640 320])
exportgraphics(gca,name_3,'Resolution',300)