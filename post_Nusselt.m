%% Plot the evolution in time of the Nusselt number in RB case
% Author:     Daniel Regener Roig
% Supervisor: Arnau Miró Jané
% Date:       17/08/2023
% Program developed for the master's thesis "TFM-220MUAERON- 
% Advanced methods for numerical simulations of turbulent flows"
% ESEIAAT - UPC

% Description:
% This program plots the evolution in time of the Nusselt number in RB case
% indicating also the volume in Top and Bottom, counting the average from
% convergence_time to the end of the case. For the reference, Rayleigh 
% number is 10^7 Prandtl is 0.7.

clear

time_converged = 100; % convergence time for the averages

sch = {'AB','RK4 classic O2'}; % Select the names for the legend

% Selected cases for the example case, any number of cases can be plotted
file_names = ["RB/RB_AB/nusselt.txt",...
    "RB/RB_LES3_AB/nusselt.txt",...
    ];
% We will also need the number of calls to the solver
file_names2 = ["RB/RB_AB/ins_solveP.txt",...
    "RB/RB_LES3_AB/ins_solveP.txt"];

% For each case in the list, assign three colors, for Cold, Hot and average
% Nusselt
colors{1} = [0,0,1;...
             1,0,0;...
             0,1,0];
colors{2} = [0 0.4470 0.7410;...
             0.6350 0.0780 0.1840;...
             0.4660 0.6740 0.1880];
colors{3} = [204 255 204;...
            255 204 204;...
            153 255 153]./255;


% Formats of the files to read
formatSpec1 = 'NUSSELT RB time %e NuTop %e NuBot %e\n';
formatSpec2 = "cr_info name ins_solveP00         n     %d tmin %e tmax %e tavg %e tsum %e\n";%,...


for i = 1:length(file_names)
    % read the files and save the data
    fileID = fopen(file_names(i),'r');
    A{i}   = fscanf(fileID,formatSpec1,[3 Inf])';
    fclose(fileID);
    fileID = fopen(file_names2(i),'r');
    solveP{i}   = fscanf(fileID,formatSpec2,[5 Inf]);
    fclose(fileID);
    count = 1;
    % Get the evolution of the mean Nusselt when converged
    for j=1:length(A{i})
       if A{i}(j,1) >= time_converged
            Nu_converged{i}(count) = 0.5*(A{i}(j,2)+A{i}(j,3));
            count = count + 1;
       end
    end
end


figure
hold on
grid on
p = xline(time_converged,'k:',{'Convergence','Time'});
% von Hardenberg, J., Parodi, A., Passoni, G., Provenzale, A., and Spiegel, 
% E. A., "Large-scale patterns in rayleigh–bénard convection,” 
% Physics letters. A, vol. 372, no. 13, p. 2223–2229, 2008.
q = yline(16.36,'k:',{'Hardenberg et al.','(2007)'});
fprintf("Nu reference = %f\n",16.36)
count = 1;
for i=1:length(file_names)
   h(count) = plot(A{i}(:,1),A{i}(:,2),'--','color',colors{i}(1,:));
   h(count+1) = plot(A{i}(:,1),A{i}(:,3),'--','color',colors{i}(2,:));
   h(count+2) = plot(A{i}(:,1),(0.5*(A{i}(:,2)+A{i}(:,3))),'-','color',colors{i}(3,:));
   leg{count} = ['Top ' sch{i}];
   leg{count+1} = ['Bottom ' sch{i}];
   leg{count+2}= ['Mean ' sch{i}];
   count = count +3;
   fprintf(['Nu SA3 ' sch{i} ' = %f\n'],mean(Nu_converged{i}));
   fprintf(['Poisson eqn solutons ' sch{i} ' = %d\n'],solveP{i}(1))
end
legend(h,leg,'Interpreter','latex');
xlabel("Time [s]",'Interpreter','latex');
ylabel("Nu",'Interpreter','latex');
ylim([0 100])
