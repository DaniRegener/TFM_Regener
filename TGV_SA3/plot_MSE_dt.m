%% Plot the TKE error as function of different parameters for fixed dt
% Author:     Daniel Regener Roig
% Supervisor: Arnau Miró Jané
% Date:       17/08/2023
% Program developed for the master's thesis "TFM-220MUAERON- 
% Advanced methods for numerical simulations of turbulent flows"
% ESEIAAT - UPC

% Description: This program uses the data generated by the successive
% simulations create by run_some_dt.sh and stored in the folders added to
% path. It will plot the error dependence on the constant timestep of the 
% cases


clear
% Add the path of 32^3 meshes
addpath("32_t/")
mesh = "32";
% Names of the cases to plot, check with the files in the added folder
sch  = {'AB','RK4','RK4a','SSP3'};
% Lines for the plots
markers = {'s-','o-','+-','^-'};
% Maximum dt of each scheme
dtmax.(sch{1}) = 0.18;
dtmax.(sch{2}) = 0.71;
dtmax.(sch{3}) = 0.49;
dtmax.(sch{4}) = 0.49;

% Reference SA3 1024^3 AB+CFL
tab_ref = readtable("energy_1024.csv");

for i = 1:length(sch) % for each scheme
    for j = 0.01:0.01:dtmax.(sch{i}) % for each dt
        % generate the name of the table
        myTab = strcat('energy_',mesh,'_',sch{i},'_dt',num2str(j*1000),...
                                                                   '.csv');
        tab = readtable(myTab);
        % interpolate the reference to the abcissas of the readed file
        ref_interpolated = interp1(tab_ref.Var1,tab_ref.Var2,tab.Var1,...
                                                        'linear','extrap');
        % Compute MSE
        MSE.(sch{i})(uint8(j*100)) = mean((tab.Var2-ref_interpolated).^2);
    end
end

figure
hold on
grid on 
for i = 1:length(sch)
    plot(0.01:0.01:dtmax.(sch{i}),MSE.(sch{i}),markers{i})     
end
legend(sch)
xlabel('\Delta t','Interpreter','latex')
ylabel("MSE in TKE",'Interpreter','latex')