%% Present a table with the perfromance study on the RB case
% Author:     Daniel Regener Roig
% Supervisor: Arnau Miró Jané
% Date:       17/08/2023
% Program developed for the master's thesis "TFM-220MUAERON- 
% Advanced methods for numerical simulations of turbulent flows"
% ESEIAAT - UPC

% Description:
% This program returns a table with useful information for the analysis of
% the performance of SA3 in resolving the RB case. Maybe you want to launch
% the following cmds (if linux) to generate the files that are called here
% grep "cr_info name ins_solveP00" (somecase)/stdout00.txt > 
%                                               (somecase)/ins_solveP.txt
% grep "cr_info name solveStep00" (somecase)/stdout00.txt > 
%                                               (somecase)/ins_solveSte.txt

% For legend
sch = ["AB","AB(LES)","LF","LF(LES)","RK4","RK4(LES)","RK4a","RK4a(LES)",...
    "SSP","SSP(LES)"];
% f admitted for each scheme, in the order of sch
fmax = [0.4,0.4,1.0,1.0,1.2,1.0,1.2,1.0,0.6,0.6]';

% files containing the information of the use of the solver function
file_names1 = ["RB/RB_AB/ins_solveP.txt",...
    "RB/RB_LES3_AB/ins_solveP.txt",...
    "RB/RB_LF/ins_solveP.txt",...
    "RB/RB_LES3_LF/ins_solveP.txt",...
    "RB/RB_RK4/ins_solveP.txt",...
    "RB/RB_LES3_RK4/ins_solveP.txt",...
    "RB/RB_RK4a/ins_solveP.txt",...
    "RB/RB_LES3_RK4a/ins_solveP.txt",...
    "RB/RB_SSP/ins_solveP.txt",...
    "RB/RB_LES3_SSP/ins_solveP.txt",...
    ];
% files containing the information of the use of the solveStep
file_names2 = ["RB/RB_AB/ins_solveSte.txt",...
    "RB/RB_LES3_AB/ins_solveSte.txt",...
    "RB/RB_LF/ins_solveSte.txt",...
    "RB/RB_LES3_LF/ins_solveSte.txt",...
    "RB/RB_RK4/ins_solveSte.txt",...
    "RB/RB_LES3_RK4/ins_solveSte.txt",...
    "RB/RB_RK4a/ins_solveSte.txt",...
    "RB/RB_LES3_RK4a/ins_solveSte.txt",...
    "RB/RB_SSP/ins_solveSte.txt",...
    "RB/RB_LES3_SSP/ins_solveSte.txt",...
    ];

% formats in each file
formatSpec1 = strcat("cr_info name ins_solveP00         n     %d ",...
                "tmin %e tmax %e tavg %e tsum %e\n");
formatSpec2 = strcat("cr_info name solveStep00          n     %d ",...
                "tmin %e tmax %e tavg %e tsum %e\n");


for i = 1:length(file_names1)
    % Read solveP file
    fileID = fopen(file_names1(i),'r');
    solveP{i}   = fscanf(fileID,formatSpec1,[5 Inf]);
    fclose(fileID);
    % Read solveStep file
    fileID = fopen(file_names2(i),'r');
    solveStep{i}   = fscanf(fileID,formatSpec2,[5 Inf]);
    fclose(fileID);
    % Get the useful information of the data extracted
    Iterations(i,1) = solveStep{i}(1);
    Calls2solver(i,1)     = solveP{i}(1);
    Tot_time(i,1)  = solveStep{i}(5);
    SolverWeight(i,1)     = solveP{i}(5)/solveStep{i}(5);
end

% Arrange the data into a table
tab = table(sch',fmax,Iterations,Calls2solver,SolverWeight,Tot_time);

% present the result
disp(tab)