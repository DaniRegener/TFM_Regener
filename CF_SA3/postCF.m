close all

% Which folders to be plotted
cases = {'LF','RK','RKa'};
leg = {"Hoyas \& Jim\'enez (2007)",'LF','RK4 classic','RK4 classic O2'};
mu = 5.55555556e-3;
rho = 1;


% Process the references
dns    = readtable('channel_for_dani/data/Re180_DNS.dat');    
dns_budgets{1} = readtable('CF/data/Re180_DNS_UU.dat');
dns_budgets{2} = readtable('CF/data/Re180_DNS_UV.dat');
dns_budgets{3} = readtable('CF/data/Re180_DNS_VV.dat');
dns_budgets{4} = readtable('CF/data/Re180_DNS_WW.dat');
color_ref = '--k';


for i = 1:length(cases)
    averages{i} = readtable(strcat('CF/',cases{i},...
        '/averages.csv'));
    budgets{i} = readtable(strcat('CF/',cases{i},...
        '/budgets.csv'));
end

for i = 1:length(cases)
    processed_averages{i} = process_averages(averages{i});
    processed_budgets{i}  = process_budgets(budgets{i});
    [tw(i),Re_tau(i),utau(i)] = obtain_adim(rho,mu, ...
                                           processed_averages{i}.u,...
                                           processed_averages{i}.y);
    r{i} = bl(processed_averages{i},processed_budgets{i},mu,utau(i));
end



figure
subplot(2,2,1)
hold on
grid on
plot(dns.Var2,dns.Var3,color_ref)
for i = 1:length(cases)
    plot(r{i}.ystar,r{i}.ustar)
end
xlabel('$y^+$',Interpreter='latex')
ylabel('$u^+$',Interpreter='latex')
set(gca,'xscale','log')

subplot(2,2,2)
hold on
grid on
plot(dns.Var2,dns.Var4,color_ref)
for i = 1:length(cases)
    plot(r{i}.ystar,r{i}.urstar)
end
xlabel('$y^+$',Interpreter='latex')
ylabel('$U_{RMS}$',Interpreter='latex')

subplot(2,2,3)
hold on
grid on
plot(dns.Var2,dns.Var5,color_ref)
for i = 1:length(cases)
    plot(r{i}.ystar,r{i}.vrstar)
end
xlabel('$y^+$',Interpreter='latex')
ylabel('$V_{RMS}$',Interpreter='latex')

subplot(2,2,4)
hold on
grid on
plot(dns.Var2,dns.Var6,color_ref)
for i = 1:length(cases)
    plot(r{i}.ystar,r{i}.wrstar)
end
xlabel('$y^+$',Interpreter='latex')
ylabel('$W_{RMS}$',Interpreter='latex')
legend(leg,'Interpreter','latex','Location','southeast')

figure
subplot(1,3,1)
hold on
grid on
plot(dns.Var2,dns.Var11,color_ref)
for i = 1:length(cases)
    plot(r{i}.ystar,r{i}.uvstar)
end
xlabel('$y^+$',Interpreter='latex')
ylabel('$R_{12}$',Interpreter='latex')

subplot(1,3,2)
hold on
grid on
plot(dns.Var2,dns.Var12,color_ref)
for i = 1:length(cases)
    plot(r{i}.ystar,r{i}.uwstar)
end
xlabel('$y^+$',Interpreter='latex')
ylabel('$R_{13}$',Interpreter='latex')

subplot(1,3,3)
hold on
grid on
plot(dns.Var2,dns.Var13,color_ref)
for i = 1:length(cases)
    plot(r{i}.ystar,r{i}.vwstar)
end
xlabel('$y^+$',Interpreter='latex')
ylabel('$R_{23}$',Interpreter='latex')
legend(leg,'Interpreter','latex','Location','southeast')

% toLabels = {'11','12','22','33'};
toLabels = {'11','13','33','22'};
toLegend = {'11','12','22','33'};
for j=1:length(dns_budgets)
    figure 
    subplot(2,3,1)
    hold on
    grid on
    plot(dns_budgets{j}.Var2,dns_budgets{j}.Var3,color_ref)
    for i = 1:length(cases)
        plot(r{i}.ystar,r{i}.(strcat('e',toLabels{j})))
    end
    xlabel('$y^+$',Interpreter='latex')
    ylabel(strcat('$\varepsilon_{',toLegend{j},'}$'),Interpreter='latex')
    
    subplot(2,3,2)
    hold on
    grid on
    plot(dns_budgets{j}.Var2,dns_budgets{j}.Var4,color_ref)
    for i = 1:length(cases)
        plot(r{i}.ystar,r{i}.(strcat('p',toLabels{j})))
    end
    xlabel('$y^+$',Interpreter='latex')
    ylabel(strcat('$P{',toLegend{j},'}$'),Interpreter='latex')
    
    
    subplot(2,3,3)
    hold on
    grid on
    plot(dns_budgets{j}.Var2,dns_budgets{j}.Var5,color_ref)
    for i = 1:length(cases)
        plot(r{i}.ystar,r{i}.(strcat('pstr',toLabels{j})))
    end
    xlabel('$y^+$',Interpreter='latex')
    ylabel(strcat('$\Phi_{',toLegend{j},'}$'),Interpreter='latex')
    
    subplot(2,3,4)
    hold on
    grid on
    plot(dns_budgets{j}.Var2,dns_budgets{j}.Var7,color_ref)
    for i = 1:length(cases)
        plot(r{i}.ystar,r{i}.(strcat('tdif',toLabels{j})))
    end
    xlabel('$y^+$',Interpreter='latex')
    ylabel(strcat('$D^1_{',toLegend{j},'}$'),Interpreter='latex')
    
    subplot(2,3,5)
    hold on
    grid on
    plot(dns_budgets{j}.Var2,dns_budgets{j}.Var6,color_ref)
    for i = 1:length(cases)
        plot(r{i}.ystar,r{i}.(strcat('pdif',toLabels{j})))
    end
    xlabel('$y^+$',Interpreter='latex')
    ylabel(strcat('$D^2_{',toLegend{j},'}$'),Interpreter='latex')
    
    subplot(2,3,6)
    hold on
    grid on
    plot(dns_budgets{j}.Var2,dns_budgets{j}.Var8,color_ref)
    for i = 1:length(cases)
        plot(r{i}.ystar,r{i}.(strcat('vdif',toLabels{j})))
    end
    xlabel('$y^+$',Interpreter='latex')
    ylabel(strcat('$D^3_{',toLegend{j},'}$'),Interpreter='latex')
    legend(leg,'interpreter','latex')
end


function r = bl(pa,pb,mu,utau)
    r.ystar = pa.y*utau/mu;
    r.ustar  = pa.u/utau;
    r.uustar = sqrt(pa.Rxx)/utau;
    r.vvstar = sqrt(pa.Rzz)/utau;
    r.wwstar = sqrt(pa.Ryy)/utau;
    r.uvstar = pa.Rxz./utau^2;
    r.uwstar = pa.Rxy./utau^2;
    r.vwstar = pa.Ryz./utau^2;
    r.urstar = pa.u_rms/utau; 
    r.vrstar = pa.w_rms/utau; 
    r.wrstar = pa.v_rms/utau;   

    % Gradient
    r.g11 = pa.g11;
    r.g22 = pa.g22;
    r.g33 = pa.g33;
    r.g12 = pa.g12;
    r.g13 = pa.g13;
    r.g23 = pa.g23;

    from_processed = 'ecpsdtvb';
    to_result = {'e','c','p','pstr','pdif','tdif','vdif','resi'};
    for k=1:length(from_processed)
        for i = 1:3
            for j=i:3
                r.(strcat(to_result{k},num2str(i),...
                    num2str(j))) = ...
                pb.(strcat(from_processed(k),num2str(i),num2str(j)))...
                    *(mu/utau^4);    
            end
        end
    end
   
end

function processed_budgets  = process_budgets(budgets)
    to_result = 'cptdvseb';
    from_budgets = {'Conve','Produ','Tdiff','Pdiff','Vdiff','Pstra',...
        'Dissi','B'};

    for k=1:length(to_result)
        for i = 1:3
            for j=i:3
                processed_budgets.(strcat(to_result(k),num2str(i),...
                    num2str(j))) = ...
                budgets.(strcat(from_budgets{k},num2str(i),num2str(j)))(1:0.5*end);    
            end
        end
    end
    processed_budgets.e11 = -processed_budgets.e11;
    processed_budgets.e13 = -processed_budgets.e13;    
    processed_budgets.e33 = -processed_budgets.e33;
    processed_budgets.e12 = -processed_budgets.e12;
    processed_budgets.e23 = -processed_budgets.e23;
    processed_budgets.e22 = -processed_budgets.e22;
end

function processed_averages = process_averages(averages)
    
    midlineAvg = @(x) 0.5*(x(1:0.5*end)+flip(x(0.5*end+1:end)));

    processed_averages.y = averages.z(1:0.5*end);
    processed_averages.u = midlineAvg(averages.u);
    processed_averages.v = midlineAvg(averages.v);
    processed_averages.w = midlineAvg(averages.w);
    processed_averages.Rxx = midlineAvg(averages.Rxx);
    processed_averages.Rxy = averages.Rxy(1:0.5*end);
    processed_averages.Rxz = averages.Rxz(1:0.5*end);
    processed_averages.Ryy = midlineAvg(averages.Ryy);
    processed_averages.Ryz = averages.Ryz(1:0.5*end);
    processed_averages.Rzz = midlineAvg(averages.Rzz);
    processed_averages.u_rms  = midlineAvg(averages.u_rms);
    processed_averages.v_rms  = midlineAvg(averages.v_rms);
    processed_averages.w_rms  = midlineAvg(averages.w_rms);
    processed_averages.g11  = midlineAvg(averages.g11);
    processed_averages.g12  = averages.g12(1:0.5*end);
    processed_averages.g13  = averages.g13(1:0.5*end);
    processed_averages.g22  = midlineAvg(averages.g22);
    processed_averages.g23  = averages.g23(1:0.5*end);
    processed_averages.g33  = midlineAvg(averages.g33);
end

function [tw,Re_tau,utau] = obtain_adim(rho,mu,u,z)
    tw = abs(mu*(u(2)-u(1))/(z(2)-z(1)));
    Re_tau = sqrt(tw/rho);
    utau   = sqrt(tw/rho);
end
  




