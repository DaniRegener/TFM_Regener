%% Compute the ROS of a ERK scheme and approximate it by a polynomial
% Author:     Daniel Regener Roig
% Supervisor: Arnau Miró Jané
% Date:       17/08/2023
% Program developed for the master's thesis "TFM-220MUAERON- 
% Advanced methods for numerical simulations of turbulent flows"
% ESEIAAT - UPC

% Description:
% This program plots the region of stability of a given ERK or LSERK
% scheme, plots it as function of phi, does 9th degree polynomial
% approximation to it and prints the coefficients from the higher to the
% lower degree, e.g. a_1*phi^n + a_2*phi^{n-1] + ...  + a_{n}

% RK scheme in study, options
% Classical ERK
%   RK4, 4p7q, 3p5q, kutta_RK3, heun_rk3, ralston_RK3, midpoint_RK2, 
%   SSP_RK3,SSP_RK4
% LSERK
%   RK4s5, RK4s6, RK4s7, RK4s12, RK4s14, LDDRK2s5, LDDRK4s6, RK3LR, RK3LT

RK_Type = 'RK4';
isLSRK = false; % set to true if it's a LSERK scheme

[a,b,c] = my_RK(RK_Type);                 % get RK coefficients
if isLSRK, [a,b,c] = convert(a,b,c); end  % convert the tableau if LSERK

% compute the stability func coefficients 
gamma = compute_region_of_stability(a,b,c); 

% plot the ROS
subplot(1,2,1)
figure(1)
plot_region_of_stability(gamma)
grid minor

% create a meshgrid and evaluate gamma
x = linspace(-30,20,10000);
y = linspace(-20,20,10000);
[xx,yy] = meshgrid(x,y);
z = abs(polyval(gamma,xx+yy*1j));
z = z-1;

% get the limits of the ROS
[xf,yf] = get_frontier(z,xx,yy);
xf = xf(~isnan(xf));
yf = yf(~isnan(xf));

[theta,rho] = cart2pol(xf,yf); % convert the limits to polar
[theta,I] = sort(theta);       % get the order of the points in polar coord       
rho = rho(I);                  % order the points in polar coords
[p3, S, mu] = polyfit(theta,rho,9); % do the polynomial interpolation
[rhoint, err] = polyval(p3,theta,S,mu); % evaluate the resulting approx
Rsq1 = 1 - sum((rho - rhoint).^2)/sum((rho - mean(rho)).^2); % error

% Plot the stability function as func of phi and its approximation
subplot(1,2,2)
hold on
text(pi/6,1.0*max(rho),strcat('R^2=',num2str(Rsq1)))
plot(pi-theta,rho)
plot(pi-theta,rhoint,'--')
xticks([0 pi/6 pi/3 pi/2])
xticklabels({'0','\pi/6','\pi/3','\pi/2'})
grid minor
xlabel('\phi')
ylabel('T_{opt}')
legend('data','interpolation','Location','Northeast','interpreter','latex');

% Functionality to copy this coeffs to larger programs
st = to_py(p3)

function [a,b,c] = convert(A,B,C)
    % Niegemann, J., Diehl, R., and Busch, K., “Efficient low-storage 
    % runge–kutta schemes with optimized stability regions," Journal of 
    % computational physics, vol. 231, no. 2, p. 364–372, 2012.
    s = length(B);
    a = zeros(s);
    b = zeros(1,s);
    c = zeros(1,s);
    b(s) = B(s);
    
    for i = s-1:-1:1
        b(i) = A(i+1)*b(i+1) + B(i);
    end

    for i = s:-1:1
        c(i) = C(i);
        for j = s-1:-1:1
            if j >= i
                a(i,j) = 0;
            elseif i == j+1
                a(i,j) = B(j);
            else
                a(i,j) = A(j+1)*a(i,j+1) + B(j);
            end
        end
    end

end

function [xf,yf] = get_frontier(z,x,y)
    s = sign(z);
    xf = [];
    yf = [];
    for i = 2:size(z,1)-1
        for j = 2:size(z,2)-1
            if x(i,j)<=0 && y(i,j)>=0
                if s(i,j-1)*s(i,j) < 0
                    xc = interp1([z(i,j-1),z(i,j)],[x(i,j-1),x(i,j)],0);
                    xf = [xf,xc];
                    yf = [yf,y(i,j)];
                end
                if s(i,j+1)*s(i,j) < 0
                    xc = interp1([[z(i,j),z(i,j+1)],x(i,j),x(i,j+1)],0);
                    xf = [xf,xc];
                    yf = [yf,y(i,j)];
                end
                if s(i-1,j)*s(i,j) < 0
                    yc = interp1([z(i-1,j),z(i,j)],[y(i-1,j),y(i,j)],0);
                    xf = [xf,x(i,j)];
                    yf = [yf,yc];
                end
                if s(i+1,j)*s(i,j) < 0
                    yc = interp1([z(i,j),z(i+1,j)],[y(i,j),y(i+1,j)],0);
                    xf = [xf,x(i,j)];
                    yf = [yf,yc];
                end
            end
        end
    end
    [xf,I] = sort(xf);
    yf = yf(I);
end

function plot_region_of_stability(gamma)
    x = linspace(-6,6,100);
    y = linspace(-6,6,100);
    [xx,yy] = meshgrid(x,y);
    z = abs(polyval(gamma,xx+yy*1j));
    hold on
    contour(x,y,z,[1,1])
    plot(x,zeros(size(x)),'--k')
    plot(zeros(size(y)),y,'--k')
    xlim([-6,1])
    ylim([-6,6])
end

function gamma = compute_region_of_stability(a,b,c)
    % Niegemann, J., Diehl, R., and Busch, K., “Efficient low-storage 
    % runge–kutta schemes with optimized stability regions," Journal of 
    % computational physics, vol. 231, no. 2, p. 364–372, 2012.
    s = length(b);
    gamma = zeros(1,s+1);
    gamma(1) = 1;
    gamma(2) = sum(b);
    for i = 3:s+1
        aux = b;
        for j = 1:i-3
            aux = aux*a;
        end
        gamma(i) = dot(aux,c);
    end
    gamma = flip(gamma);
end

function [a,b,c] = my_RK(RK_Type)
    if strcmp(RK_Type,'RK4')
        % Kutta, W., Beitrag zur näherungsweisen Integration totaler 
        % Differentialgleichungen. Teubner, 1901.
        a = [0   0   0  0;...
             0.5 0   0  0;...
             0   0.5 0  0;...
             0   0   1  0];
        b = [1/6 1/3 1/3 1/6];
        c = [0   0.5 0.5 1  ];
    elseif strcmp(RK_Type,'4p7q')
        % Capuano, F., Coppola, G., Rández, L., and de Luca, L., "Explicit 
        % runge–kutta schemes for incompressible flow with improved 
        % energy-conservation properties," Journal of computational 
        % physics, vol. 328, p. 86–94, 2017
        a = [[ 0., 0., 0.,0.,0.,0.];...
            [0.23593377,0.,0.,0.,0.,0.];...
            [0.34750736,-0.13561935,0.,0.,0.,0.];...
            [-0.20592852,1.89179077,-0.89775024,0.,0.,0.];...
            [-0.09435493,1.75617141,-0.9670785,0.06932826,0.,0.];...
	        [ 0.14157883,-1.17039696,1.30579112,-2.20354137,2.92656838,0.]];
        b = [0.07078942,0.87808571,-0.44887512,-0.44887512,0.87808571,0.07078942];
        c = sum(a,2);
    elseif strcmp(RK_Type,'3p5q')
        % Capuano, F., Coppola, G., Rández, L., and de Luca, L., "Explicit 
        % runge–kutta schemes for incompressible flow with improved 
        % energy-conservation properties," Journal of computational 
        % physics, vol. 328, p. 86–94, 2017
        a = [[0.,0.,0.,0.];...
            [0.375,0.,0.,0.];...
            [0.91666667,-0.66666667,0.,0.];...
            [-0.08333333,1.83333333,-0.75,0.]];
        b = [0.111,0.889,-0.222,0.222];
        c = sum(a,2);
    elseif strcmp(RK_Type,'kutta_RK3')
        % Kutta, W., Beitrag zur näherungsweisen Integration totaler 
        % Differentialgleichungen. Teubner, 1901.
        a = [[0.,0.,0.];...
            [1./2.,0.,0.];...
            [-1.,2.,0.]];
        b = [1./6.,2./3.,1./6.];
        c = sum(a,2);
    elseif strcmp(RK_Type,'heun_RK3')
        % Suli, E. and Mayers, D. F., An introduction to numerical 
        % analysis. Cambridge, England: Cambridge University Press, 2012.
        a = [[0.,0.,0.];...
            [1./3.,0.,0.];...
            [0.,2./3.,0.]];
        b = [1./4.,0,3./4.];
        c = sum(a,2);
    elseif strcmp(RK_Type,'ralston_RK3')
        % Ehle, B. L., On Padé approximations to the exponential function 
        % and A-stable methods for the numerical solution of initial value 
        % problems. PhD thesis, 1969.
        a = [[0.,    0.,    0.];...
             [0.5,   0.,    0.];...
             [0., 3./4.,    0.]];
        b = [2./9.,1./3.,4./9.];    %???????
        c = sum(a,2);
    elseif strcmp(RK_Type,'midpoint_RK2')
        % Suli, E. and Mayers, D. F., An introduction to numerical 
        % analysis. Cambridge, England: Cambridge University Press, 2012
        a = [[0.,0.];[0.5,0.]];
        b = [0.,1.];
        c = sum(a,2);
    elseif strcmp(RK_Type,'SSP_RK3')ç
        % Suli, E. and Mayers, D. F., An introduction to numerical 
        % analysis. Cambridge, England: Cambridge University Press, 2012
        a = [[0.,0.,0.];...
            [1.,0.,0.];...
            [1./4.,1./4.,0.]];
        b = [1./6.,1./6.,2./3.];
        c = [0.,1.,0.5];
    elseif strcmp(RK_Type,'SSP_RK4')
        % Suli, E. and Mayers, D. F., An introduction to numerical 
        % analysis. Cambridge, England: Cambridge University Press, 2012
        a = [[0.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0];...
            [0.0,0.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0];...
            [0.0,0.0,0.0,1.0/6.0,1.0/6.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0];...
            [0.0,0.0,0.0,0.0,1.0/6.0 ,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0];...
            [0.0,0.0,0.0,0.0,0.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0,1.0/15.0];...
            [0.0,0.0,0.0,0.0,0.0,0.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0];...
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0/6.0,1.0/6.0,1.0/6.0];...
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0/6.0,1.0/6.0];...
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0/6.0];...
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]];
        b = [1.0/10.0,1.0/10.0,1.0/10.0,1.0/10.0,1.0/10.0,1.0/10.0,1.0/10.0,1.0/10.0,1.0/10.0,1.0/10.0];
        a  = a'; % hard solution to a typo
        c = sum(a,2);
    elseif strcmp(RK_Type,'RK4s5')
        % Allampalli, V., Hixon, R., Nallasamy, M., Sawyer, S.D., 2009.
        % High-accuracy large-step explicit Runge–Kutta (HALE-RK) schemes 
        % for computational aeroacoustics. Journal of Computational 
        % Physics 228, 3837–3850. 
        a = [0.0,-0.4178904745,-1.192151694643,-1.697784692471,-1.514183444257];
        b = [0.1496590219993,0.3792103129999,0.8229550293869,0.6994504559488,0.1530572479681];
        c = [0.0,0.1496590219993,0.3704009573644,0.6222557631345,0.9582821306748];
    elseif strcmp(RK_Type,'RK4s6')
        % Allampalli, V., Hixon, R., Nallasamy, M., Sawyer, S.D., 2009.
        % High-accuracy large-step explicit Runge–Kutta (HALE-RK) schemes 
        % for computational aeroacoustics. Journal of Computational 
        % Physics 228, 3837–3850.  
        a  = [0.000000000000,-0.691750960670,-1.727127405211,-0.694890150986,-1.039942756197,-1.531977447611];
        b = [0.122000000000,0.477263056358,0.381941220320,0.447757195744,0.498614246822,0.186648570846];
        c = [0.000000000000,0.122000000000,0.269115878630,0.447717183551,0.749979795490,0.898555413085];
    elseif strcmp(RK_Type,'RK4s7')
        % Allampalli, V., Hixon, R., Nallasamy, M., Sawyer, S.D., 2009.
        % High-accuracy large-step explicit Runge–Kutta (HALE-RK) schemes 
        % for computational aeroacoustics. Journal of Computational 
        % Physics 228, 3837–3850. 
        a = [0.0,-0.647900745934,-2.704760863204,-0.460080550118,-0.500581787785,-1.906532255913,-1.450000000000];
        b = [0.117322146869,0.503270262127,0.233663281658,0.283419634625,0.540367414023,0.371499414620,0.136670099385];
        c = [0.000000000000,0.117322146869,0.294523230758,0.305658622131,0.582864148403,0.858664273599,0.868664273599];
    elseif strcmp(RK_Type,'RK4s12')
        % Niegemann, J., Diehl, R., Busch, K., 2012. Efficient low-storage
        % Runge–Kutta schemes with optimized stability regions. 
        % Journal of Computational Physics 231, 364–372. 
        a = [0.0000000000000000,-0.0923311242368072,-0.9441056581158819,-4.3271273247576394,-2.1557771329026072,-0.9770727190189062,-0.7581835342571139,-1.7977525470825499,-2.6915667972700770,-4.6466798960268143,-0.1539613783825189,-0.5943293901830616];
        b = [0.0650008435125904,0.0161459902249842,0.5758627178358159,0.1649758848361671,0.3934619494248182,0.0443509641602719,0.2074504268408778,0.6914247433015102,0.3766646883450449,0.0757190350155483,0.2027862031054088,0.2167029365631842];
        c = [0.0000000000000000 ,0.0650008435125904 ,0.0796560563081853 ,0.1620416710085376 ,0.2248877362907778 ,0.2952293985641261 ,0.3318332506149405 ,0.4094724050198658 ,0.6356954475753369 ,0.6806551557645497 ,0.7143773712418350 ,0.9032588871651854];
    elseif strcmp(RK_Type,'RK4s13')
        % Niegemann, J., Diehl, R., Busch, K., 2012. Efficient low-storage
        % Runge–Kutta schemes with optimized stability regions. 
        % Journal of Computational Physics 231, 364–372. 
        a = [0.0000000000000000,-0.6160178650170565,-0.4449487060774118,-1.0952033345276178,-1.2256030785959187,-0.2740182222332805,-0.0411952089052647,-0.1797084899153560,-1.1771530652064288,-0.4078831463120878,-0.8295636426191777,-4.7895970584252288,-0.6606671432964504];
        b = [0.0271990297818803,0.1772488819905108,0.0378528418949694,0.6086431830142991,0.2154313974316100,0.2066152563885843,0.0415864076069797,0.0219891884310925,0.9893081222650993,0.0063199019859826,0.3749640721105318,1.6080235151003195,0.0961209123818189];
        c = [0.0000000000000000,0.0271990297818803,0.0952594339119365,0.1266450286591127,0.1825883045699772,0.3737511439063931,0.5301279418422206,0.5704177433952291,0.5885784947099155,0.6160769826246714,0.6223252334314046,0.6897593128753419,0.9126827615920843];
    elseif strcmp(RK_Type,'RK4s14')
        % Niegemann, J., Diehl, R., Busch, K., 2012. Efficient low-storage
        % Runge–Kutta schemes with optimized stability regions. 
        % Journal of Computational Physics 231, 364–372. 
        a = [0.0000000000000000,-0.7188012108672410,-0.7785331173421570,-0.0053282796654044,-0.8552979934029281,-3.9564138245774565,-1.5780575380587385,-2.0837094552574054,-0.7483334182761610,-0.7032861106563359,0.0013917096117681,-0.0932075369637460,-0.9514200470875948,-7.1151571693922548];
        b = [0.0367762454319673,0.3136296607553959,0.1531848691869027,0.0030097086818182,0.3326293790646110,0.2440251405350864,0.3718879239592277,0.6204126221582444,0.1524043173028741,0.0760894927419266,0.0077604214040978,0.0024647284755382,0.0780348340049386,5.5059777270269628];
        c = [0.0000000000000000 ,0.0367762454319673 ,0.1249685262725025 ,0.2446177702277698 ,0.2476149531070420 ,0.2969311120382472 ,0.3978149645802642 ,0.5270854589440328 ,0.6981269994175695 ,0.8190890835352128 ,0.8527059887098624 ,0.8604711817462826 ,0.8627060376969976 ,0.8734213127600976];
    elseif strcmp(RK_Type,'LDDRK2s5')
        % Stanescu, D., Habashi, W.G., 1998. 2N-Storage Low Dissipation 
        % and Dispersion Runge-Kutta Schemes for Computational Acoustics. 
        % Journal of Computational Physics 143, 674–681.
        a = [0.0,-0.6913065,-2.655155,-0.8147688,-0.6686587];
        b = [0.1,0.75,0.7,0.479313,0.310392];
        c = [0.0,0.1,0.3315201,0.4577796,0.8666528];
    elseif strcmp(RK_Type,'LDDRK4s6')
        % Stanescu, D., Habashi, W.G., 1998. 2N-Storage Low Dissipation 
        % and Dispersion Runge-Kutta Schemes for Computational Acoustics. 
        % Journal of Computational Physics 143, 674–681.
        a = [0.0,-0.4919575,-0.8946264,-1.5526678,-3.4077973,-1.0742640];
        b = [0.1453095,0.4653797,0.4675397,0.7795279,0.3574327,0.15];
        c = [0.0,0.1453095,0.3817422,0.6367813,0.7560744,0.9271047];
    elseif strcmp(RK_Type,'RK3LR')
        % Williamson, J. H., “Low-storage runge-kutta schemes,” 
        % Journal of computational physicss vol. 35, no. 1, p. 48–56, 1980.
        a = [0.0,-6.573094670655647,0.046582819666687];
        b = [1.13541,0.15232,0.96366];
        c = [0.0,1.13541,0.15232];
    elseif strcmp(RK_Type,'RK3LT')
        % Williamson, J. H., “Low-storage runge-kutta schemes,” 
        % Journal of computational physicss vol. 35, no. 1, p. 48–56, 1980.
        a = [0.0,-0.637676219230114,-1.306629764111419];
        b = [0.45736,0.9253,0.39383];
        c = [0.0,0.45736,0.9253];
    end
end

function str = to_py(a)
    format long
    str = [];
    str = strcat(str,'ROS:np.array([');
    for i=1:length(a)
        str = strcat(str,num2str(a(i),'%4.16f'));
        if i<length(a), str = strcat(str,','); end
    end
    str = strcat(str,'],np.double)');
end