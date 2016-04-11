%Kalman filter
clear all;
close all;
%load data
noisyPendulum = importdata('noisy_pendulum.csv');
truePendulum = importdata('true_pendulum.csv');

%define main variables
dt = 0.5;      %sampling rate
S_frame = 1; %starting frame
u = .001;    % define acceleration magnitude
Q= [noisyPendulum(S_frame,1); noisyPendulum(S_frame,2); 0; 0];  %initialized state with four components : positionX, positionY, velocityX, velocityY
Q_estimate = Q;  %estimation for the initial location of pendulum 
PendulumAccelNoiseMag = 31;   %process noise
vtx = 31;    %measurement noise on the x axis
vty = 91;    %measurement noise on the y axis
Ez = [vtx 0; 0 vty];
Ex = [dt^4/4 0 dt^3/2 0; ...
    0 dt^4/4 0 dt^3/2; ...
    dt^3/2 0 dt^2 0; ...
    0 dt^3/2 0 dt^2].*PendulumAccelNoiseMag^2; % Convert the process noise into covariance matrix
P = Ex;     %estimate of initial pendulum position (covariance matrix)

%define the update equations
A = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];  %state update matrix
B = [(dt^2/2); (dt^2/2); dt; dt];
C = [1 0 0 0; 0 1 0 0];      % measurement function C that is applied to the state estimate Qt

%define arrays that will contain the results
Q_loc = []; % actual pendulum path
%vel = []; % actual pendulum velocity
Q_loc_meas = []; % the pendulum path obtained by the algorithm

% initize estimation variables
Q_loc_estimate = []; %  position estimate
vel_estimate = []; % velocity estimate
P_estimate = P;
predic_state = [];
predic_var = [];

for t = S_frame:size(noisyPendulum,1)
   
    % load the given tracking
    Q_loc_meas(:,t) = [ noisyPendulum(t,1); noisyPendulum(t,2)];
    
    %Prediction
    % predict next state using dynamical model
    Q_estimate = A * Q_estimate + B * u;
    predic_state = [predic_state; Q_estimate(1)] ;
    %predict error covariance
    P = A * P * A' + Ex;
    predic_var = [predic_var; P] ;
    
    %Correction
    %Kalman gain
    K = P*C'*inv(C*P*C'+Ez);
    %update state with observation
    Q_estimate = Q_estimate + K * (Q_loc_meas(:,t) - C * Q_estimate);
    %update state covariance
    P =  (eye(4)-K*C)*P;
    
    % Store data
    Q_loc_estimate = [Q_loc_estimate; (Q_estimate(1:2))'];
    vel_estimate = [vel_estimate; Q_estimate(3:4)'];
    
    
end

%calculation of RMS for noisy and estimated values
errorn = my_rms(noisyPendulum(300:400,:),truePendulum(300:400,:));
disp (errorn);

errore = my_rms(Q_loc_estimate(300:400,:),truePendulum(300:400,:));
disp (errore);

%plot evolution of x coordinate: noisy, true and estimated pendulum
plot(300:400, noisyPendulum(300:400,1),'r-');
hold on;
plot(300:400, truePendulum(300:400,1),'b-');
hold off;
hold on;
plot(300:400, Q_loc_estimate(300:400,1),'g-');
hold off;
title('plot of x coordinate');

figure;
%plot evolution of y coordinate: noisy, true and estimated pendulum
plot(300:400, noisyPendulum(300:400,2),'r-');
hold on;
plot(300:400, truePendulum(300:400,2),'b-');
hold off;
hold on;
plot(300:400, Q_loc_estimate(300:400,2),'g-');
hold off;
title('plot of y coordinate');
%%
%Kalman filter grid search for the best error values
clear all;
close all;
%load data
noisyPendulum = importdata('noisy_pendulum.csv');
truePendulum = importdata('true_pendulum.csv');
RMSstorage = [];

for vtx = 1:10:100
   disp(vtx);
    
    for vty = 1:10:100
        for PendulumAccelNoiseMag = 1:10:100
        for dt = 0.001:0.5:1    
        for u = 0.0005:0.01:0.05  
    
        
%define main variables
dt = 0.5;      %sampling rate
S_frame = 1; %starting frame
u = .001;    % define acceleration magnitude
Q= [noisyPendulum(S_frame,1); noisyPendulum(S_frame,2); 0; 0];  %initialized state with four components : positionX, positionY, velocityX, velocityY
Q_estimate = Q;  %estimation for the initial location of pendulum 
%PendulumAccelNoiseMag = 100;   %process noise
%vtx = 10;    %measurement noise on the x axis
%vty = 50;    %measurement noise on the y axis
Ez = [vtx 0; 0 vty];
Ex = [dt^4/4 0 dt^3/2 0; ...
    0 dt^4/4 0 dt^3/2; ...
    dt^3/2 0 dt^2 0; ...
    0 dt^3/2 0 dt^2].*PendulumAccelNoiseMag^2; % Convert the process noise into covariance matrix
P = Ex;     %estimate of initial pendulum position (covariance matrix)

%define the update equations
A = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];  %state update matrix
B = [(dt^2/2); (dt^2/2); dt; dt];
C = [1 0 0 0; 0 1 0 0];      % measurement function C that is applied to the state estimate Qt


%define arrays that will contain the results
Q_loc = []; % actual pendulum path
%vel = []; % actual pendulum velocity
Q_loc_meas = []; % the pendulum path obtained by the algorithm


        
        % initize estimation variables
        Q_loc_estimate = []; %  position estimate
        vel_estimate = []; % velocity estimate
        P_estimate = P;
        predic_state = [];
        predic_var = [];

for t = S_frame:size(noisyPendulum,1)
   
    % load the given tracking
    Q_loc_meas(:,t) = [ noisyPendulum(t,1); noisyPendulum(t,2)];
    
    %Prediction
    % predict next state using dynamical model
    Q_estimate = A * Q_estimate + B * u;
    predic_state = [predic_state; Q_estimate(1)] ;
    %predict error covariance
    P = A * P * A' + Ex;
    predic_var = [predic_var; P] ;
    
    %Correction
    %Kalman gain
    K = P*C'*inv(C*P*C'+Ez);
    %update state with observation
    Q_estimate = Q_estimate + K * (Q_loc_meas(:,t) - C * Q_estimate);
    %update state covariance
    P =  (eye(4)-K*C)*P;
    
   
    %Store data
    Q_loc_estimate = [Q_loc_estimate; Q_estimate(1:2)'];
    vel_estimate = [vel_estimate; Q_estimate(3:4)'];
    
 end

    errore = my_rms(Q_loc_estimate,truePendulum);
    RMSstorage = [RMSstorage; vtx vty PendulumAccelNoiseMag dt u   errore(1) errore(2)];

        end
    end
        end
    end
end

sumError = RMSstorage(:,6) + RMSstorage(:,7);
disp('min error for estimation:');
disp(RMSstorage(find(sumError==min(sumError)),:));

errorn = my_rms(noisyPendulum,truePendulum);
disp('error for noisy pendulum:');
disp(errorn);


%%
%Particle filter by using nonlinear sinusoidal model
clear all;
%load data
noisyPendulum = importdata('noisy_pendulum.csv');
truePendulum = importdata('true_pendulum.csv');

x = noisyPendulum(1,:);   %initial state
x_Nx = 1; % Noise covariance in the system for x coordinate
x_Ny = 5; % Noise covariance in the system for y coordinate
x_R = 1;  % Noise covariance in the measurement
N = 500; % The number of particles the system generates at each iteration

V = 1; %variance of the initial esimate
x_P = zeros(100,2); %the vector of particles

%parameters for sinusoid
m = mean(noisyPendulum);
freqy = 1/15;
freqx = 1/31;
ampy = (2.994-2.638)/2;
ampx = (1.415+1.438)/2;

% make the randomly generated particles from the initial prior gaussian distribution
for i = 1:N
    x_P(i,:) = x + sqrt(V) * randn(1,2);
end

x_est_out = []; % the vector of particle filter estimates.

x_P_update = zeros(size(noisyPendulum));
P_w = zeros(size(noisyPendulum));

for t = 1:size(noisyPendulum,1)
   %importance sampling step
   for i = 1:N
      %update model to make a new set of transitioned particles.
      x_P_update(i,:) = m + [ampx ampy].*sin([freqx freqy]*t+x_P(i)) + [x_Nx x_Ny];
       
      %Generate the weights for each of these particles
      P_w(i,:) = (1/sqrt(2*pi*x_R)) * exp(-(noisyPendulum(t,:) - x_P_update(i)).^2/(2*x_R));
   end
   
   % Normalize to form a probability distribution
   P_w = P_w./repmat(sum(P_w),size(P_w,1),1);
   
   % Resampling: From this new distribution, now we randomly sample from it to generate our new estimate particles
   for i = 1 : N
        x_P(i) = x_P_update(find(rand <= cumsum(P_w),1));
   end 
   
   %The final estimate is some metric of these final resampling, such as
   %the mean value or variance
   x_est = mean(x_P);
   
   x_est_out = [x_est_out; x_est];
    
end

%calculation of RMS for noisy and estimated values
errorn = my_rms(noisyPendulum,truePendulum);
disp (errorn);

errore = my_rms(x_est_out,truePendulum);
disp (errore);



%plot evolution of x coordinate: noisy, true and estimated pendulum
plot(300:400, noisyPendulum(300:400,1),'r-');
hold on;
plot(300:400, truePendulum(300:400,1),'b-');
hold off;
hold on;
plot(300:400, x_est_out(300:400,1),'g-');
hold off;
title('plot of x coordinate');

figure;
%plot evolution of y coordinate: noisy, true and estimated pendulum
plot(300:400, noisyPendulum(300:400,2),'r-');
hold on;
plot(300:400, truePendulum(300:400,2),'b-');
hold off;
hold on;
plot(300:400, x_est_out(300:400,2),'g-');
hold off;
title('plot of y coordinate');


%%
%Particle filter by using same linear model as for Kalman filter
clear all;
%load data
noisyPendulum = importdata('noisy_pendulum.csv');
truePendulum = importdata('true_pendulum.csv');


%define main variables
dt = 0.5;      %sampling rate
u = .001;    % define acceleration magnitude
x= [noisyPendulum(1,1)  noisyPendulum(1,2)  0  0];  %initialized state with four components : positionX, positionY, velocityX, velocityY
PendulumAccelNoiseMag = 1;   %process noise
vtx = 1;    %measurement noise on the x axis
vty = 0.5;    %measurement noise on the y axis
Ez = [vtx ; vty];
Ex = [dt^2/2 ;  dt^2/2; dt; dt ].*PendulumAccelNoiseMag^2; % Convert the process noise into covariance matrix
P = Ex;     %estimate of initial pendulum position (covariance matrix)

%define the update equations
A = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];  %state update matrix
B = [(dt^2/2); (dt^2/2); dt; dt];
C = [1 0 0 0; 0 1 0 0];      % measurement function C that is applied to the state estimate Qt

%Number of particles
N = 500;

V = 2; %variance of the initial esimate
x_P = zeros(100,4); %the vector of particles

% make the randomly generated particles from the initial prior gaussian distribution
for i = 1:N
    x_P(i,:) = x + sqrt(V) * randn(1,4);
end

x_est_out = []; % the vector of particle filter estimates.

x_P_update = zeros(size(noisyPendulum,1),4);
z_P_update = zeros(size(noisyPendulum));
P_w = zeros(size(noisyPendulum));

for t = 1:size(noisyPendulum,1)
   %importance sampling step
   for i = 1:N
        %update model to make a new set of transitioned particles.
        x_P_update(i,:) = (A*x_P(i,:)' + B*u + Ex)';
        z_P_update(i,:) = (C*x_P_update(i,:)' + Ez)';
        %Generate the weights for each of these particles
        P_w(i,:) = (1./sqrt(2*pi*Ez')) .* exp(-(noisyPendulum(t,:) - z_P_update(i)).^2./(2*Ez'));
   end
   
   % Normalize to form a probability distribution
   P_w = P_w./repmat(sum(P_w),size(P_w,1),1);
   
   % Resampling: From this new distribution, now we randomly sample from it to generate our new estimate particles
   for i = 1 : N
        x_P(i) = x_P_update(find(rand <= cumsum(P_w),1));
   end 
   
   %The final estimate is some metric of these final resampling, such as
   %the mean value or variance
   x_est = mean(x_P);
   
   x_est_out = [x_est_out; x_est];
    
end



%plot evolution of x coordinate: noisy, true and estimated pendulum
plot(300:400, noisyPendulum(300:400,1),'r-');
hold on;
plot(300:400, truePendulum(300:400,1),'b-');
hold off;
hold on;
plot(300:400, x_est_out(300:400,1),'g-');
hold off;
title('plot of x coordinate');

figure;
%plot evolution of y coordinate: noisy, true and estimated pendulum
plot(300:400, noisyPendulum(300:400,2),'r-');
hold on;
plot(300:400, truePendulum(300:400,2),'b-');
hold off;
hold on;
plot(300:400, x_est_out(300:400,2),'g-');
hold off;
title('plot of y coordinate');







