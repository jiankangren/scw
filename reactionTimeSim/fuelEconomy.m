clear all
% sampling time of the leader
leaderT = 0.001;
% sampling time of the follower
%followerT = linspace(0.01,0.1,10);
followerT = linspace(1,100)*0.001;
% follower external sensor noises
lidarVar  = 0.0009; % 3 cm
% 0.2 mph -> 11 mph
radarVar  = linspace(0.0447, 4.47,10);
%radarVar = linspace(1,100)*0.25;
% braking threshold
%brakingAcc = linspace(5,15,10);
brakingAcc = ones(1,100)*9.8;
% braking threshold percentiles
%%% 0.9   -> Z = 1.281552
%%% 0.99  -> Z = 2.326348
%%% 0.999 -> Z = 3.090232
threshold = [1.281552; 2.326348;3.090232];

% follower inertial sensor noises
gpsVar = 1; % 1 meter
encoderVar = 0.01;
%input noise
varInput = 0.01;
varLeaderInput = 2.76; % 2.67^2
leaderDesiredVel = 26;
followingDistance = 100;
%set one goal at 1000 m
goal = nhrSetGoals(1);
falsePositive = zeros(10,10,3);
for i = 1:10 % independent variable
    for j = 1:10 % number of samples 
        for k = 1:3 % number of percentiles
            emergencyBrakesOn = 0;
            follower = nhrInit(10,2.5);
            leader = nhrInit(200,2.5);
            leaderEstimate = nhrInit(100,2.5);
            xRealFollower= [];
            xRealLeader = [];
            velIntegral = 0;
            prevVelError = 0;
            cruiseIntegral = 0;
            prevCruiseError = 0;
            leaderCruiseIntegral = 0;
            prevLeaderCruiseError = 0;
            filterConvergence = 0;
            u = [0;0];
            uLeader(:,1) = [0;0];
            xRealFollower(:,1) = [10;2.5;0;0];
            xRealLeader(:,1) = [200;2.5;0;0];
            x(:,1) = xRealFollower;
            xLeader(:,1) = [200;2.5;0;0;0;0];
            F = [1  0  followerT(i) 0 ;
                 0  1  0  followerT(i);
                 0  0  1  0 ;
                 0  0  0  1];
            leaderF = [1  0  leaderT 0  (leaderT^2)/2 0;
                       0  1  0  leaderT 0        (leaderT^2)/2;
                       0  0  1  0  leaderT       0;
                       0  0  0  1  0        leaderT;
                       0  0  0  0  1        0
                       0  0  0  0  0        1];
            leaderFE = [1  0  followerT(i) 0  (followerT(i)^2)/2 0;
                       0  1  0  followerT(i) 0        (followerT(i)^2)/2;
                       0  0  1  0  followerT(i)       0;
                       0  0  0  1  0        followerT(i);
                       0  0  0  0  1        0
                       0  0  0  0  0        1];
            B = [0.5*(followerT(i)^2) 0 ;  
                0           0.5*(followerT(i)^2);
                followerT(i)          0 ;
                0           followerT(i)];
            leaderB = [0.5*(leaderT^2) 0 ;  
                0           0.5*(leaderT^2);
                leaderT          0 ;
                0           leaderT];
            P = [10 0  0  0 ;
                 0  10  0  0 ;
                 0  0  1  0 ;
                 0  0  0  1];
            leaderP = [1 0  0  0 0 0;
                       0  1 0  0 0 0;
                       0  0  10  0 0 0;
                       0  0  0  10 0 0;
                       0  0  0  0 10 0;
                       0  0  0  0 0 10];
            prevLeaderP = leaderP;
            H = [1  0  0  0 ;
                 0  1  0  0 ;
                 1  0  0  0 ;
                 0  1  0  0];
            leaderH =  [1  0  0  0 0 0;
                        0  1  0  0 0 0;
                        1  0  0  0 0 0;
                        0  1  0  0 0 0];
            Q = [0  0  0         0 ;
                 0  0  0         0 ;
                 0  0  varInput  0 ;
                 0  0  0         varInput];

            leaderQ = [0  0  0  0  0 0;
                       0  0  0  0  0 0;
                       0  0  0  0  0 0;
                       0  0  0  0  0 0;
                       0  0  0  0  varLeaderInput 0;
                       0  0  0  0   0 0];

            R = [gpsVar  0  0     0 ;
                 0  gpsVar  0     0 ;
                 0  0  encoderVar     0 ;
                 0  0     0   encoderVar];

            leaderR =[lidarVar  0 0   0     ;
                      0  lidarVar 0   0     ;
                      0  0  radarVar(i)  0     ;
                      0  0     0   radarVar(i) ];
            index = 2;
            frequencyCounter = int32(followerT(i)/leaderT);
            error(1) = sqrt((leader.x - follower.x)^2 +  (leader.y - follower.y)^2);
            while(true)
                if(index == 1000)
                    break;
                end
                %calculate the actual state based on previous input.
                 xRealFollower(:,index) = (F*xRealFollower(:,index-1) + leaderB*u);
                 xRealLeader(:,index) = (F*xRealLeader(:,index-1) + leaderB*uLeader(:,index-1)); 
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Leader Controller %
                
                % leader PID controller to do cruise control
                leaderCruiseError = leaderDesiredVel - leader.v;
                leaderCruiseIntegral = leaderCruiseIntegral + leaderCruiseError*leaderT;
                leaderCruiseDerivative = (leaderCruiseError - prevLeaderCruiseError)/leaderT;

                [leader, ux, uy] = nhrCruiseOneGoal(goal, leader, leaderCruiseError, leaderCruiseIntegral, leaderCruiseDerivative);
                % construct input vector with noise
                uLeader(:,index)= [ux ;
                     uy;
                ];

                prevLeaderCruiseError = leaderCruiseError;
                
                leader.x = xRealLeader(1,index);
                leader.y = xRealLeader(2,index);
                leader.v = sqrt(xRealLeader(3,index)^2 + xRealLeader(4,index)^2);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                frequencyCounter  = frequencyCounter - 1;
                if(frequencyCounter == 0)
                    frequencyCounter= int32(followerT(i)/leaderT);
                    % generate sensor mesaurements with noise for follower.
                    Z(:,index) = [xRealFollower(1,index) + normrnd(0,sqrt(gpsVar));
                      xRealFollower(2,index) + normrnd(0,sqrt(gpsVar));
                      xRealFollower(1,index) + normrnd(0,sqrt(encoderVar));
                      xRealFollower(2,index) + normrnd(0,sqrt(encoderVar))
                     ];
                    % generate sensor mesaurements with noise for leader. 
                    leaderZ(:,index) = [xRealLeader(1,index) + normrnd(0,sqrt(lidarVar));
                      xRealLeader(2,index) + normrnd(0,sqrt(lidarVar));
                      xRealLeader(1,index) + normrnd(0,sqrt(radarVar(i)));
                      xRealLeader(2,index) + normrnd(0,sqrt(radarVar(i)))
                     ];
                    % prediction for follower
                    P1 = F*P*F' + Q;
                    S  = H*P1*H' + R;
                    % prediction for leader
                    leaderP1 = leaderFE*leaderP*leaderFE' + leaderQ;
                    leaderS  = leaderH*leaderP1*leaderH' + leaderR;
                    % measurements update
                    %kalman gain for follower
                    K = P1*H'*inv(S);
                    %kalman gain for leader
                    leaderK = leaderP1*leaderH'*inv(leaderS);
                    % state covariance update for follower
                    P = P1 - K*H*P1;
                    % state covariance update for leader
                    leaderP = leaderP1 - leaderK*leaderH*leaderP1;
                    % state update for follower
                    x(:,index) = F*x(:,index-1) + B*u +  K*(Z(:,index)-H*(F*x(:,index-1)+B*u)); 
                    % state update for leader
                    xLeader(:,index) = leaderF*xLeader(:,index-1)  +  leaderK*(leaderZ(:,index)-leaderH*(leaderF*xLeader(:,index-1))); 
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Follower's braking condition
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    if(emergencyBrakesOn ==1)
                        if( (xLeader(5,index) - sqrt(leaderP(5,5)) * threshold(k)) <= -brakingAcc(i))
                            falsePositive(i,j,k) = falsePositive(i,j,k) + 1;
                        end
                    end
                    followerEstimate.x = x(1,index);
                    followerEstimate.y = x(2,index);
                    followerEstimate.v = sqrt(x(3,index)^2 + x(4,index)^2);

                    leaderEstimate.x = xLeader(1,index);
                    leaderEstimate.y = xLeader(2,index);
                    leaderEstimate.v = sqrt(xLeader(3,index)^2 + xLeader(4,index)^2);

                    %v(index) = followerEstimate.v;
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Follower Controller %
                    % Maintain a following distance to the leader
                    velError = error(index-1) - followingDistance;
                    velIntegral = velIntegral + velError*followerT(i);
                    velDerivative = (velError - prevVelError)/followerT(i);

                    % calculate u using PID
                    [followerEstimate, ux, uy] = nhrNavOneGoal(leaderEstimate, followerEstimate, velError,velIntegral,velDerivative);
                    % construct input vector with noise
                    u = [ux ;%+ normrnd(0,sqrt(varInput));
                         0%uy %+ normrnd(0,sqrt(varInput))
                    ];
                    prevVelError = velError;
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    error(index) = sqrt((leaderEstimate.x - follower.x)^2 +  (leaderEstimate.y - follower.y)^2);
                    index = index + 1;
                    % check if state covariance matrix has converged. If yes, apply the
                    % brakes
                    if((round(leaderP,3) - round(prevLeaderP,3))== zeros(6))
                        emergencyBrakesOn = 1;
                    end
                    prevLeaderP = leaderP;
                end
            end
        end
    end
end
figure, hold on;
for k = 1:3
    % Follower Sampling Frequency
    %xlabel('Follower Sampling Frequency (Hz)');
    %errorbar(1./followerT,leaderT*1000* mean(falsePositive(:,:,k),2)', leaderT*1000*std(falsePositive(:,:,k),0,2)')
    % Emergency Braking Deceleration
    %xlabel('Deceleration Threshold (m/sec^2)');
    %errorbar(brakingAcc,leaderT*1000* mean(falsePositive(:,:,k),2)', leaderT*1000*std(falsePositive(:,:,k),0,2)')
    % Radar Variance
    xlabel('Radar Variance (m/s)^2');
    errorbar(radarVar,leaderT*1000* mean(falsePositive(:,:,k),2)', leaderT*1000*std(falsePositive(:,:,k),0,2)')
end
ylabel('False Positives');

%xlabel('Radar Measurments Variance');
legend('90th Percentile','99th Percentile', '999th Percentile');
