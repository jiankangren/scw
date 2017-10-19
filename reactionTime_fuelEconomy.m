clear all;
close all;
% sampling time of the leader
leaderT = 0.001;
for sim = 2:2
    % sampling time of the follower
    if(sim == 1)
        followerT = linspace(0.001,0.1,10);
    else
        followerT = ones(1,100)*0.01;
    end
    % braking threshold
    if(sim == 2)
        brakingAcc = linspace(1,9.8,10);
    else
        brakingAcc = ones(1,100)*9.8;
    end
    actualBrakingAcc = 9.8;
    %follower external sensor noises
    if(sim == 3)
    % 0.2 mph -> 11 mph
        radarVar  = linspace(0.0447, 26.8224,10);
    else
        radarVar = ones(1,100)*0.25;
    end
    if(sim == 4)
        lidarVar = linspace(0.0009,1,10);
    else
        lidarVar  = ones(1,100)*0.0009; % 3 cm
    end
    % braking threshold percentiles
    %%% 0.9   -> Z = 1.281552
    %%% 0.99  -> Z = 2.326348
    %%% 0.999 -> Z = 3.090232
    threshold = [1.281552; 2.326348;3.090232];

    % follower inertial sensor noises
    gpsVar = 1; % 1 meter
    encoderVar = 0.01;
    %input noise
    varInput = 0.0001;
    varLeaderInput =0.1; % 2.67^2
    leaderDesiredVel = 26;
    followingDistance = 100;
    %set one goal at 1000 m
    goal = nhrSetGoals(1);
    reactionTime = zeros(10,10,3);
    for i = 1:10 % independent variable
        for j = 1:10 % number of samples 
            for k = 1:3 % number of percentiles
                emergencyBrakesOn = 0;
                follower = nhrInit(10,2.5);
                leader = nhrInit(200,2.5);
                leaderEstimate = nhrInit(100,2.5);
                xRealFollower= [];
                xRealLeader = [];
                uLeader = [];
                xLeader = [];
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
                F = [1  0  leaderT 0 ;
                     0  1  0  leaderT;
                     0  0  1  0 ;
                     0  0  0  1];
                followerEstimateF = [1  0  followerT(i) 0 ;
                     0  1  0  followerT(i);
                     0  0  1  0 ;
                     0  0  0  1];
                leaderEstimateF = [1  0  followerT(i) 0  (followerT(i)^2)/2 0;
                           0  1  0  followerT(i) 0        (followerT(i)^2)/2;
                           0  0  1  0  followerT(i)       0;
                           0  0  0  1  0        followerT(i);
                           0  0  0  0  1        0
                           0  0  0  0  0        1];
                followerEstimateB = [0.5*(followerT(i)^2) 0 ;  
                    0           0.5*(followerT(i)^2);
                    followerT(i)          0 ;
                    0           followerT(i)];
                B = [0.5*(leaderT^2) 0 ;  
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

                leaderR =[lidarVar(i)  0 0   0     ;
                          0  lidarVar(i) 0   0     ;
                          0  0  radarVar(i)  0     ;
                          0  0     0   radarVar(i) ];
                index = 2;
                frequencyCounter = int32(followerT(i)/leaderT);
                error(1) = sqrt((leader.x - follower.x)^2 +  (leader.y - follower.y)^2);
                while(true)
                    %calculate the actual state based on previous input.
                     xRealFollower(:,index) = (F*xRealFollower(:,index-1) + B*u);
                     xRealLeader(:,index) = (F*xRealLeader(:,index-1) + B*uLeader(:,index-1));
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Leader Controller %
                    leader.x = xRealLeader(1,index);
                    leader.y = xRealLeader(2,index);
                    leader.v = sqrt(xRealLeader(3,index)^2 + xRealLeader(4,index)^2);
                    if(emergencyBrakesOn == 1)
                            uLeader(:,index)= [-actualBrakingAcc ;0];
                            reactionTime(i,j,k) = reactionTime(i,j,k) + 1;
                    else
                        % leader PID controller to do cruise control
                        leaderCruiseError = leaderDesiredVel - leader.v;
                        leaderCruiseIntegral = leaderCruiseIntegral + leaderCruiseError*leaderT;
                        leaderCruiseDerivative = (leaderCruiseError - prevLeaderCruiseError)/leaderT;

                        [leader, ux, uy] = nhrCruiseOneGoal(goal, leader, leaderCruiseError, leaderCruiseIntegral, leaderCruiseDerivative);
                        % construct input vector with noise
                        uLeader(:,index)= [ux ;
                             uy;
                        ];
                    end
                    prevLeaderCruiseError = leaderCruiseError;
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
                        leaderZ(:,index) = [xRealLeader(1,index) + normrnd(0,sqrt(lidarVar(i)));
                          xRealLeader(2,index) + normrnd(0,sqrt(lidarVar(i)));
                          xRealLeader(1,index) + normrnd(0,sqrt(radarVar(i)));
                          xRealLeader(2,index) + normrnd(0,sqrt(radarVar(i)))
                         ];
                        % prediction for follower
                        P1 = followerEstimateF*P*followerEstimateF' + Q;
                        S  = H*P1*H' + R;
                        % prediction for leader
                        leaderP1 = leaderEstimateF*leaderP*leaderEstimateF' + leaderQ;
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
                        x(:,index) = followerEstimateF*x(:,index-1) + followerEstimateB*u +  K*(Z(:,index)-H*(followerEstimateF*x(:,index-1)+followerEstimateB*u)); 
                        % state update for leader
                        xLeader(:,index) = leaderEstimateF*xLeader(:,index-1)  +  leaderK*(leaderZ(:,index)-leaderH*(leaderEstimateF*xLeader(:,index-1))); 
                       % check if state covariance matrix has converged
                        if(emergencyBrakesOn == 0)
                            if((round(leaderP,3) - round(prevLeaderP,3))== zeros(6))
                                % if the leader acheived 60 mph
                                %if(abs(xRealLeader(3,index)-26) < 0.1)
                                % if leader estimate reached steady state (0)
                                if((abs(uLeader(1,index))<0.1) && (abs(xLeader(5,index))< 0.1))
                                    % start applying the brakes
                                    emergencyBrakesOn = 1;
                                end
                            end
                        end
                    else
                        x(:,index) = x(:, index-1);
                        xLeader(:,index) = xLeader(:,index-1);
                    end
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Follower's braking condition
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    if(emergencyBrakesOn ==1)
                        if((xLeader(5,index) - sqrt(leaderP(5,5)) * threshold(k)) <= -brakingAcc(i))
                            break;
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
                    % no negative velocities and no going backwards
                    if((xRealLeader(3,index) < 0)|| (xRealLeader(1,index) < xRealLeader(1,index-1)))
                        disp('Leader Stopped');
                        break;
                    end
                    % If leader reaches the goal break
                    if (xRealLeader(1,index) >= 1000)
                       break;
                    end
                    if(index > 100000)
                        break;
                    end

                    index = index + 1;
                    prevLeaderP = leaderP;
                end
                % plot the kalman filter error
%                 xaxis = 1:index;
%                 figure, plot(xaxis,xRealLeader(1,:))
%                 hold on
%                 plot(xaxis,xLeader(1,:))
%                 legend('Real Position (m)', 'Estimated Position (m)')
%                 figure, plot(1:index,xRealLeader(3,:))
%                 hold on
%                 plot(xaxis,xLeader(3,:))
%                 legend('Real Velocity (m/s)', 'Estimated Velocity (m/s)')
%                 figure, plot(xaxis,uLeader(1,:))
%                 hold on
%                 plot(xaxis,xLeader(5,:))
%                 legend('Real Acc (m/s^2)', 'Estimated Acc (m/s^2)')
%                 xlabel('Time (msec)');
%                 title(strcat('Variance of leader acceleration =', num2str(varLeaderInput)));
%                 disp(reactionTime(1,1,1));
            end
        end
    end
    figure, hold on;
    yyaxis left
    for k = 1:3
         if(sim == 1)
            % Follower Sampling Frequency
            xlabel('Follower Sampling Time (msec)')
            errorbar(1000*followerT,leaderT*1000* mean(reactionTime(:,:,k),2)', leaderT*1000*std(reactionTime(:,:,k),0,2)')
        elseif (sim == 2)
            % Emergency Braking Deceleration
            xlabel('Deceleration Threshold (m/sec^2)');
            errorbar(brakingAcc,leaderT*1000* mean(reactionTime(:,:,k),2)', leaderT*1000*std(reactionTime(:,:,k),0,2)')
        elseif (sim == 3)
            % Radar Variance
            xlabel('Radar Variance (m/s)^2');
            errorbar(radarVar,leaderT*1000* mean(reactionTime(:,:,k),2)', leaderT*1000*std(reactionTime(:,:,k),0,2)')
        elseif (sim == 4)
            % LIDAR Variance
            xlabel('LIDAR Variance (m)');
            errorbar(lidarVar,leaderT*1000* mean(reactionTime(:,:,k),2)', leaderT*1000*std(reactionTime(:,:,k),0,2)')
         end
    end
    ylabel('Reaction Time (msec)');

    %xlabel('Radar Measurments Variance');
    %legend('90th Percentile','99th Percentile', '999th Percentile');
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
                uLeader = [];
                xLeader = [];
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
                F = [1  0  leaderT 0 ;
                     0  1  0  leaderT;
                     0  0  1  0 ;
                     0  0  0  1];
                followerEstimateF = [1  0  followerT(i) 0 ;
                     0  1  0  followerT(i);
                     0  0  1  0 ;
                     0  0  0  1];
                leaderEstimateF = [1  0  followerT(i) 0  (followerT(i)^2)/2 0;
                           0  1  0  followerT(i) 0        (followerT(i)^2)/2;
                           0  0  1  0  followerT(i)       0;
                           0  0  0  1  0        followerT(i);
                           0  0  0  0  1        0
                           0  0  0  0  0        1];
                followerEstimateB = [0.5*(followerT(i)^2) 0 ;  
                    0           0.5*(followerT(i)^2);
                    followerT(i)          0 ;
                    0           followerT(i)];
                B = [0.5*(leaderT^2) 0 ;  
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

                leaderR =[lidarVar(i)  0 0   0     ;
                          0  lidarVar(i) 0   0     ;
                          0  0  radarVar(i)  0     ;
                          0  0     0   radarVar(i) ];
                index = 2;
                frequencyCounter = int32(followerT(i)/leaderT);
                error(1) = sqrt((leader.x - follower.x)^2 +  (leader.y - follower.y)^2);
                while(true)
                    if(index == 50000)
                        break;
                    end
                    %calculate the actual state based on previous input.
                     xRealFollower(:,index) = (F*xRealFollower(:,index-1) + B*u);
                     xRealLeader(:,index) = (F*xRealLeader(:,index-1) + B*uLeader(:,index-1));
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Leader Controller %
                    leader.x = xRealLeader(1,index);
                    leader.y = xRealLeader(2,index);
                    leader.v = sqrt(xRealLeader(3,index)^2 + xRealLeader(4,index)^2);
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
                        leaderZ(:,index) = [xRealLeader(1,index) + normrnd(0,sqrt(lidarVar(i)));
                          xRealLeader(2,index) + normrnd(0,sqrt(lidarVar(i)));
                          xRealLeader(1,index) + normrnd(0,sqrt(radarVar(i)));
                          xRealLeader(2,index) + normrnd(0,sqrt(radarVar(i)))
                         ];
                        % prediction for follower
                        P1 = followerEstimateF*P*followerEstimateF' + Q;
                        S  = H*P1*H' + R;
                        % prediction for leader
                        leaderP1 = leaderEstimateF*leaderP*leaderEstimateF' + leaderQ;
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
                        x(:,index) = followerEstimateF*x(:,index-1) + followerEstimateB*u +  K*(Z(:,index)-H*(followerEstimateF*x(:,index-1)+followerEstimateB*u)); 
                        % state update for leader
                        xLeader(:,index) = leaderEstimateF*xLeader(:,index-1)  +  leaderK*(leaderZ(:,index)-leaderH*(leaderEstimateF*xLeader(:,index-1))); 
                    else
                        x(:,index) = x(:, index-1);
                        xLeader(:,index) = xLeader(:,index-1);
                    end
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
                    % no negative velocities and no going backwards
                    if((xRealLeader(3,index) < 0)|| (xRealLeader(1,index) < xRealLeader(1,index-1)))
                        disp('Leader Stopped');
                        break;
                    end
                    % If leader reaches the goal break
                    if (xRealLeader(1,index) >= 1000)
                       break;
                    end
                    if(index > 100000)
                        break;
                    end
                    % check if state covariance matrix has converged and the leader acheived 60 mph. If yes, apply the
                    % brakes
                    if((round(leaderP,3) - round(prevLeaderP,3))== zeros(6))
                        if(abs(xRealLeader(3,index)-26) < 0.01)
                            emergencyBrakesOn = 1;
                        end
                    end
                    index = index + 1;
                    prevLeaderP = leaderP;
                end
            end
        end
    end
    yyaxis right
    for k = 1:3
        if(sim == 1)
            % Follower Sampling Frequency
            errorbar(1000*followerT,leaderT*1000* mean(falsePositive(:,:,k),2)', leaderT*1000*std(falsePositive(:,:,k),0,2)')
        elseif (sim == 2)
            % Emergency Braking Deceleration
            errorbar(brakingAcc,leaderT*1000* mean(falsePositive(:,:,k),2)', leaderT*1000*std(falsePositive(:,:,k),0,2)')
        elseif (sim == 3)
            % Radar Variance
            errorbar(radarVar,leaderT*1000* mean(falsePositive(:,:,k),2)', leaderT*1000*std(falsePositive(:,:,k),0,2)')
        elseif (sim == 4)
            % LIDAR Variance
            errorbar(lidarVar,leaderT*1000* mean(falsePositive(:,:,k),2)', leaderT*1000*std(falsePositive(:,:,k),0,2)')
        end
    end
    ylabel('False Positives');
    legend('90th Percentile','99th Percentile', '999th Percentile','90th Percentile','99th Percentile', '999th Percentile');
    hold off;
    if(sim == 1)   
        % Follower Sampling Frequency
        saveas(gcf,'fig/followerSamplingTimeCombined.fig')
        saveas(gcf,'fig/followerSamplingTimeCombined.jpg')
    elseif (sim == 2)
        % Emergency Braking Deceleration
        saveas(gcf,'fig/decelerationThresholdCombined.fig')
        saveas(gcf,'fig/decelerationThresholdCombined.jpg')
    elseif (sim == 3)
        % Radar Variance
        saveas(gcf,'fig/radarVarianceCombined.fig')
        saveas(gcf,'fig/radarVarianceCombined.jpg')
    elseif (sim == 4)
        % LIDAR Variance
        saveas(gcf,'fig/lidarVarianceCombined.fig')
        saveas(gcf,'fig/lidarVarianceCombined.jpg')
    end
    figure, hold on
    for k = 1:3
        errorbar(mean(falsePositive(:,:,k),2)', leaderT*1000*mean(reactionTime(:,:,k),2)',  leaderT*1000*std(reactionTime(:,:,k),0,2)')
    end
    legend('90th Percentile','99th Percentile', '999th Percentile','90th Percentile','99th Percentile', '999th Percentile');
    ylabel('Reaction Time');
    xlabel('False Positives');
    hold off
    saveas(gcf,strcat('fig/FP_RT',num2str(sim),'.fig'))
    saveas(gcf,strcat('fig/FP_RT',num2str(sim),'.jpg'))
    figure, hold on
    for k = 1:3
        errorbar(leaderT*1000*mean(reactionTime(:,:,k),2)', mean(falsePositive(:,:,k),2)',  std(falsePositive(:,:,k),0,2)')
    end
    legend('90th Percentile','99th Percentile', '999th Percentile','90th Percentile','99th Percentile', '999th Percentile');
    ylabel('False Positives');
    xlabel('Reaction Time');
    hold off
    saveas(gcf,strcat('fig/RT_FP',num2str(sim),'.fig'))
    saveas(gcf,strcat('fig/RT_FP',num2str(sim),'.jpg'))
end