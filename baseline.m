robot = nhrInit(100,100);
cla(handles.axes1);
cla(handles.axes2);
canvasInit(handles.axes1);
%Plot initial config
%robotPlot = nhrPlot(robot, 'r');
%set one goal
goal = nhrSetGoals(1);
%plot goal
nhrPlotGoals(goal);
%run simulation
%Init Control Parameters
dt = 0.2;
t = 0;
x_true= [];
% attack enable
enableAttack = 1;

error(1) = sqrt((goal(1).x - robot.x)^2 +  (goal(1).y - robot.y)^2);
desiredVel = 6;
velIntegral = 0;
prevVelError = 0;
%sensor noises
var1 = 0.1;
var2 = 0.2;
%input noise
varInput = 0.01;
%initial state
u = [0;
     0;
     0.1;
     0.1
     ];
x_true(:,1) = [20;
          50
          ];
x(:,1) = [round(100 *rand(1,1));
     round(100 *rand(1,1));
     0;
     0
    ];
F = [1  0  dt 0 ;
     0  1  0  dt;
     0  0  1  0 ;
     0  0  0  1];
 
B = [0 0 0 0 ;  
     0 0 0 0 ;
     0 0 1 0 ;
     0 0 0 1 ];
 
P = [10 0  0  0 ;
     0  10  0  0 ;
     0  0  1  0 ;
     0  0  0  1];
 
H = [1  0  0  0 ;
     0  1  0  0 ;
     1  0  0  0 ;
     0  1  0  0];

Q = [0  0  0  0 ;
     0  0  0  0 ;
     0  0  varInput  0 ;
     0  0  0  varInput];
 
R = [var1  0  0     0 ;
     0  var1  0     0 ;
     0  0  var2     0 ;
     0  0     0   var2];
index = 2;
v(1) = 0;
while(error(index-1) > 2)
    %calculate the actual state based on previous input.
    % x_true = (F*x_true + B*u);
    x_true(1,index) = x_true(1,index-1)+ dt *u(3);
    x_true(2,index) = x_true(2,index-1)+ dt *u(4);
    %  x_true_sim{k} = x_true;
    % generate sensor mesaurements with noise.
    Z(:,index) = [x_true(1,index) + normrnd(0,sqrt(var1));
                  x_true(2,index) + normrnd(0,sqrt(var1));
                  x_true(1,index) + normrnd(0,sqrt(var2));
                  x_true(2,index) + normrnd(0,sqrt(var2))
                 ];
    % attack
    if(enableAttack == 1)
        Z(2,index) = Z(2,index) + index/10;
    end
    % prediction
    P1 = F*P*F' + Q;
    S  = H*P1*H' + R;
    
    % measurements update
    K = P1*H'*inv(S);
    P = P1 - K*H*P1;
    x(:,index) = F*x(:,index-1) + B*u +  K*(Z(:,index)-H*(F*x(:,index-1)+B*u)); 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    robot.x = x(1,index);
    robot.y = x(2,index);
    robot.v = sqrt(x(3,index)^2 + x(4,index)^2);
    v(index) = robot.v;
    % velocity_sim{k} = robot.v;
    
    %PID controller
    velError = desiredVel - robot.v;
    velIntegral = velIntegral + velError*dt;
    velDerivative = (velError - prevVelError)/dt;

    % calculate u using PID
    [robot, ux, uy] = nhrNavOneGoal(goal, robot, velError,velIntegral,velDerivative);
    % construct input vector with noise
    u = [0;
         0;
         ux + normrnd(0,sqrt(varInput));
         uy + + normrnd(0,sqrt(varInput))
        ];
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %plot robot position and velocity
    robot.x = x_true(1,index);
    robot.y = x_true(2,index);
    robot.theta =  atan2(u(4),u(3));
    axes(handles.axes1);
    %robotPlot = nhrUpdateRobotPlot(robot, robotPlot, 'r');
    plot(robot.x,robot.y, '.', 'Color', 'r');
    plot(Z(1,index), Z(2,index), '.', 'Color', 'b');
    plot(Z(3,index), Z(4,index), '.', 'Color', 'g');
    axes(handles.axes2); %set the current axes to axes2
    ylabel('velocity (m/s)');
    xlabel('time (sec)');
    plot(t,robot.v,'.','color', 'g');
    hold on

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    prevVelError = velError;
    error(index) = sqrt((goal(1).x - robot.x)^2 +  (goal(1).y - robot.y)^2);
    
    t = dt + t;
    index = index + 1;
    pause(dt);
end
hold off
axes(handles.axes1);
hold off
%plotting the speed of the robot and distance to the goal (estimated)
time = 0:dt:t+dt;
time = time(1:length(v));
figure, plot(time,v); xlabel('time (sec)'); ylabel('speed (m/s)'); title('estimated speed of the robot');
figure, plot(time,error); xlabel('time (sec)'); ylabel('Distance to goal (m)'); title('Estimated distance to goal over time');

% plotting the estimated x and y values of the robot
figure, plot(time,x(1,:)); xlabel('time (sec)'); ylabel('estimated x (m)'); title('estimated x position of the robot');
figure, plot(time,x(2,:)); xlabel('time (sec)'); ylabel('estimated y (m)'); title('estimated y position of the robot');

%plotting the estimated x and y position together with the sensory
%measurements and the actual values for comparison.
figure, plot(time,x(1,:),'b'); hold on;
plot(time,Z(1,:), 'r'); hold on;
plot(time,Z(3,:), 'g'); hold on; 
plot(time,x_true(1,:), 'k'); xlabel('time (sec)'); ylabel('x position (m)'); hold on; 
legend('estimated x position', 'sensor 1','sensor 2', 'actual x');

figure, plot(time,x(2,:),'b'); hold on;
plot(time,Z(2,:), 'r'); hold on;
plot(time,Z(4,:), 'g'); hold on; 
plot(time,x_true(2,:), 'k'); xlabel('time (sec)'); ylabel('y position (m)'); hold on; 
legend('estimated y position', 'sensor 1','sensor 2', 'actual y');


a=2000;
b=300;
c=a+(b)*randn(100000,1);
prova=fitdist(c,'normal');
pdfval=pdf(prova,linspace(a-4*b,a+4*b,100));
pctval = prctile(pdfval,90)
