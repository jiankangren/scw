function [robot,ux,uy]= nhrNavOneGoal(goal, robot, error, integral, derivative)
    %Init Control Parameters
    u_max = 2.6;
    v_max = 26;
%    kThetaP = 2.5;
    kVP = 0.5;
    kVI = 0.02;
    kVD = 0.05;
    %calculate new robot angle
    thetaNew = atan2(goal.y - robot.y , goal.x - robot.x);
    % calculate new robot input
    u = (kVP*error + kVI*integral + kVD*derivative);
    %update robot 
    if u > u_max
        u = u_max;
    end
    ux = u * cos(thetaNew);
    uy = u * sin(thetaNew);  
    if robot.v >= v_max
          ux = 0;
    end
end

