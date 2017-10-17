function [robot,vx,vy]= nhrCruiseOneGoal(goal, robot, error, integral, derivative)
    %Init Control Parameters
%    kThetaP = 2.5;
    kVP = 1;
    kVI = 0;
    kVD = 0.05;
    %calculate new robot angle
    thetaNew = atan2(goal.y - robot.y , goal.x - robot.x);
    % calculate new robot velocity
    v = (kVP*error + kVI*integral + kVD*derivative);
    %update robot 
    vx = v * cos(thetaNew);
    vy = v * sin(thetaNew);
    if(vx > 10)
        vx = 10;
    elseif (vx < -10)
        vx = -10;
    end
end


