
function TenNodeRealTimePlotFast_RealTime_basicAnalysis()
clear all; close all; clc;
echo on;
ret = cell(100,1);     % Holds the return values - 1x1 CSI is 95 bytes big, so this should be upper bound
cur = 0;                        % Current offset into file
count = 1;
iter=0;
np=10000; % <- data/trace
x=1:np;
y=zeros(np,5);

%filename='../sample_data/2018-1-23-Reflection Analysis/LinkLabManual1-27-02-2018-162404/LinkLabManual1-27-02-2018-162404tx4rx6tr2';
%filename = '../netlink/exp1.dat';
filename = 'out.fifo';
%uncomment the following lines for realTime analysis
f = fopen([ filename ], 'rb');
if (f < 0)
    error('Couldn''t open file %s', filename);
    return;
end

%ret = read_bf_file([filename]);
%subplot(2,3,1);
figure;
lHandle = line(nan, nan);
%plot(x,y(1,1));
ax=gca;
%ylim([0 50]);///

xlabel('Time');
ylabel('SNR [dB]');
ax.YTick=[1,10,20,30,40,50];
title('CSI Measurements');

hold on
%  subplot(2,3,2);
% lh(2)=plot(x,y(:,2));
% ax=gca;
% xlim([1 np]);
% ylim([0 50]);
% ax.YTick=[1,10,20,30,40,50];
% title('RSSI 2');
% 
%  subplot(2,3,3);
% lh(3)=plot(x,y(:,3));
% ax=gca;
% xlim([1 np]);
% ylim([0 50]);
% ax.YTick=[1,10,20,30,40,50];
% title('RSSI 3');
while 1   
   
    %uncomment the following line for realtime analysis
    [ret,cur,count] = read_bf_file_realTime(f,cur,count,ret);

    csi_entry=ret{count};
    %csi = get_scaled_csi_sm(csi_entry);
    csi = get_scaled_csi(csi_entry);
    eff_SNR = db(get_eff_SNRs(csi), 'pow');
    
    %plot(db(abs(squeeze(csi).')))

%     magnit(1,:)=(db(abs(squeeze(csi(1,1, :)).')));
%     magnit(2,:)=(db(abs(squeeze(csi(1,2, :)).')));
%     
%     magnit(3,:)=(db(abs(squeeze(csi(1,3, :)).')));
%     magnit(:,:)=mat2gray(magnit);
%     
     RSSI=csi_entry.rssi_a;
 %    RSSI(2)=csi_entry.rssi_b;
  %   RSSI(3)=csi_entry.rssi_c;
    ix=rem(count-1,np)+1;
     y(ix,1:4)=eff_SNR(1,:); % <- new data
     y(ix,5) =RSSI;
  
%         
%         subplot(2,3,4);
%         h2=plot(1:30,magnit(1,:));
%         ax=gca;
%         xlim([1 30]);
%        % ylim([0 1]);
%         ax.YTick=[1,10,20,30];
%         title('SNR Rx1');
%         
%         subplot(2,3,5);
%         h3=plot(1:30,magnit(2,:));
%         ax=gca;
%         xlim([1 30]);
%        % ylim([0 1]);
%         ax.YTick=[1,10,20,30];
%         title('SNR Rx2');
%         
%         subplot(2,3,6);
%         h4=plot(1:30,magnit(3,:));
%         ax=gca;
%         xlim([1 30]);
%        % ylim([0 1]);
%         ax.YTick=[1,10,20,30];
%         title('SNR Rx3');

%         set(h2,'ydata',magnit(1,:));
%         set(h3,'ydata',magnit(2,:));
%         set(h4,'ydata',magnit(3,:));
 
        xlim([1 ix]);
        set(lHandle,'xdata',[1 :ix], 'ydata',y(1:ix,4));
        %legend('BPSK', 'QPSK', '16QAM', '64QAM','RSSI-A' );
        legend('64QAM');
%         set(lh(2),'ydata',y(:,2));
%         set(lh(3),'ydata',y(:,3));
%         
        drawnow
       % pause(0.00001);
    end
end

