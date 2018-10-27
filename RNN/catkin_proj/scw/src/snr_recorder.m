
function snr_recorder()
clear all; close all; clc;
echo on;
ret = cell(100,1);     % Holds the return values - 1x1 CSI is 95 bytes big, so this should be upper bound
cur = 0;                        % Current offset into file
count = 1;

filename = 'out.dat';
%uncomment the following lines for real Time analysis
f = fopen([ filename ], 'rb');
if (f < 0)
    error('Couldn''t open file %s', filename);
    return;
end

output_file = fopen('test.csv', 'w');
while 1   
   
    %uncomment the following line for realtime analysis
    [ret,cur,count] = read_bf_file_realTime(f,cur,count,ret);

    csi_entry=ret{count};
    %csi = get_scaled_csi_sm(csi_entry);
    csi = get_scaled_csi(csi_entry);
    eff_SNR = db(get_eff_SNRs(csi), 'pow');
    
   % RSSI=csi_entry.rssi_a;
   % y(ix,1:4)=eff_SNR(1,:); % <- new data
   % y(ix,5) =RSSI;
   
    time = datestr(datetime('now'));
    csv_data = {time, eff_SNR(1,1), time};
    fprintf(output_file,'%s,%f\n', time, x(1),x(2),x(3))
    %dlmwrite('test.csv',csv_data,'delimiter',',','-append');
    end
end

