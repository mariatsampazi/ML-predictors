clc; clear all; close all;

%--------------------------------------------------------------------------
% In this code, I save the output of the random_forest_ex1.py code to make
% plots.
%--------------------------------------------------------------------------

% extracting/saving the predicted timestamps
pred_time_Table = readtable('pred_time.xlsx','Sheet','Sheet1','ReadVariableNames',false);

% extracting/saving the predicted throughput 
pred_Table = readtable('pred.xlsx','Sheet','Sheet1','ReadVariableNames',false );

% extracting/saving the actual timestamps
real_time_Table = readtable('real_time.xlsx','Sheet','Sheet1','ReadVariableNames',false);

% extracting/saving the actual throughput
real_Table = readtable('real.xlsx','Sheet','Sheet1','ReadVariableNames',false );

% --------------- table to array for the aforementioned values ---------------------

%% actual throughput and time
throughput_real=table2array(real_Table); 
throughput_real=transpose(throughput_real); % final value
time_real=table2array(real_time_Table); 
time_real=transpose(time_real); % final value

%% predicted throughput and time
throughput_pred=table2array(pred_Table); 
throughput_pred=transpose(throughput_pred); % final value
time_pred=table2array(pred_time_Table); 
time_pred=transpose(time_pred); % final value
% -----------------------------------------------------------------------------------

%%  -- statistics on the mean actual and predicted throughput values--
mean_real=mean(throughput_real);
mean_pred=mean(throughput_pred);
% -------------------------------------------------------------------

% The predicted timestamps and throughput values are not sorted. The RF(Random Forest) model
% outputs the timestamps and the corresponding throughputs in a random
% (usorted way). To make the plots, we sort the predicted timestamps in an
% ascending way. Based on this sorting, we sort the predicted throughput
% values correspondingly. 

%% ---- predicted throughput and timestamps ----
% sort and keep the sort index in "sortIdx"
[time_pred2,sortIdx] = sort(time_pred,'ascend');
% sort using the sorting index
throughput_pred2 = throughput_pred(sortIdx);

%%  ---- Moving Average Plots ----
real=movmean(throughput_real,15); % real data
pred=movmean(throughput_pred2,15); % predicted data

plot(time_real,real,'b'); hold on; title('Moving Averages Plots');
grid on; plot(time_pred2,pred,'r'); legend('Actual','Predicted');


