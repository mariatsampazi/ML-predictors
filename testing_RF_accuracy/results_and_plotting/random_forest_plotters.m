clc; clear all; close all;

folder = '/Users/maria/Documents/GitHub/oran_efforts/testing_RF_accuracy/results_and_plotting';

%--------------------------------------------------------------------------
% In this code, I save the output of the py code to make
% plots.
%--------------------------------------------------------------------------

% extracting/saving the predicted timestamps
pred_time_Table = readtable('pred_time_1.csv');

% extracting/saving the predicted throughput 
pred_Table = readtable('pred_1.csv' );

% extracting/saving the actual timestamps
real_time_Table = readtable('real_time_1.csv');

% extracting/saving the actual throughput
real_Table = readtable('real_1.csv' );

% actual labels from the dataset
labels = readtable('labels.csv' );
labels=table2array(labels);
labels=transpose(labels);
labels=labels./1000;

% --------------table to array for the aforementioned values---------------
%% actual throughput and time
throughput_real=table2array(real_Table); 
throughput_real=transpose(throughput_real); % final value
time_real=table2array(real_time_Table); 
time_real=transpose(time_real); % final value

% the input was x1000 scaled for the training and testing of the algorithm,
% now it is again downscaled
throughput_real=throughput_real./1000;
time_real=time_real./1000;

%% predicted throughput and time
throughput_pred=table2array(pred_Table); 
throughput_pred=transpose(throughput_pred); % final value
time_pred=table2array(pred_time_Table); 
time_pred=transpose(time_pred); % final value

% downscale by 1000 the by x1000 set used for predictions
throughput_pred=throughput_pred./1000;
time_pred=time_pred./1000;

%--------------------------------------------------------------------------

%%  -- statistics on the mean actual and predicted throughput values--
mean_real=mean(throughput_real);
mean_pred=mean(throughput_pred);
% -------------------------------------------------------------------

% The predicted timestamps and throughput values are not sorted. The RF
% (Random Forest) model
% outputs the timestamps and the corresponding throughputs in a random
% (usorted way). To make the plots, we sort the predicted timestamps in an
% ascending way. Based on this sorting, we sort the predicted throughput
% values correspondingly. 

%% ---- predicted throughput and timestamps ----
% sort and keep the sort index in "sortIdx"
[time_pred2,sortIdx] = sort(time_pred,'ascend');
% sort using the sorting index
throughput_pred2 = throughput_pred(sortIdx);

%% actual
% sort and keep the sort index in "sortIdx_r"
[time_real2,sortIdx_r] = sort(time_real,'ascend');
% sort using the sorting index
throughput_real2 = throughput_real(sortIdx_r);


%%  ---- Moving Average Plots ----
% window is set to 1500
real=movmean(throughput_real2,250); % real data (50, 15 work)
pred=movmean(throughput_pred2,250); % predicted data

figure(1);
plot(time_real2,real,'b','LineWidth',1); hold on;
title('Moving Averages Plots for the testing set and the predictions set'); xlabel('Timestamps'); ylabel('Tx brate downlink [Mbps]');
grid on; plot(time_pred2,pred,'--r'); legend('Actual','Predicted');


%% -------------do a histogram for the actual throughput data--------------
hist_real=throughput_real;
%--------------------------------------------------------------------------
figure(2); hold on; grid on; title('Histogram for the actual downlink throughput');
ylabel('Number of occurences in the testing set'); xlabel('Tx brate downlink in [Mbps] values')
histogram(hist_real);

figure(3); 
hold on; grid on; title('Histogram for the labels - downlink throughput of the input dataset');
ylabel('Number of occurences in the input set'); xlabel('Tx brate downlink in [Mbps] values')
histogram(labels);

%% Performance metrics
errors = abs(throughput_pred2 - throughput_real2);
disp('Metrics for Random Forest Trained on Expanded Data:');
%disp('Mean value of errors:')
%disp(mean(errors));

%Calculate mean absolute percentage error (MAPE)
mean_mape = mean(100 * (errors / throughput_real2));
accuracy = 100-mean_mape;
fprintf('The accuracy of the Random Forest Regressor Algorithm is %.1f%%. ', round(accuracy,1));
fprintf('\n') ;


%}