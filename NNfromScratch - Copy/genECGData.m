function [Ts, m, XTrain, YTrain,RRTrain, XTest, YTest, RRTest] = genECGData(name)

[ecg_NN, ecg_A, typeecg_NN, typeecg_A, ecg_NNXrr, ecg_AXrr, ecg_NNw, ecg_Aw] = readPhysionet(name);

%% organise ECG
XTrain = [ecg_NN(1:20,:);ecg_A(1:20,:)];
YTrain = zeros(40,2);
RRTrain = [ecg_NNXrr(1:20,:);ecg_AXrr(1:20,:)];
YTrain(1:20,1) = 1; % Normal
YTrain(21:40,2) = 1; % Abnormal


randP = randperm(40);
XTrain = XTrain(randP,:)';
XTrain = (XTrain - mean(XTrain))./ (max(XTrain)-min(XTrain));

movavgWindow = dsp.MovingAverage(10);
%movavgExp = dsp.MovingAverage('Method','Exponential weighting','ForgettingFactor',0.9);
for i = 1:40
    XTrain(:,i) = movavgWindow(XTrain(:,i));
end

YTrain = YTrain(randP,:)';
RRTrain = RRTrain(randP,:)';
[Ts,m] = size(XTrain);


XTest = [ecg_NN(21:34,:);ecg_A(21:34,:)]';
YTest = zeros(2,size(XTest,2));
RRTest = [ecg_NNXrr(21:34,:);ecg_AXrr(21:34,:)]';
YTest(1,1:14) = 1; % Normal
YTest(2,15:28) = 1; % Abnormal

%%
% figure;
% plot(1:10,XTest);

end
