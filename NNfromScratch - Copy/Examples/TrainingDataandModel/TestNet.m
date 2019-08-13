%%
clear all
clc
[Ts, m, XTrain, XTest, YTrain, YTest] = genSampleData(100);
X = XTrain;   
Y = YTrain;


%%
clear all
clc
[Ts, m, XTrain, YTrain,RRTrain, XTest, YTest, RRTest] = genECGData('100');

X = XTrain;    %X - n_x * m   
Y = YTrain;

%%
D = RRTrain./mean(RRTrain);
%%
clc
options = containers.Map;
options('Learning_Rate') = 1;
options('Weight_Factor') = 3;
options('beta') = 0.9;
options('max_Epochs') = 100;
options('mini_BatchSize') = 1;
options('totalSamples') = m;
options('batches') = (m / options('mini_BatchSize'));

inputV = inputVectorLayer_Siva(3 , 'input');
LSTM_1 = lstmLayer_Siva(8, 2, Ts, 'lstm');
FL_1 = fullyConnectedLayer_Siva(3,'mlp1');
FL_2 = fullyConnectedLayer_Siva(5,'mlp1');
MLclass = multiClassLayer_Siva(2,'class'); 
net = netSiva(options,inputV,LSTM_1,FL_2,MLclass);

%%
clc
net = net.training(net, X, Y, options);

%% Training Accuracy
A = predictClasses(net, X);
[~,predictions] = max(A);
[~,labels] = max(YTrain);
TrainingAccuracy = sum((predictions==labels))/length(labels)
figure;
plotconfusion(categorical(labels),categorical(predictions));

%% Test Accuracy
clc
t=1;
Xtest =  XTest(:,t);
Ytest = YTest(:,t);
A = predictClasses(net, Xtest);
[~,predictions] = max(A);
[~,labels] = max( Ytest );
TestAccuracy = sum((predictions==labels))/length(labels)
%figure;
%plotconfusion(categorical(labels),categorical(predictions));

