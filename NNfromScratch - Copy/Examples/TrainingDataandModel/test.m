clear all
clc
[Ts, m, X, Y,RRTrain, WWTrain, XTest, YTest, RRTest] = genECGData();

%%
clc
[A, Z, Zsum] = pqrstProc(X,m);
[ATest, ZTest, ZsumTest] = pqrstProc(XTest,50);
Ts = 6;

%% Multinet
clc
options = containers.Map;
options('Learning_Rate') = 0.5;
options('Weight_Factor') = 3;
options('beta') = 0.9;
options('max_Epochs') = 5000;
options('mini_BatchSize') = 3;
options('totalSamples') = m;
options('batches') = (m / options('mini_BatchSize'));

%% Multinet
inputV1 = inputVectorLayer_Siva(252 , 'input');
FL_0 = fullyConnectedLayer_Siva(16,'mlp1'); Ts = 16;
LSTM_1 = lstmLayer_Siva(8, 1, 2, Ts, 'lstm');
FL_1 = fullyConnectedLayer_Siva(6,'mlp1');
partNet1 = partnetSiva(inputV1,FL_0,LSTM_1);

inputV2 = inputVectorLayer_Siva(3 , 'input');
FL_2 = fullyConnectedLayer_Siva(3,'mlp1');
partNet2 = partnetSiva(inputV2,FL_2);

concat = concatVectorLayer_Siva('concat');
FL = fullyConnectedLayer_Siva(5,'mlp1');
MLclass = multiClassLayer_Siva(2,'class'); 
classnet = classnetSiva(concat, FL, MLclass);

net = multinetSiva(options, partNet1, partNet2, classnet);

%%
clc
%net = net.training(net, reshape(A,[6,40]), Y, options);
net = net.training(net, X, RRTrain, Y, options);


%% Training Accuracy
clc
At = predictMulitinetClasses(net, X, RRTrain);
[~,predictions] = max(At);
[~,labels] = max(Y);
TrainingAccuracy = sum((predictions==labels))/length(labels)
figure;
plotconfusion(categorical(labels),categorical(predictions));

%%
clc
At = predictMulitinetClasses(net, XTest,RRTest);
[~,predictions] = max(At);
[~,labels] = max( YTest );
TestAccuracy = sum((predictions==labels))/length(labels)
figure;
plotconfusion(categorical(labels),categorical(predictions));









%%
%% Single Net
netSingle = netSiva(options,inputV2,FL_2,MLclass);
%%
clc
net = netSingle.training(netSingle, RRTrain, Y, options);