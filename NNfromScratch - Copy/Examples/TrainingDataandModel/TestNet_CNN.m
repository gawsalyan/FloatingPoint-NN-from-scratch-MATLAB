%%
clear all
clc
[Ts, m, XTrain, XTest, YTrain, YTest] = genSampleData(100,2);
X = XTrain;   
Y = YTrain;


%%
clear all
clc
%%
[Ts, m, XTrain, YTrain,RRTrain, WWTrain, XTest, YTest, RRTest] = genECGData();

X = XTrain;    %X - n_x * m   
Y = YTrain;

%%
D = RRTrain./mean(RRTrain);
%%
clc
options = containers.Map;
options('Learning_Rate') = 0.001;
options('Weight_Factor') = 3;
options('beta') = 0.9;
options('max_Epochs') = 5000;
options('mini_BatchSize') = 5;
options('totalSamples') = m;
options('batches') = (m / options('mini_BatchSize'));

inputV = inputVectorLayer_Siva(252, 'input');
LSTM_1 = lstmLayer_Siva(64, 1, 2, 'lstm');
CN_1 = convolutionLayer_Siva(20,5,5,'cnn1');
CN_2 = convolutionLayer_Siva(5,1,1,'cnn2');
FL_1 = fullyConnectedLayer_Siva(64,'mlp1');
FL_2 = fullyConnectedLayer_Siva(16,'mlp1');
MLclass = multiClassLayer_Siva(2,'class'); 
net = netSiva(options,inputV,CN_1,CN_2,MLclass);

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
t=1:50;
Xtest =  XTest(:,t);
Ytest = YTest(:,t);
A = predictClasses(net, Xtest);
[~,predictions] = max(A);
[~,labels] = max( Ytest );
TestAccuracy = sum((predictions==labels))/length(labels)
figure;
plotconfusion(categorical(labels),categorical(predictions));


%%
netS = net;
A1 = netS.Layers{2}.predict(netS.Layers{2}, X(:,1))
A2 = netS.Layers{3}.predict(netS.Layers{3}, A1)
A3 = netS.Layers{4}.predict(netS.Layers{4}, A2)
