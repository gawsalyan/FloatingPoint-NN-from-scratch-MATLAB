%%
clear all;
clc;
[Ts, m, XTrain, XTest, YTrain, YTest] = genSampleData(200,2);
X = XTrain;   
Y = YTrain;

%%
clear all
clc
[Ts, m, XTrain, YTrain,RRTrain, WWTrain, XTest, YTest, RRTest] = genECGData('100');

downsampleR = 1;
X = downsample(XTrain,downsampleR);    %X - n_x * m  
XTest = downsample(XTest,downsampleR); 
Y = YTrain;
D = RRTrain./mean(RRTrain);
E = RRTest./mean(RRTest);


%%
clc
options = containers.Map;
options('Learning_Rate') = 1;
options('Weight_Factor') = 1;
options('beta') = 0.9;
options('max_Epochs') = 500;
options('mini_BatchSize') = 5;
options('totalSamples') = m;
options('batches') = (m / options('mini_BatchSize'));

%Ts = 17;
inputV1 = inputVectorLayer_Siva(3 , 'input');
LSTM_1 = lstmLayer_Siva(10, 1, Ts, 'lstm');
FL_1 = fullyConnectedLayer_Siva(ceil(Ts/2),'mlp1');
partNet1 = partnetSiva(inputV1,LSTM_1,FL_1);

inputV2 = inputVectorLayer_Siva(2 , 'input');
FL_2 = fullyConnectedLayer_Siva(2,'mlp1');
partNet2 = partnetSiva(inputV2,FL_2);

Tww = 64;
inputV3 = inputVectorLayer_Siva(Tww , 'input');
LSTM_3 = lstmLayer_Siva(10, 1, Tww, 'lstm');
FL_3 = fullyConnectedLayer_Siva(ceil(Tww/2),'mlp1');
partNet3 = partnetSiva(inputV3,LSTM_3,FL_3);

concat = concatVectorLayer_Siva('concat');
FL = fullyConnectedLayer_Siva(10,'mlp1');
MLclass = multiClassLayer_Siva(2,'class'); 
classnet = classnetSiva(concat, FL, MLclass);

net = multinetSiva(options, partNet1, partNet3, classnet);

%%
clc
net = net.training(net, X,WWTrain, Y, options);

%% Training Accuracy
A = predictMulitinetClasses(net, X, Y);
[~,predictions] = max(A);
[~,labels] = max(YTrain);
TrainingAccuracy = sum((predictions==labels))/length(labels)
figure;
plotconfusion(categorical(labels),categorical(predictions));

%% Test Accuracy
A = predictMulitinetClasses(net, XTest, YTest);
[~,predictions] = max(A);
[~,labels] = max(YTest);
TestAccuracy = sum((predictions==labels))/length(labels)
figure;
plotconfusion(categorical(labels),categorical(predictions));


%%
display(['activations...',net.Nets{1}.Layers{2}.Name])
A = net.Nets{1}.Layers{2}.predict(net.Nets{1}.Layers{2},trial)%ones(Ts,1))
B = net.Nets{1}.Layers{3}.predict(net.Nets{1}.Layers{3},A)
C = net.Nets{2}.Layers{2}.predict(net.Nets{2}.Layers{2},ones(2,1))
D = net.Nets{3}.Layers{2}.predict(net.Nets{3}.Layers{2},[B;C])
E = net.Nets{3}.Layers{3}.predict(net.Nets{3}.Layers{3},D)
%%
clc
BB = net.Nets{3}.Layers{3}.W;
BBb = net.Nets{3}.Layers{3}.b
CC = net.Nets{3}.Layers{2}.W;
CCb = net.Nets{3}.Layers{2}.b;
DD = (( [0;1] - BBb)\ BB);
DD(DD > 0) = 1; DD(DD<0) = 0;
DD
DDconcat = (( DD  - CCb)\CC);
DDconcat(DDconcat > 0) = 1; DDconcat(DDconcat<0) = 0;
DDconcat
EE = net.Nets{2}.Layers{2}.W;
EEb = net.Nets{2}.Layers{2}.b;
FFin2 = (DDconcat(:,end-1:end) - EEb)\EE;
FFin2(FFin2 > 0) = 1; FFin2(FFin2<0) = 0;
FFin2
GG = net.Nets{1}.Layers{3}.W;
HHin1 =  DD(1:end-2) * GG 
%%
clc
checkLayer = net.Nets{3}.Layers{end};
DD = checkLayer.calculatecontribution(checkLayer,[1;0])  
checkLayer = net.Nets{3}.Layers{end-1};
DD = checkLayer.calculatecontribution(checkLayer,DD)  
checkLayer = net.Nets{2}.Layers{end};
DD = checkLayer.calculatecontribution(checkLayer,DD(end-1:end))


