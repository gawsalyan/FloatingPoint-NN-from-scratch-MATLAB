

%%
clear all;
[Label, PCA, Rinterval] = readTrainData();

%%
clc
meanR = mean([Rinterval{1};Rinterval{2};Rinterval{3};Rinterval{4};Rinterval{5};Rinterval{6};Rinterval{7};Rinterval{8};...
        Rinterval{9};Rinterval{10};Rinterval{11};Rinterval{12};Rinterval{13};Rinterval{14};Rinterval{15};Rinterval{16};...
        Rinterval{17};Rinterval{18};Rinterval{19};Rinterval{20}]);
%%
clc
clear Y
overalllist = [];
for listI = 1:44
list = unique(Label{listI});
overalllist = [overalllist; list];
end
overalllist = unique(overalllist)

%%
Y = zeros(2,length(Label{1}));
Y(2,:) = 1;
Y(1,Label{1}=='N') = 1;
Y(2,Label{1}=='N') = 0;

%%
clc
clear XTrain;
clear YTrain;
clear RTrain;
minSigLen = 1500;
for j = 1:44
    clear Y;
        Y = zeros(2,length(Label{j}));
        Y(2,:) = 1;
        Y(1,Label{j}=='N') = 1;     %N as Normal beat under AAMI
        Y(2,Label{j}=='N') = 0;     %N as Normal beat under AAMI
        Y(1,Label{j}=='R') = 1;     %R as Normal beat under AAMI
        Y(2,Label{j}=='R') = 0;     %R as Normal beat under AAMI
        Y(1,Label{j}=='L') = 1;     %L as Normal beat under AAMI
        Y(2,Label{j}=='L') = 0;     %L as Normal beat under AAMI
        Y(1,Label{j}=='j') = 1;     %j as Normal beat under AAMI
        Y(2,Label{j}=='j') = 0;     %j as Normal beat under AAMI
        Y(1,Label{j}=='e') = 1;     %j as Normal beat under AAMI
        Y(2,Label{j}=='e') = 0;     %j as Normal beat under AAMI
    for i = 1: minSigLen
        RTrain(:,(j-1)*minSigLen + i) = [Rinterval{j}(i+10); Rinterval{j}(i+11); mean(Rinterval{j}(i+1:i+10))]/meanR;  
        temp = PCA{j}(i+1:i+10,1:20);
        XTrain(1:20,1:10,(j-1)*minSigLen + i) = mapstd(temp)';%./median(PCA{j}(i+1:i+10,1:20)).*...
                                                                 %[1 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.01...
                                                                 %0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01])';                   
        YTrain(:,(j-1)*minSigLen + i) = Y(:,i+10);
%         if (Y(1,i+10) == 0)
%             display((j-1)*minSigLen + i);
%             plot(1:21, XTrain(1:21,1:9,(j-1)*minSigLen + i) , 'b-');
%             hold on;
%             plot(1:21, XTrain(1:21,10,(j-1)*minSigLen + i) , 'r*-');
%             pause;
%             close all;
%         end
    end

end

% %%
% clear listTemp;
% clear list;
% list = [];
% for i = 1:20
%     listTemp = unique(Label{i});
%     list = unique([list; listTemp]);
% end
% display(list);
%%
%abnorm
clear XTrainAB;
clear YTrainAB;
abnorm = XTrain(:,:,YTrain(1,:)==0);
abnormR = RTrain(:,YTrain(1,:)==0); 
abnormT = [];
abnormRT = [];
for i = 1:2
    abnormT = cat(3,abnormT, abnorm);  
    abnormRT = cat(2,abnormRT, abnormR);  
end
YabnormT = zeros(2,size(abnormT,3));
YabnormT(2,:) = 1;
%
XTrainAB = cat(3, XTrain, abnormT);
RTrainAB = cat(2, RTrain, abnormRT);
YTrainAB = cat(2, YTrain, YabnormT);
%
range = randperm(size(XTrainAB,3));
XTrainAB = XTrainAB(:, :,range);
RTrainAB = RTrainAB(:,range);
YTrainAB = YTrainAB(:,range);
%%
m = size(YTrainAB,2);
options = containers.Map;
options('Learning_Rate') = 1;
options('Weight_Factor') = 1;
options('beta') = 0.9;
options('max_Epochs') = 150;
options('mini_BatchSize') = 1;
options('totalSamples') = m;
options('batches') = (m / options('mini_BatchSize'));

%% 
% inputV1 = inputVectorLayer_Siva(10 , 'input1');
% LSTM1 = lstmLayer_Siva(50, 20, 2,'lstm1');
% partNet1 = partnetSiva(inputV1,LSTM1);
% 
% inputV2 = inputVectorLayer_Siva(3 , 'input');
% FL_2 = fullyConnectedLayer_Siva(2,'mlp1');
% partNet2 = partnetSiva(inputV2,FL_2);
% 
% concat = concatVectorLayer_Siva('concat');
% FL = fullyConnectedLayer_Siva(10,'mlp1');
% MLclass = multiClassLayer_Siva(2,'class'); 
% classnet = classnetSiva(concat, FL, MLclass);
% 
% net = multinetSiva(options, partNet1, partNet2, classnet);

%%
% 1075 A  ,  28925 N

net = net.training(net, XTrainAB, RTrainAB, YTrainAB, options);

%%
load ('model_1_44_9631.mat')
%%
save('model_1_44_9631','net')
%%
 net = net.setLearningRate(net,0.0005);
%%
A = predictMulitinetClasses(net, XTrain, RTrain);
[~,predictions] = max(A);
[~,labels] = max(YTrain);
TrainingAccuracy = sum((predictions==labels))/length(labels)*100.0

%%
figure;
plotconfusion(categorical(labels),categorical(predictions));
figure;
montage({net.Nets{1, 1}.Layers{1,2}.Wf, net.Nets{1, 1}.Layers{1,2}.Wi, net.Nets{1, 1}.Layers{1,2}.Wc, net.Nets{1, 1}.Layers{1,2}.Wo, net.Nets{1, 1}.Layers{1,2}.Wy, net.Nets{1, 2}.Layers{1,2}.W, net.Nets{1, 3}.Layers{1,2}.W, net.Nets{1, 3}.Layers{1,3}.W});

%% Check for faulty data
listWrData = find(labels ~= predictions);
%%
listWrDataSET = ceil(listWrData/1500);
%%
figure;
plot(1:20,XTrain(:,:,listWrData(1)));

%%
histogram(listWrDataSET)

%%
%%
load ('ECG_203.mat')
%%
listECGSig = mod(listWrData(listWrDataSET==24),1500);
%%
selBeat = 10;
plot(1:252*10,reshape(ECGSig(listECGSig(selBeat)+1:listECGSig(selBeat)+10,:)',[1 252*10]))
