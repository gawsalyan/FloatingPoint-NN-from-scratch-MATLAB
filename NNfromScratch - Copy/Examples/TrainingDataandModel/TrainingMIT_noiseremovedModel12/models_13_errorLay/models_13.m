%TestPCAfilt_LSTM_CNNpool_MIT



%%
clear all;
for selItem = 1:44
    load(['Label_',num2str(selItem)]);
    load(['PCAfilt_',num2str(selItem)]);
    load(['R_',num2str(selItem)]);
    load(['ECGfilt_',num2str(selItem)]);
    load(['ECGTrio_',num2str(selItem)]);
    Label(selItem) = {L};
    PCA(selItem) = {A};
    Rinterval(selItem) = {R};
    ECG(selItem) = {ECGSig};
    ECGTrio(selItem) = {ECGSigTrio};
    clear L;
    clear R;
    clear A;
    clear ECGSig;
    clear ECGSigTrio;
end

%%
clc
meanR = mean([Rinterval{1};Rinterval{2};Rinterval{3};Rinterval{4};Rinterval{5};Rinterval{6};Rinterval{7};Rinterval{8};...
        Rinterval{9};Rinterval{10};Rinterval{11};Rinterval{12};Rinterval{13};Rinterval{14};Rinterval{15};Rinterval{16};...
        Rinterval{17};Rinterval{18};Rinterval{19};Rinterval{20}]);
% meanR = mean([Rinterval{1};Rinterval{2};Rinterval{3};Rinterval{4};Rinterval{5};Rinterval{6};Rinterval{7};Rinterval{8};...
%         Rinterval{9};Rinterval{10};Rinterval{11};Rinterval{12};Rinterval{13};Rinterval{14};Rinterval{15};Rinterval{16};...
%         Rinterval{17};Rinterval{18};Rinterval{19};Rinterval{20};Rinterval{21};Rinterval{22};Rinterval{23};Rinterval{24};...
%         Rinterval{25};Rinterval{26};Rinterval{27};Rinterval{28};Rinterval{29};Rinterval{30};Rinterval{31};Rinterval{32};...
%         Rinterval{33};Rinterval{34};Rinterval{35};Rinterval{36};Rinterval{37};Rinterval{38};Rinterval{39};Rinterval{40};...
%         Rinterval{41};Rinterval{42};Rinterval{43};Rinterval{44}]);

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
for upToDataBase = 44:44

%%
clear XTrain;
clear YTrain;
clear RTrain;
clear SigTrain;
clear SigTrioTrain;
minSigLen = 1000;
tests = 1000;
count = 0;
for j = 1:44
%     if (j == 24) %%j == 6
%         continue;
%     end
    minSigLen = length(Label{j})-11;
    clear Y;
        Y = zeros(2,length(Label{j}));
        Y(2,:) = 1;
        Y(1,Label{j}=='N') = 1;     %N as Normal beat under AAMI
        Y(2,Label{j}=='N') = 0;     %N as Normal beat under AAMI
%         Y(1,Label{j}=='R') = 1;     %R as Normal beat under AAMI
%         Y(2,Label{j}=='R') = 0;     %R as Normal beat under AAMI
%         Y(1,Label{j}=='L') = 1;     %L as Normal beat under AAMI
%         Y(2,Label{j}=='L') = 0;     %L as Normal beat under AAMI
%         Y(1,Label{j}=='j') = 1;     %j as Normal beat under AAMI
%         Y(2,Label{j}=='j') = 0;     %j as Normal beat under AAMI
%         Y(1,Label{j}=='e') = 1;     %j as Normal beat under AAMI
%         Y(2,Label{j}=='e') = 0;     %j as Normal beat under AAMI
    for i = tests+1: minSigLen
        RTrain(:,count + i - tests) = mapstd([Rinterval{j}(i+10); Rinterval{j}(i+11); mean(Rinterval{j}(i+1:i+10))]/meanR); 
        SigTrain(:,count + i - tests) = mapstd(movmean(ECG{j}(i+10,:),3));
        SigTrioTrain(:,count + i - tests) = mapstd(downsample(movmean(ECGTrio{j}(i+10,:),3),3));
        temp = PCA{j}(i+1:i+10,1:20); 
        XTrain(1:20,1:10,count + i - tests) = mapstd(temp)';%./median(PCA{j}(i+1:i+10,1:20)).*...
                                                                 %[1 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.01...
                                                                 %0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01])';                   
        YTrain(:,count + i - tests) = Y(:,i+10);
%         if (Y(1,i+10) == 0)
%             display((j-1)*minSigLen + i);
%             plot(1:21, XTrain(1:21,1:9,(j-1)*minSigLen + i) , 'b-');
%             hold on;
%             plot(1:21, XTrain(1:21,10,(j-1)*minSigLen + i) , 'r*-');
%             pause;
%             close all;
%         end

        
    end
    count = count + minSigLen - tests;
end
clear count;

%%
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
clear RTrainAB;
clear SigTrainAB;
abnorm = XTrain(:,:,YTrain(1,:)==0);
abnormR = RTrain(:,YTrain(1,:)==0);
abnormSig = SigTrain(:,YTrain(1,:)==0);
abnormSigTrio = SigTrioTrain(:,YTrain(1,:)==0);
abnormT = [];
abnormRT = [];
abnormSigT = [];
abnormSigTrioT = [];
% for i = 1:2
%     abnormT = cat(3,abnormT, abnorm);  
%     abnormRT = cat(2,abnormRT, abnormR);  
%     abnormSigT = cat(2,abnormSigT, abnormSig);
%     abnormSigTrioT = cat(2,abnormSigTrioT, abnormSigTrio);  
% end
% YabnormT = zeros(2,size(abnormT,3));
% YabnormT(2,:) = 1;
%
XTrainAB = cat(3, XTrain, abnormT);
RTrainAB = cat(2, RTrain, abnormRT);
SigTrainAB = cat(2, SigTrain, abnormSigT);
SigTrioTrainAB = cat(2, SigTrioTrain, abnormSigTrioT);
YTrainAB = cat(2, YTrain, YabnormT);
%
range = randperm(size(XTrainAB,3));
XTrainAB = XTrainAB(:, :,range);
RTrainAB = RTrainAB(:,range);
SigTrainAB = SigTrainAB(:,range);
SigTrioTrainAB = SigTrioTrainAB(:,range);
YTrainAB = YTrainAB(:,range);


%%
 inputV1 = inputVectorLayer_Siva(10 , 'input1');
 LSTM1 = lstmLayer_Siva(50, 20, 2,'lstm1');
 partNet1 = partnetSiva(inputV1,LSTM1);
 
 inputV2 = inputVectorLayer_Siva(3 , 'input2');
 FL_2 = fullyConnectedLayer_Siva(2,'mlp1');
 partNet2 = partnetSiva(inputV2,FL_2);
 
 inputV3 = inputVectorLayer_Siva(252 , 'input3');
 CNN_1 = convolutionLayer_Siva(20,8,5,'mlp1');
 mpool_1 =  maxpoolLayer_Siva(3,3,'mpool1');
 selu_1 = seluLayer_Siva('selu1');
 FL_3 = fullyConnectedLayer_Siva(5,'mlp13');
 partNet3 = partnetSiva(inputV3,CNN_1,mpool_1,selu_1,FL_3);
 
 inputV4 = inputVectorLayer_Siva(252 , 'input4');
 rmse_4  = rmeLayer_Siva('rmse4');
 FL_4 = fullyConnectedLayer_Siva(5,'mlp13');
 partNet4 = partnetSiva(inputV4,rmse_4,FL_4);
 
 concat = concatVector3Layer_Siva('concat');
 MLclass = multiClassLayer_Siva(2,'class'); 
 classnet = classnet3Siva(concat,MLclass);
  
 net5 = multinet3Siva(options, partNet2, partNet3, partNet4, classnet);
 
%%
m = size(YTrainAB,2);
options = containers.Map;
options('Learning_Rate') = 0.001;
options('Weight_Factor') = 1;
options('beta') = 0.9;
options('max_Epochs') = 500;
options('mini_BatchSize') = 1;
options('totalSamples') = m;
options('batches') = (m / options('mini_BatchSize'));
options('TrainAccThresh') = 99.85;

 %%
for i = 1:30
    
net5 = net5.training(net5, RTrainAB, SigTrioTrainAB, SigTrainAB, YTrainAB, options);

A = predictMulitinet3Classes(net5, RTrain, SigTrioTrain, SigTrain);
[~,predictions] = max(A);
[~,labels] = max(YTrain);
TrainingAccuracy = sum((predictions==labels))/length(labels)*100.0;
TestAccuracy = TrainingAccuracy

%%
name = ['model13_1_',num2str(upToDataBase),'_',num2str(ceil(TrainingAccuracy*100.0))];
save(name,'net5')

pause(1);
options('TrainAccThresh') = options('TrainAccThresh') + 0.05;
end

end

%%
clear netSiva;
load ('model10_1_44_9800.mat');
net5 = netSiva;
%%
load ('model13_1_44_9985_9692.mat');

%%
%%
% A = predictMulitinetClasses(net4, XTrain, SigTrain);
% [~,predictions] = max(A);
% [~,labels] = max(YTrain);
% TrainingAccuracy = sum((predictions==labels))/length(labels)*100.0
A = predictMulitinet3Classes(net5, RTrainAB, SigTrioTrainAB, SigTrainAB);
[~,predictions] = max(A);
[~,labels] = max(YTrainAB);
TrainingAccuracy = sum((predictions==labels))/length(labels)*100.0

%%MLclass
figure;
plotconfusion(categorical(labels),categorical(predictions));
%figure;
%montage({net.Nets{1, 1}.Layers{1,2}.Wf, net.Nets{1, 1}.Layers{1,2}.Wi, net.Nets{1, 1}.Layers{1,2}.Wc, net.Nets{1, 1}.Layers{1,2}.Wo, net.Nets{1, 1}.Layers{1,2}.Wy, net.Nets{1, 2}.Layers{1,2}.W, net.Nets{1, 3}.Layers{1,2}.W, net.Nets{1, 3}.Layers{1,3}.W});

%%Check for faulty data
listWrData = find(labels ~= predictions);

%%
minSigLen = 1000;
listWrDataSET = ceil(listWrData/minSigLen);
figure;hist(listWrDataSET,unique(listWrDataSET));
%% plot rr interval
cellstr(YTrainCat);

figure(5); 
plotRangeN = (YTrainAB(2,:)==0);
plotRangeA = (YTrainAB(1,:)==0);

YTrainCat(YTrainAB(1,:) == 1) = 'N';
YTrainCat(YTrainAB(2,:) == 1) = 'A';
YTrainCat = categorical(cellstr(YTrainCat));

%1848, 1849
sellistWrDatatemp = listWrData(YTrainCat(listWrData) == 'A');
%sellistWrDatatemp = (YTrainCat == 'A');
sellistWrData = sellistWrDatatemp(28:36)
A(:,sellistWrData)
sellistWrDataSET = ceil(sellistWrData/minSigLen);

plot(RTrainAB(1,plotRangeN),RTrainAB(2,plotRangeN),'.y' ); 
grid on;
hold on; 
plot(RTrainAB(1,plotRangeA),RTrainAB(2,plotRangeA),'*b' );
plot(RTrainAB(1,sellistWrData),RTrainAB(2,sellistWrData),'or' );

figure(6);

for countSig = 1: length(sellistWrData)
    
    clear tempSig;
    %tempSig = SigTrain(:,sellistWrData(countSig)-9:sellistWrData(countSig));
    %plot(1:252*10,reshape(tempSig,[252*10,1]));
    subplot(length(sellistWrData),1,countSig);
    plot(1:252,SigTrioTrainAB(:,sellistWrData(countSig)));
    hold on;
    text(115,SigTrioTrainAB(115,sellistWrData(countSig)),YTrainCat(sellistWrData(countSig)));
end
%text(252*10-length(sellistWrData)+1:252*10,SigTrain(end,sellistWrData),YTrainCat(sellistWrData));
%text(252-length(sellistWrData)+1:252,SigTrioTrainAB(end,sellistWrData),YTrainCat(sellistWrData));


figure(7);
for countSig = 1: length(sellistWrData)
    subplot(length(sellistWrData),1,countSig);
    clear tempSig;
    tempSig = SigTrainAB(:,sellistWrData(countSig)-9:sellistWrData(countSig));
    plot(1:252*10,reshape(tempSig,[252*10,1]));
    %plot(1:252,SigTrioTrain(:,sellistWrData));
    hold on;
    text(90:251:252*10,SigTrainAB(90,sellistWrData(countSig)-9:sellistWrData(countSig)),YTrainCat(sellistWrData(countSig)-9:sellistWrData(countSig)));
end
%text(252*10-length(sellistWrData)+1:252*10,SigTrainAB(end,sellistWrData),YTrainCat(sellistWrData));



%%
figure(20);
plot(1:20,net5.Nets{1, 2}.Layers{1, 2}.W(2,:));
hold on;
plot(21:40,net5.Nets{1, 2}.Layers{1, 2}.W(2,:));plot(41:60,net5.Nets{1, 2}.Layers{1, 2}.W(3,:));plot(61:80,net5.Nets{1, 2}.Layers{1, 2}.W(4,:));plot(81:100,net5.Nets{1, 2}.Layers{1, 2}.W(5,:));plot(101:120,net5.Nets{1, 2}.Layers{1, 2}.W(6,:));plot(121:140,net5.Nets{1, 2}.Layers{1, 2}.W(7,:));plot(141:160,net5.Nets{1, 2}.Layers{1, 2}.W(8,:));