function [XTrain, YTrain, XTest, YTest] =  loadMNISTData()
d = load('mnist.mat');
%%
XTrain = double(d.trainX)'/255; %normalize the data to keep our gradients manageable:
YTrain = double(d.trainY);
XTest = double(d.testX)'/255; %normalize the data to keep our gradients manageable:
YTest = double(d.testY);

%%
%The default MNIST labels record 7 for an image of a seven, 4 for an image of a four, etc. But we’re just building a zero-classifier for now. So we want our labels to say 1 when we have a zero, and 0 otherwise (intuitive, I know). So we’ll overwrite the labels to make that happen:
digitis = 10;
eyeV = eye(digitis);

m = size(YTrain,2);
temp = zeros(digitis, m);
for i = 1:m
    index = YTrain(1,i);
    if index == 0
        index = 10;
    end
    temp(:, i) = eyeV(:,index);
end
YTrain = temp;

m = size(YTest,2);
temp = zeros(digitis, m);
for i = 1:m
    index = YTest(1,i);
    if index == 0
        index = 10;
    end
    temp(:, i) = eyeV(:,index);
end
YTest = temp;


%% shuffle the training set for good measure:
randshuffle = randperm(size(XTrain,2));
XTrain = XTrain(:,randshuffle);
YTrain = YTrain(:, randshuffle);


% %% plot image
% i = 8;
% montage(reshape(XTrain(i,:),28,28)'*255);
% title(['if 0 = 1 else 0, the answer: ',num2str(YTrain(i))]);

end