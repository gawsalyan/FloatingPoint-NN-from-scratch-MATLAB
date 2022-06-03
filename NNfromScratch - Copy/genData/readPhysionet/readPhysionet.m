function [ecg_NN, ecg_A, typeecg_NN, typeecg_A, ecg_NNXrr, ecg_AXrr, ecg_NNw, ecg_Aw] = readPhysionet(name)
% MIT-BIH ECG arrhythmia database [43] is used to evaluate the proposed algorithm 
% and compare its performance with previous works. This database consists of 
% ECG recordings of 48 patients. Each record has two leads. The ?rst lead is 
% modi?ed limb lead II (MLII). The second lead is modi?ed lead V1 or in some 
% cases V2, V4 or V5. Two or more cardiologists independently annotated each 
% record

% The database contains two sets of data, which we call DS100 and DS200. DS100 
% numbered from 100 to 124 with some numbers missing) includes representative 
% samples of the variety of ECG waveforms and artifacts that an arrhythmia detector 
% might encounter in routine clinical practice. DS200 (numbered from 200 to 234 
% with some numbers missing) includes complex ventricular, junctional, and supraventricular 
% arrhythmias and conduction abnormalities. According to AAMI standards, the 
% records which contain paced beats, i.e. records 102, 104, 107, and 217, are 
% excluded from the study [25]. 

% Global training data is formed by randomly selecting representative heartbeats 
% from all arrhythmia classes in DS100 records. 
% Patient-speci?c training data is the ?rst ?ve minutes of a patient’s record in DS200. This 
% is in compliance with AAMI standards [25]. Test data is all the records in 
% DS200. The ?rst ?ve minutes of all the records are skipped in the test data.

%%
DS100 = [100, 101, ...               %102 excluded contain paced heartbeats
    103, ...                           %104 excluded contain paced heartbeats
    105, 106,...                     %107 excluded contain paced heartbeats
    108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124];



%%
Total_numberOfBeats = 0;
Total_numberOfBeatsN = 0;
Total_numberOfBeatsA = 0;

numberofSamples = 1000;

prev_countA = 1;

ecg_N_gl = zeros(numberofSamples,252);
ecg_NN = zeros(numberofSamples,252);
ecg_NNw = zeros(numberofSamples,64);
ecg_A = zeros(numberofSamples,252);
ecg_Aw = zeros(numberofSamples,64);
ecg_NXrr = zeros(numberofSamples,3);
ecg_AXrr = zeros(numberofSamples,3);

for i=1:1 %length(DS100)
    display('Reading .. ',num2str(DS100(i))); 
    ecg_N = zeros(numberofSamples,252);
    prev_countN = 1;  
    [ecg,ann,type] =  readECG(name); %%readECG(num2str(DS100(i)));
    numberOfBeats = length(ann);
    
    omit_Edge = 5;
    
    for j=1+omit_Edge:numberOfBeats-omit_Edge
        range = ann(j)-89:ann(j)+162;
        if (type(j)=='N')
            ecg_N(prev_countN,:) = ecg(range);
            typeecg_N(prev_countN) = type(j);
            ecg_NXrr(prev_countN,:) = Xrr(ann(j-5:j+5));
            
            prev_countN = prev_countN+1;
        else
            ecg_A(prev_countA,:) = ecg(range);
            typeecg_A(prev_countA) = type(j);
            ecg_AXrr(prev_countA,:) = Xrr(ann(j-5:j+5));
            [dbwcA ,dbwcD] = dwt(downsample(ecg_A(prev_countA,:),2),'db2');
            ecg_Aw(prev_countA,:) = dbwcA;
            
            prev_countA = prev_countA+1;
        end
        
%         if (Total_numberOfBeatsN+prev_countN==3324)
%             print('hello..');
%         elseif (Total_numberOfBeatsN+prev_countN==3325)
%             print('hello..');
%         end
    end
%     RRi = (zeros(numberOfBeats-3,1)); %rejecting the first and last beat 
%     RRi  = ann(3:end-1)'-ann(2:end-2)';
%     RRavg = mean(RRi);
    numberOfBeatsN = prev_countN -1;
    numberOfBeatsA = prev_countA - Total_numberOfBeatsA-1;
    
    ecg_N_gl(Total_numberOfBeatsN+1:Total_numberOfBeatsN+numberOfBeatsN,:) = ecg_N(1:numberOfBeatsN,:); 
    typeecg_N_gl(Total_numberOfBeatsN+1:Total_numberOfBeatsN+numberOfBeatsN) = typeecg_N(1:numberOfBeatsN); 
    ecg_N_glXrr(Total_numberOfBeatsN+1:Total_numberOfBeatsN+numberOfBeatsN,:) =  ecg_NXrr(1:numberOfBeatsN,:); 
    
    range = Total_numberOfBeatsA+1:prev_countA-1;
    
    if (numberOfBeatsN<numberOfBeatsA)
        randrange = randperm(Total_numberOfBeatsN+numberOfBeatsN, numberOfBeatsA);
        %ecg_NN(range,:) = ecg_N_gl(randrange,:);
        disp('N < A');
        
        ecg_NN(range,:) = ecg_N_gl(randrange,:);
        typeecg_NN(range) = typeecg_N_gl(randrange);
        ecg_NNXrr(range,:) = ecg_N_glXrr(randrange,:);
        
        for k = range
           [dbwcA ,dbwcD] = dwt(downsample(ecg_N_gl(k,:),2),'db2');
           ecg_NNw(k,:) = dbwcA;
        end
    else
        randrange = randperm(numberOfBeatsN, numberOfBeatsA);
        
        ecg_NN(range,:) = ecg_N(randrange,:);
        typeecg_NN(range) = typeecg_N(randrange);
        ecg_NNXrr(range,:) = ecg_NXrr(randrange,:);
%         ecg_NN(range,:) = ecg_N(1:numberOfBeatsA,:);
%         typeecg_NN(range) = typeecg_N(1:numberOfBeatsA);
        for k = range
           [dbwcA ,dbwcD] = dwt(downsample(ecg_N(k,:),2),'db2');
           ecg_NNw(k,:) = dbwcA;
        end
    end
    
    Total_numberOfBeats = Total_numberOfBeats + numberOfBeats;
    Total_numberOfBeatsN = Total_numberOfBeatsN + numberOfBeatsN;
    
    
    Total_numberOfBeatsA = Total_numberOfBeatsA + numberOfBeatsA;  
    
    
end


%%
% plots all channels
% testSig_len = 60000;
% Fs = 360;
% 
% plot(ecg(1:testSig_len))
% hold on
% % plots markers on annotation positions 'ann' as # of sample
% % markers are adjusted to channel 1
% plotlev = testSig_len/Fs*1.3;
% 
% plot(ann(1:plotlev),ecg(ann(1:plotlev),1),'o');
% % plots marker labels
% text(ann(1:plotlev),ecg(ann(1:plotlev),1), char(type(1:plotlev)));
% hold off


%end
