%%
clear all;
load 'ECGfilt.mat'
load 'ANN.mat'
load 'TYPE.mat'

%%
clc
clear Y
overalllist = [];
for listI = 1:44
list = unique(TYPE{listI});
overalllist = [overalllist; list];
end
overalllist = unique(overalllist)

%% Available Beat Types

load 'PCAof_NBeats.mat';
Fs = 360;

%%
for selItem = 1:44
    M = ECGfilt{selItem};
    ann = ANN{selItem};
    type = TYPE{selItem};

    clear A;
    clear L;
    clear R;
    clear ECGSig;
    clear ECGSigTrio;
    clear Loc;
    beatCount = 0;

    L = [];
    R = [];
    Loc = [];

    for beatPOS = 5: length(ann)-5
         if (type(beatPOS) == '+' || type(beatPOS) == '|' || type(beatPOS) == '~' || type(beatPOS) == 'x' || type(beatPOS) == '!'...
                 || type(beatPOS) == '[' || type(beatPOS) == ']' || type(beatPOS) == '"')
            display(type(beatPOS));
         else
            beatCount = beatCount+1; 
            ECGSig(beatCount,:) = M(ann(beatPOS)-89:ann(beatPOS)+162,1);
            ECGSigTrio(beatCount,:) = M(ann(beatPOS)-341:ann(beatPOS)+414,1);
            A(beatCount,:) = findPCAcoeff(ECGSig(beatCount,:)', SB);
            L = [L; type(beatPOS)];
            Loc = [Loc; ann(beatPOS)];
            if(beatCount>1)
                R = [R; Loc(end) - Loc(end-1)];
            else
                R = [R; Loc(end)-ann(beatPOS-1)];
            end
            display(beatPOS);   
         end
    end

    save(['Label_',num2str(selItem)],'L');
    save(['PCAfilt_',num2str(selItem)],'A');
    save(['R_',num2str(selItem)],'R');
    save(['ECGfilt_',num2str(selItem)],'ECGSig');
    save(['ECGTrio_',num2str(selItem)],'ECGSigTrio');
end