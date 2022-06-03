function [ecg,ann,typeCAT] = readECG(name) 

    [ecg,Fs,tm] = rdsamp(['mitdb/' name],1);
    [ann,type,subtype,chan,num] = rdann(['mitdb/' name],'atr',1);
    
    typeCAT = categorical(cellstr(type));

    %% Plot 2D version of signal and labels
%     N = 10000;
%     figure
%     plot(tm(1:N),ecg(1:N));hold on;grid on
%     plot(tm(ann(ann<N)+1),ecg(ann(ann<N)+1),'ro');

    %% 

    %%

    %%

    %%

    %%
end
