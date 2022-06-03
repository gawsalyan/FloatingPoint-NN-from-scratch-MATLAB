function [Ts, m, XTrain, XTest, YTrain, YTest] = genSampleData()

Ts = 10;
m = 100;

X_Nbase = [0 0 0 1 2 3 2 1 0 0];


    Total_no_of_examples = m*1.4;
    X_Database = randn([Ts Total_no_of_examples])*0.1; 
    Y_Database = zeros([2 Total_no_of_examples]);
    Y_Database(2,:) = 1;
    randPerm = randperm(Total_no_of_examples,Total_no_of_examples/2);
        for i = randPerm
            X_Database(:,i) = X_Database(:,i) + X_Nbase';
            Y_Database(:,i) = [1;0];
        end

    XTrain = X_Database(:,1:m);
    XTest =  X_Database(:,m:m*1.2);

    YTrain = Y_Database(:,1:m);
    YTest = Y_Database(:,m:m*1.2);
    
    
%     figure(1);
%     for i = 1:100
%         if (YTrain(1,i)==1) 
%             plot(1:10,XTrain(:,i)); 
%             hold on;
%         end
%     end
%     figure(2);
%     for i = 1:100
%         if (YTrain(2,i)==1) 
%             plot(1:10,XTrain(:,i)); 
%             hold on;
%         end
%     end

end