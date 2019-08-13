classdef multinetSiva

    properties
      Name = 'NET';
      no_ofLayer = 2;
      no_ofNETS = 3;
      Layers;
      Nets;
    end
    
    methods       
      function netSiva = multinetSiva(options, partNet1, partNet2, classnet)
           netSiva.no_ofNETS = nargin-1;
           netSiva.no_ofLayer = partNet1.no_ofLayer + partNet2.no_ofLayer + classnet.no_ofLayer;
           
            partNet1 = partNet1.initNet(partNet1,options);
            partNet2 = partNet2.initNet(partNet2,options);
            classnet = classnet.initNet(classnet,partNet1,partNet2,options);
           
           netSiva.Nets = {partNet1,partNet2,classnet};              
      end   
    end
    
    methods(Static)
        
       function netSiva = setLearningRate(netSiva, LR)
           netSiva.Nets{1} = netSiva.Nets{1}.setLearningRate(netSiva.Nets{1},LR);
           netSiva.Nets{2} = netSiva.Nets{2}.setLearningRate(netSiva.Nets{2},LR);
           netSiva.Nets{3} = netSiva.Nets{3}.setLearningRate(netSiva.Nets{3},LR);
       end 
       
       function A = predict()
       end
       
       function netSiva = training(netSiva, X1,X2, Y, options)
           figure();
           startSpot = 0;
           step = 0.1;
           costX = 0;
           maxcost = 0;
           costprev = 0;
            for i = 1:options('max_Epochs')  
                range = randperm(size(Y,2));
                X1 = X1(:,range);
                X2 = X2(:,range);
                Y = Y(:,range);
                for j=1:options('batches')

                    start = (j-1)*options('mini_BatchSize') + 1;
                    stop = min(start+options('mini_BatchSize') - 1, options('totalSamples'));
                    m_batch = stop - start + 1;
                    
                    if size(size(X1),2) == 3
                         X1_miniBatch = X1(:,:,start:stop);
                    else
                        X1_miniBatch = X1(:,start:stop);
                    end
                    if size(size(X2),2) == 3
                         X2_miniBatch = X2(:,:,start:stop);
                    else
                        X2_miniBatch = X2(:,start:stop);
                    end
                    Y_miniBatch = Y(:,start:stop);
                    
                    [A1, memory1] = netSiva.Nets{1}.forward(netSiva.Nets{1}, X1_miniBatch, m_batch);
                    [A2, memory2] = netSiva.Nets{2}.forward(netSiva.Nets{2}, X2_miniBatch, m_batch);
                    
                    [A3, memory3] = netSiva.Nets{3}.forward(netSiva.Nets{3}, ...
                                                [A1{netSiva.Nets{1}.no_ofLayer};A2{netSiva.Nets{2}.no_ofLayer}] , m_batch);
                    

                    
                    [dLdX3,grads3] = netSiva.Nets{3}.backward(netSiva.Nets{3},...
                        [A1{netSiva.Nets{1}.no_ofLayer};A2{netSiva.Nets{2}.no_ofLayer}],A3, Y_miniBatch, memory3);                     
                    [dLdX2,grads2] = netSiva.Nets{2}.backward(netSiva.Nets{2},...
                        X2_miniBatch,A2, dLdX3{1}(end - size(A2{end},1)+1:end,:), memory2);
                    [dLdX1,grads1] = netSiva.Nets{1}.backward(netSiva.Nets{1},...
                        X1_miniBatch,A1, dLdX3{1}(1:size(A1{end},1),:), memory1);
                         
                    
                    
                    netSiva.Nets{1} = netSiva.Nets{1}.updateWeights(netSiva.Nets{1}, grads1);
                    netSiva.Nets{2} = netSiva.Nets{2}.updateWeights(netSiva.Nets{2}, grads2);
                    netSiva.Nets{3} = netSiva.Nets{3}.updateWeights(netSiva.Nets{3}, grads3);
                    
                end

                A = predictMulitinetClasses(netSiva, X1, X2);
                [~,predictions] = max(A);
                [~,labels] = max(Y);
                TrainingAccuracy = sum((predictions==labels))/length(labels)*100.0;
                                
                cost = computeMultiClassLoss(Y, A); 
                display(['Epoch: ', num2str(i), '      Accuracy : ', num2str(TrainingAccuracy),'       Training cost: ', num2str(cost),'     df: ', num2str(costprev-cost)])
                costprev = cost;
                
                if i==1
                    costX = cost;
                    accX = TrainingAccuracy;
                else
                    costX = [costX,cost];
                    accX = [accX, TrainingAccuracy];
                end
                
                if (i-500 > 0)
                      startSpot = i-500;
                end
                
                figure(1);
                subplot1 = subplot(2,1,1);
                plot(accX, 'r-');
                ylim(subplot1,[0 100]);
                xlim([startSpot, (i+50)]);
                title(['Accuracy : ', num2str(TrainingAccuracy), '%']);
                grid on;
                subplot2 = subplot(2,1,2);
                plot(costX, 'b-');
                title(['Computed Multi Class Loss : ', num2str(cost)]);
                  
                maxcost = max(maxcost,cost);
                if maxcost == 0
                    maxcost =1;
                end
                ylim(subplot2,[0 maxcost]);
                xlim([ startSpot, (i+50)]);
                %axis([ startSpot, (i+50), 0 , maxcost]);
                grid
                drawnow;  
                pause(step);
                
                figure(2);
                %montage({netSiva.Nets{1, 1}.Layers{1,2}.Wf, netSiva.Nets{1, 1}.Layers{1,2}.Wi, netSiva.Nets{1, 1}.Layers{1,2}.Wc, netSiva.Nets{1, 1}.Layers{1,2}.Wo, netSiva.Nets{1, 1}.Layers{1,2}.Wy, netSiva.Nets{1, 2}.Layers{1,2}.W, netSiva.Nets{1, 3}.Layers{1,2}.W, netSiva.Nets{1, 3}.Layers{1,3}.W});
                %montage({netSiva.Nets{1, 1}.Layers{1,2}.W, netSiva.Nets{1, 2}.Layers{1,2}.W, netSiva.Nets{1, 2}.Layers{1,5}.W, netSiva.Nets{1, 3}.Layers{1,2}.W});
                montage({netSiva.Nets{1, 1}.Layers{1,2}.W, netSiva.Nets{1, 2}.Layers{1,2}.b, netSiva.Nets{1, 2}.Layers{1,3}.W, netSiva.Nets{1, 3}.Layers{1,2}.W});
                figure(3); 
                plot(1:252, netSiva.Nets{1, 2}.Layers{1, 2}.b(end:-1:1))
                hold on;

                if cost<1e-06   % Threshold to stop learning
                    break;
                elseif TrainingAccuracy > 99.5
                    break;
                end
                
                clear A;

            end
            display("Done...!");
       end
       
    end

end
    