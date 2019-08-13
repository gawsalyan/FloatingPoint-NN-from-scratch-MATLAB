classdef multiClassLayer_Siva
    
    properties
      Name = 'FC_Default';
      No_HiddenNodes = 1;
      Learning_Rate = 1;
      Weight_Factor = 1;
      beta = 0.9; % for minibatch gradient descent method, 
      W;
      b;
      V_dLdW;
      V_dLdb;
      InputSize;
      OutputSize;
      miniBatchSize;
      local_miniBatchSize;
    end
    
    methods
        
      function obj = multiClassLayer_Siva(n_H, name)
           obj.No_HiddenNodes = n_H;
           obj.Name = name;
      end
    
    end
    
    methods(Static)
      
      function obj = setLearningRate(obj, LR)
            obj.Learning_Rate = LR;
      end
        
      function obj = initLayer(obj, in, options)        
          obj.Learning_Rate = options('Learning_Rate');
          obj.Weight_Factor = options('Weight_Factor');
          obj.beta = options('beta');
          obj.InputSize = in.OutputSize;
          obj.W = rand(obj.No_HiddenNodes, obj.InputSize) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.b = zeros(obj.No_HiddenNodes,1);
          obj.V_dLdW = zeros(obj.No_HiddenNodes, obj.InputSize);
          obj.V_dLdb = zeros(obj.No_HiddenNodes,1);
          obj.OutputSize = obj.No_HiddenNodes;
      end
      
      function A = predict(obj, X)
         Z = obj.W*X + obj.b;
         Z(Z>100) = 100; 
         Z(Z<-100) = -100; 
         expZ = exp(Z);        
         A = expZ./sum(expZ,1);
      end
      
      function [A, memory] = forward(obj,X, m_batch)
        memory = containers.Map;
        Z = obj.W*X + obj.b;
        expZ = exp(Z);
        A = expZ./sum(expZ,1);
        memory('Z') = Z;
        memory('A') = A;
        memory('X') = X;
        memory('m_Batch') = m_batch;
      end
      
      function [dLdX,grads] = backward(obj,X,A,Y, memory)
        dLdZ = A - Y; 
        dLdW = (1/memory('m_Batch')) * dLdZ * X';
        dLdb = (1/memory('m_Batch')) * sum( dLdZ,2);  
        dLdX = obj.W' * dLdZ;
        
        gradsMC = containers.Map;
        gradsMC('dLdW') = dLdW;
        gradsMC('dLdb') = dLdb;
        grads = clipGradients(gradsMC);
      end
      
      function obj = updateLayer(obj, grads)
         
         obj.V_dLdW = obj.beta * obj.V_dLdW + (1 - obj.beta) * grads('dLdW');
         obj.V_dLdb = obj.beta * obj.V_dLdb + (1 - obj.beta) * grads('dLdb');
         obj.W = obj.W - obj.Learning_Rate * obj.V_dLdW;
         obj.b = obj.b - obj.Learning_Rate * obj.V_dLdb;
      end
      
      function out  = calculatecontribution(obj,in)  
          out = obj.W\(in - obj.b);
      end
      
    end
    
end