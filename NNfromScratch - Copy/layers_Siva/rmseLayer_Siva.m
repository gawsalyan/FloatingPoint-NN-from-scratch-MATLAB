classdef rmseLayer_Siva
    
    properties
      Name = 'FC_Default';
      No_HiddenNodes = 1;
      Learning_Rate = 1;
      Weight_Factor = 1;
      beta = 0.9; % for minibatch gradient descent method, 
      b;
      V_dLdb;
      InputSize;
      OutputSize;
      miniBatchSize;
      local_miniBatchSize;
    end
    
    methods
        
      function obj = rmseLayer_Siva(name)
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
          obj.b = zeros(obj.InputSize,1) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.V_dLdb = zeros(obj.InputSize,1);
          obj.OutputSize = obj.InputSize;
      end
      
      function A = predict(obj, X)
         Z = X - obj.b;
         A = Z.^2;
      end
      
      function [A, memory] = forward(obj,X, m_batch)
        memory = containers.Map;
        Z = X - obj.b;
        A = Z.^2;
        memory('Z') = Z; 
        memory('A') = A;
        memory('X') = X;
        memory('m_Batch') = m_batch;
      end
      
      function [dLdX,grads] = backward(obj,X,A,dLdA, memory)
         dLdZ = real(sqrt(dLdA));
         dLdb = -sign(memory('X')).*((1/memory('m_Batch')) * sum(dLdZ,2));      % n_h * 1
         dLdX = dLdZ;
         grads = containers.Map;
         grads('dLdb') = dLdb;
      end
      
      function obj = updateLayer(obj, grads)
         obj.V_dLdb = obj.beta * obj.V_dLdb + (1 - obj.beta) * grads('dLdb');
         obj.b = obj.b - obj.Learning_Rate * obj.V_dLdb;
      end
      
      function out  = calculatecontribution(obj,in)  
      end
      
    end
    
end