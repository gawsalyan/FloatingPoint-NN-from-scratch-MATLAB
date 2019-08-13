classdef reluLayer_Siva
    
    properties
      Name = 'Pool_Default';
      No_HiddenNodes = 1;
      Learning_Rate = 1;
      Weight_Factor = 1;
      beta = 0.9; % for minibatch gradient descent method, 
      
      InputSize;
      OutputSize;
      miniBatchSize;
      local_miniBatchSize;
    end
    
    methods
        
      function obj = reluLayer_Siva(name)
           obj.Name = name;
      end
      
    end
    
    methods(Static)
        
      function obj = initLayer(obj, in, options)        
          obj.Learning_Rate = options('Learning_Rate');
          obj.Weight_Factor = options('Weight_Factor');
          obj.beta = options('beta');
          obj.InputSize = in.OutputSize;
          
          obj.OutputSize = obj.InputSize;
      end
      
      function obj = setLayerFilter(obj, FilterIndex, FilterValue)
              
      end
      
      function A = predict(obj, X)
        A = relu(obj,X);        
      end
      
      function [A, memory] = forward(obj,X, m_batch)
        memory = containers.Map;
        A = relu(obj,X);
        memory('A') = A;
        memory('X') = X;
        memory('m_Batch') = m_batch;
      end
      
      function [dLdX,grads] = backward(obj,X,A,dLdA, memory) 
         dLdX =  (X>0) .* dLdA;
         gradsCNN = containers.Map;       
         grads = clipGradients(gradsCNN);
      end
      
      function obj = updateLayer(obj, grads)

      end
           
    end
    
end



function A_out = relu(obj,X)
        A_out = X .* (X>0);
end
