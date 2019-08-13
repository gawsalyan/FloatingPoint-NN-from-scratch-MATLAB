classdef seluLayer_Siva
    
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
      
        alpha = 1.6732632423543772848170429916717;
        scale = 1.0507009873554804934193349852946;
    end
    
    methods
        
      function obj = seluLayer_Siva(name)
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
        A = selu(obj,X);        
      end
      
      function [A, memory] = forward(obj,X, m_batch)
        memory = containers.Map;
        A = selu(obj,X);
        memory('A') = A;
        memory('X') = X;
        memory('m_Batch') = m_batch;
      end
      
      function [dLdX,grads] = backward(obj,X,A,dLdA, memory) 
         dLdX =  1/obj.scale* dLdA .* ((X>=0) + (X<0).*(obj.alpha * exp(X)));
         gradsCNN = containers.Map;       
         grads = clipGradients(gradsCNN);
      end
      
      function obj = updateLayer(obj, grads)

      end
           
    end
    
end



function A_out = selu(obj,X)
            A_out = obj.scale * ((X).*(X>=0) + (obj.alpha * exp(X) - obj.alpha).*(X<0));
end
