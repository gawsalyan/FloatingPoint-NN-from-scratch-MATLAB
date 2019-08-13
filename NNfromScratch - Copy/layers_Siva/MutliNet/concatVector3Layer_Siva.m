classdef concatVector3Layer_Siva
    
    properties
      Name = 'input';
      OutputSize;
      InputSize;
    end
    
    methods
        
      function obj = concatVector3Layer_Siva(name)
           obj.Name = name;
      end
      
    end
    
    methods(Static)
        
      function A = predict(obj, X)
         A = X;
      end
      
      function obj = initLayer(obj, in1, in2,in3, options)        
          obj.OutputSize = in1.OutputSize + in2.OutputSize + in3.OutputSize;
          obj.InputSize = obj.OutputSize;
      end 
            
    end
    
end