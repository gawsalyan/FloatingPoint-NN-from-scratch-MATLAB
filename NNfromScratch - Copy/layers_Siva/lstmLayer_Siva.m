classdef lstmLayer_Siva
    
    properties
      Name = 'LSTM_Default';
      No_HiddenNodes = 1;
      Learning_Rate = 1;
      Weight_Factor = 1;
      beta = 0.9; % for minibatch gradient descent method, 
      InputSize;
      OutputSize;
      miniBatchSize;
      local_miniBatchSize;
      
      n_a = 1;   % No of features in Activation vector (A and C)
      n_f = 1;   % No of elements in input X vector i.e. it will represent the onehotvector 
      n_x = 1;   % No of features in input X vector
      n_y = 1;   % No of feature in output Y vector
      Ts = 10;
      
      Wf; Wi; Wc; Wo; Wy;
      bf; bi; bc; bo; by;
      
      a0;
      dy; da; dc;
      
      V_dLdWf; V_dLdWi; V_dLdWc; V_dLdWo; V_dLdWy;
      V_dLdbf; V_dLdbi; V_dLdbc; V_dLdbo; V_dLdby;
      
    end
    
    methods
        
      function obj = lstmLayer_Siva(n_a,n_x, n_y,name)
           obj.No_HiddenNodes = n_a;
           obj.n_a = n_a;
           obj.n_x = n_x;
           obj.n_y = n_y;
           %obj.Ts = Ts;
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
          obj.InputSize = in.OutputSize;
          obj.Ts = obj.InputSize;
          
          
          obj.Wf = rand(obj.n_a, obj.n_a + obj.n_x) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.bf = zeros(obj.n_a, obj.n_f) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.Wi = rand(obj.n_a, obj.n_a + obj.n_x) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.bi = zeros(obj.n_a, obj.n_f) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.Wc = rand(obj.n_a, obj.n_a + obj.n_x) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.bc = zeros(obj.n_a, obj.n_f) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.Wo = rand(obj.n_a, obj.n_a + obj.n_x) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.bo = zeros(obj.n_a, obj.n_f) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.Wy = rand(obj.n_y, obj.n_a) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          obj.by = zeros(obj.n_y, obj.n_f) * obj.Weight_Factor * sqrt(1/obj.InputSize);
          
          obj.V_dLdWf = zeros(obj.n_a, obj.n_a + obj.n_x);
          obj.V_dLdbf = zeros(obj.n_a, obj.n_f);
          obj.V_dLdWi = zeros(obj.n_a, obj.n_a + obj.n_x);
          obj.V_dLdbi = zeros(obj.n_a, obj.n_f);
          obj.V_dLdWc = zeros(obj.n_a, obj.n_a + obj.n_x);
          obj.V_dLdbc = zeros(obj.n_a, obj.n_f);
          obj.V_dLdWo = zeros(obj.n_a, obj.n_a + obj.n_x);
          obj.V_dLdbo = zeros(obj.n_a, obj.n_f);
          obj.V_dLdWy = zeros(obj.n_y, obj.n_a);
          obj.V_dLdby = zeros(obj.n_y, obj.n_f);
          
          obj.a0 = zeros(obj.n_a,obj.n_f);
          obj.da = zeros(obj.n_a,obj.n_f); 
          obj.dc = zeros(obj.n_a,obj.n_f);
          
          obj.OutputSize = obj.n_y * obj.n_f * obj.Ts;  %%need attentom
      end
      
      function A = predict(obj, X)  
        sizeX = size(X);
        if size(sizeX,2) > 2
            m_batch = sizeX(end);
        else
            m_batch = 1;
        end
        
        x = reshape(X,[obj.n_x, obj.n_f, obj.Ts, m_batch]);
        y = zeros(obj.n_y*obj.n_f, obj.Ts ,m_batch);
            for i = 1:m_batch
                xt_lstm  = reshape(x(:,:,:,i),[obj.n_x, obj.n_f, obj.Ts]);
                params = createLSTMPara(obj);
                [a, yloc, c, cachesloc] = lstm_forward(xt_lstm, obj.a0, params);
                y(:,:,i) = reshape(yloc,[obj.n_y*obj.n_f, obj.Ts, 1]);
            end                        
         A = reshape(y,[obj.n_y*obj.n_f*obj.Ts, m_batch]);
      end
      
      function [A, memory] = forward(obj,X, m_batch)   
       x = reshape(X,[obj.n_x, obj.n_f, obj.Ts, m_batch]);
       y = zeros(obj.n_y*obj.n_f, obj.Ts ,m_batch);
            for i = 1:m_batch
                xt_lstm  = reshape(x(:,:,:,i),[obj.n_x, obj.n_f, obj.Ts]);
                params = createLSTMPara(obj);
                [a, yloc, c, cachesloc] = lstm_forward(xt_lstm, obj.a0, params);
                y(:,:,i) = reshape(yloc,[obj.n_y*obj.n_f, obj.Ts, 1]);
                memory{i} = cachesloc;
            end
         A = reshape(y,[obj.n_y*obj.n_f*obj.Ts, m_batch]);
      end
      
      function [dLdX,grads] = backward(obj,X,A,dLdA, memory) 
         sizeA = size(A);
         m_batch = sizeA(end);
         dy_batch = reshape(dLdA,[obj.n_y, obj.n_f, obj.Ts, m_batch]);
         
         dy = reshape(dy_batch(:,:,:,1),[obj.n_y,obj.n_f,obj.Ts]);
         gradLSTM = lstm_backward(dy,obj.da,obj.dc, memory{1});
         gradLSTM('dLdX')  = reshape(gradLSTM('dLdX'),[obj.n_x, obj.Ts]);
             for i = 2:m_batch
                dy = reshape(dy_batch(:,:,:,i),[obj.n_y,obj.n_f,obj.Ts]);
                gradients = lstm_backward(dy,obj.da,obj.dc, memory{i});
                gradLSTM('dLdX') = [gradLSTM('dLdX')  reshape(gradients('dLdX'),[obj.n_x, obj.Ts])];
                gradLSTM('dLda0') = gradLSTM('dLda0') + gradients('dLda0');
                gradLSTM('dLdWf') = gradLSTM('dLdWf') + gradients('dLdWf');  
                gradLSTM('dLdbf') = gradLSTM('dLdbf') + gradients('dLdbf');
                gradLSTM('dLdWi') = gradLSTM('dLdWi') + gradients('dLdWi'); 
                gradLSTM('dLdbi') = gradLSTM('dLdbi') + gradients('dLdbi');
                gradLSTM('dLdWc') = gradLSTM('dLdWc') + gradients('dLdWc');
                gradLSTM('dLdbc') = gradLSTM('dLdbc') + gradients('dLdbc');
                gradLSTM('dLdWo') = gradLSTM('dLdWo') + gradients('dLdWo');  
                gradLSTM('dLdbo') = gradLSTM('dLdbo') + gradients('dLdbo');
                gradLSTM('dLdWy') = gradLSTM('dLdWy') + gradients('dLdWy');  
                gradLSTM('dLdby') = gradLSTM('dLdby') + gradients('dLdby');
            end
    
            gradLSTM('dLdX') =  gradLSTM('dLdX');
            gradLSTM('dLda0') = (1/m_batch)* gradLSTM('dLda0');
            gradLSTM('dLdWf') = (1/m_batch)* gradLSTM('dLdWf');  
            gradLSTM('dLdbf') = (1/m_batch)* gradLSTM('dLdbf');
            gradLSTM('dLdWi') = (1/m_batch)* gradLSTM('dLdWi');
            gradLSTM('dLdbi') = (1/m_batch)* gradLSTM('dLdbi');
            gradLSTM('dLdWc') = (1/m_batch)* gradLSTM('dLdWc');
            gradLSTM('dLdbc') = (1/m_batch)* gradLSTM('dLdbc');
            gradLSTM('dLdWo') = (1/m_batch)* gradLSTM('dLdWo');  
            gradLSTM('dLdbo') = (1/m_batch)* gradLSTM('dLdbo');
            gradLSTM('dLdWy') = (1/m_batch)* gradLSTM('dLdWy');  
            gradLSTM('dLdby') = (1/m_batch)* gradLSTM('dLdby');
            
            
            grads = clipGradients(gradLSTM);
            dLdX = gradLSTM('dLdX');  % no clipping on dLdX
          
      end
      
      function obj = updateLayer(obj, grads)
         obj.V_dLdWf = obj.beta * obj.V_dLdWf + (1 - obj.beta) * grads('dLdWf');
         obj.V_dLdbf = obj.beta * obj.V_dLdbf + (1 - obj.beta) * grads('dLdbf');
         obj.V_dLdWi = obj.beta * obj.V_dLdWi + (1 - obj.beta) * grads('dLdWi');
         obj.V_dLdbi = obj.beta * obj.V_dLdbi + (1 - obj.beta) * grads('dLdbi');
         obj.V_dLdWc = obj.beta * obj.V_dLdWc + (1 - obj.beta) * grads('dLdWc');
         obj.V_dLdbc = obj.beta * obj.V_dLdbc + (1 - obj.beta) * grads('dLdbc');
         obj.V_dLdWo = obj.beta * obj.V_dLdWo + (1 - obj.beta) * grads('dLdWo');
         obj.V_dLdbo = obj.beta * obj.V_dLdbo + (1 - obj.beta) * grads('dLdbo');
         obj.V_dLdWy = obj.beta * obj.V_dLdWy + (1 - obj.beta) * grads('dLdWy');
         obj.V_dLdby = obj.beta * obj.V_dLdby + (1 - obj.beta) * grads('dLdby');
         
         obj.Wf = obj.Wf - obj.Learning_Rate * obj.V_dLdWf;
         obj.bf = obj.bf - obj.Learning_Rate * obj.V_dLdbf;
         obj.Wi = obj.Wi - obj.Learning_Rate * obj.V_dLdWi;
         obj.bi = obj.bi - obj.Learning_Rate * obj.V_dLdbi;
         obj.Wc = obj.Wc - obj.Learning_Rate * obj.V_dLdWc;
         obj.bc = obj.bc - obj.Learning_Rate * obj.V_dLdbc;
         obj.Wo = obj.Wo - obj.Learning_Rate * obj.V_dLdWo;
         obj.bo = obj.bo - obj.Learning_Rate * obj.V_dLdbo;
         obj.Wy = obj.Wy - obj.Learning_Rate * obj.V_dLdWy;
         obj.by = obj.by - obj.Learning_Rate * obj.V_dLdby;
         
      end
      
    end
    
end


function params = createLSTMPara(obj)
%Temporary function to map the net class properties to layer function
%parameters, in future layer funtions need to be changed such that it
%straight away taking in the net properties
        params = containers.Map;
                params('Wf') = obj.Wf;  params('bf') = obj.bf; 
                params('Wi') = obj.Wi;  params('bi') = obj.bi; 
                params('Wc') = obj.Wc;  params('bc') = obj.bc; 
                params('Wo') = obj.Wo;  params('bo') = obj.bo; 
                params('Wy') = obj.Wy;  params('by') = obj.by; 
end