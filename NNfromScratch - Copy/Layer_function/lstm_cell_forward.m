function [a_next, c_next, yt_pred, cache] = lstm_cell_forward(xt, a_prev, c_prev, Wf, bf, Wi, bi, Wc, bc, Wo, bo, Wy, by)
   
% Implement a single forward step of the LSTM-cell as described in Figure (4)
% 
%     Arguments:
%     xt -- your input data at timestep "t", numpy array of shape (n_x, m).
%     a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
%     c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
%     parameters -- python dictionary containing:
%                         Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
%                         bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
%                         Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
%                         bi -- Bias of the update gate, numpy array of shape (n_a, 1)
%                         Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
%                         bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
%                         Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
%                         bo -- Bias of the output gate, numpy array of shape (n_a, 1)
%                         Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
%                         by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
%                         
%     Returns:
%     a_next -- next hidden state, of shape (n_a, m)
%     c_next -- next memory state, of shape (n_a, m)
%     yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
%     cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
%     
%     Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
%           c stands for the memory value
    
    % Retrieve dimensions from shapes of xt and Wy
    [n_x, m] = size(xt);
    [n_y, n_a] = size(Wy);

    
    % Concatenate a_prev and xt 
    concat = zeros(n_a + n_x, m,'like',a_prev);
    concat(1: n_a, :) = a_prev; 
    concat(n_a+1:end, :) = xt;

    % Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (?6 lines)   
    ft = sigmoid(Wf*concat + bf);
    it = sigmoid(Wi*concat + bi);
    cct = tanh(Wc*concat + bc);
    c_next = ft .* c_prev + it .* cct;
    ot = sigmoid(Wo*concat + bo);
    a_next = ot .* tanh(c_next);
    
    % Compute prediction of the LSTM cell (?1 line)
    yt_pred = softmax(Wy*a_next + by);
    
    % store values needed for backward propagation in cache
    cache = {a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, Wf, bf, Wi, bi, Wc, bc, Wo, bo, Wy, by};
    
end