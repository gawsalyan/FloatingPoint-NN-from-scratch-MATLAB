function  gradients = lstm_cell_backward(dy_next, da_next, dc_next, cache)
   
%     Implement the backward pass for the LSTM-cell (single time-step).
% 
%     Arguments:
%     da_next -- Gradients of next hidden state, of shape (n_a, m)
%     dc_next -- Gradients of next cell state, of shape (n_a, m)
%     cache -- cache storing information from the forward pass
% 
%     Returns:
%     gradients -- python dictionary containing:
%                         dxt -- Gradient of input data at time-step t, of shape (n_x, m)
%                         da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
%                         dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
%                         dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
%                         dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
%                         dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
%                         dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
%                         dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
%                         dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
%                         dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
%                         dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)


    
    % Retrieve information from "cache"
    [a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, Wf, bf, Wi, bi, Wc, bc, Wo, bo, Wy, by] = cache{:};
    
    % Retrieve dimensions from xt's and a_next's shape 
    [n_x, m] = size(xt);
    [n_a, m]= size(a_next);
    
    
    % Compute gates related derivatives, you can find their values can be found by looking carefully at equations (7) to (10)
    da_next = Wy'*dy_next + da_next;
    dot = da_next .* tanh(c_next).* ot.* (1 - ot);
    dcct_temp = (da_next .* ot .* (1 - tanh(c_next).^2) + dc_next);
    dcct = dcct_temp .* it .* (1 - cct.^2);
    dit = dcct_temp .* cct .* (1 - it) .* it;
    dft = dcct_temp .* c_prev .* ft .* (1 - ft);

    % Compute parameters related derivatives. Use equations (11)-(14) (?8 lines    
    dWf = dft*[a_prev', xt'];
    dWi = dit*[a_prev', xt'];
    dWc = dcct*[a_prev', xt'];
    dWo = dot*[a_prev', xt'];
    dWy = dy_next*a_next';   %%added need to test
    dbf = sum(dft, 2);
    dbi = sum(dit, 2);
    dbc = sum(dcct, 2);
    dbo = sum(dot, 2);
    dby = sum(dy_next,2);   %%added need to test

    % Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17)    
    da_prev = Wf(:, 1:n_a)'*dft + Wc(:, 1:n_a)'*dcct + Wi(:, 1:n_a)'*dit + Wo(:, 1:n_a)'*dot;
    dc_prev = (da_next .* ot .* (1 - tanh(c_next).^2) + dc_next) .* ft;
    dxt = Wf(:, n_a+1:end)'*dft + Wc(:, n_a+1:end)'*dcct + Wi(:, n_a+1:end)'*dit + Wo(:, n_a+1:end)'*dot;
        
    % Save gradients in dictionary
    gradients = { dxt, da_prev, dc_prev, dWf, dbf,  dWi, dbi,...
                dWc, dbc,  dWo, dbo, dWy, dby};

end