function model = train(X, Y, parameter)

[~, num_dim] = size(X);
[~, num_class] = size(Y);

XTX = X'*X;
XTY = X'*Y;
rho1 = 2;
W   = (XTX + rho1*eye(num_dim)) \ (XTY);
% W = eye(num_dim, num_class);
W_1 = W;
C = zeros(num_class,num_class);
% C = eye(num_class, num_class);
% C = 1-pdist2(Y', Y', 'cosine');
Q = C;
Lambda = C - Q;
[N] = local_neighbor(X, parameter.num_K, parameter.paraDc);

bk = 1; 
bk_1 = 1; 
iter = 1;
obj_loss = zeros(1,parameter.maxIter);

% Lip
Lip = sqrt(2*(norm(XTX)^2 + parameter.lambda3*norm((X-N*X)'*(X-N*X))^2));
% neighbor point
[N] = local_neighbor(X, parameter.num_K, parameter.paraDc);
while iter <= parameter.maxIter
    
    %% update W
    W_k  = W + (bk_1 - 1)/bk * (W - W_1);
    Gw_x_k = W_k - 1/Lip * gradientOfW(X,Y,N,W,C,parameter.lambda3);
    W_1  = W;
    W    = softthres(Gw_x_k,parameter.lambda5/Lip);
    bk_1   = bk;
    bk     = (1 + sqrt(4*bk^2 + 1))/2;
    
    %% update C
    paramC_left = (1+parameter.lambda1)*Y'*Y + parameter.mu*eye(num_class, num_class) + parameter.lambda2*(Y - N*Y)'*(Y - N*Y);
    paramC_right = parameter.mu*Q - Lambda + Y'*X*W + parameter.lambda1*Y'*Y;
    C = paramC_left\paramC_right;
    
    %% update Q
    [U, Sigma, V] = svd(C + Lambda/parameter.mu,'econ');
    Q = U * softthres(Sigma,(parameter.lambda4)/parameter.mu) * V';
    
    %% update Lambda
    % Lmabda
    Lambda = Lambda + parameter.mu*(C - Q);
    % mu
    parameter.mu = min(parameter.maxMu, parameter.mu*parameter.rho);
    
    %% stop conditions
    loss = 0.5*norm((X*W-Y*C),'fro')^2 + 0.5*parameter.lambda1*norm(Y*C-Y,'fro')^2 + 0.5*parameter.lambda2*norm(Y*C - N*Y*C,'fro')^2;
    loss = loss + parameter.lambda3*norm(X*W - N*X*W,'fro')^2 + 0.5*parameter.lambda4*trace(sqrt(C'*C)) + parameter.lambda5*norm(W,1);
    if loss < parameter.minLoss
        break
    end
    obj_loss(1,iter) = loss;
    
    if norm(C - Q, 'inf') < parameter.epsilon
        break;
    end
    iter=iter+1;
end
model.W = W;
model.Q = Q;
model.C = C;
model.Lambda = Lambda;
model.obj_loss = obj_loss;
model.iter = iter;
end

%% soft thresholding operator
function Ws = softthres(W,lambda)
Ws = max(W-lambda,0) - max(-W-lambda,0);
end
%% gradient W
function gradient = gradientOfW(X,Y,N,W,C,lambda3)
gradient =X'*(X*W - Y*C);
gradient = gradient + lambda3*(X-N*X)'*(X-N*X)*W;
end