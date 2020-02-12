close all;clear all;clc; 
dataset = load('iris.txt'); 
x = dataset(:,4:end); 
xmax = max(x); 
xmin = min(x);
Xnorm = (x-xmin)./(xmax-xmin); 
T = dataset(:,1:3);
I = randperm(150); 
xTrain = Xnorm(I(1:50),:);  
xTest = Xnorm(I(51:end),:); 
tTrain = T(I(1:50),:); 
tTest = T(I(51:end),:); 
clear X T tic;
%Training phase 
dim = size(xTrain,2); 
hidden_node = 30; 
input_weight = -1+2*rand(dim,hidden_node); 
bias = -1+2*rand(1,hidden_node); 
hidden_layer = 1./(1+exp(-xTrain*input_weight+repmat(bias,size(xTrain,1),1))); 
output_weight = pinv(hidden_layer)*tTrain; 
output_train = hidden_layer*output_weight; 
%Test phase 
hidden_layer = 1./(1+exp(-xTest*input_weight+repmat(bias,size(xTest,1),1))); 
output_test = hidden_layer*output_weight; 
error_of_ELM = mse(tTrain-output_train) 
Y = output_train; 
%Performance of Traning
[tmp,Index1] = max(Y,[],2); 
[tmp,Index2] = max(tTrain,[],2); 
fprintf('Training acc.: %f \n',mean(mean(Index1 == Index2))*100); 
Y = output_test; 
% Performance of Testing 
[tmp,Index1] = max(Y,[],2); 
[tmp,Index2] = max(tTest,[],2); 
fprintf('Testing acc.: %f \n',mean(mean(Index1 == Index2))*100);
tic;

data = load('iris.txt'); 
X = mapminmax(data(:,4:end)',0,1)'; 
T = data(:,1:3); 
I = randperm(150); 
xTrain = X(I(1:50),:); 
tTrain = T(I(1:50),:); 
xTest = X(I(51:end),:); 
tTest = T(I(51:end),:); 
clear X T tic;
n = 0.01; 
L = 30;
wi = rands(size(xTrain,2),L); 
bi = rands(1,L);
wo = rands(L,size(tTrain,2)); 
bo = rands(1,size(tTrain,2)); 
E=[]; 
for k = 1:1000
    for i = 1:size(xTrain,1) 
        H = logsig(xTrain(i,:)*wi+bi);
        Y = logsig(H*wo+bo); 
        e = tTrain(i,:)-Y; 
        dy = e .* Y.*(1-Y); 
        dH = H.*(1-H) .* (dy*wo');
        wo = wo + n * H'*dy; 
        bo = bo + n * dy;
        wi = wi + n * xTrain(i,:)'*dH; 
        bi = bi + n * dH; 
    end
        H = logsig(xTrain *wi + repmat(bi,size(xTrain,1),1)); 
        Y = logsig(H*wo + repmat(bo,size(xTrain,1),1)); 
        subplot(1,2,1); 
        E(k) = mse(tTrain-Y); 
    plot(E); 
    title('MLP-BP Training'); 
    hold off
    xlabel('Iteration (n)');
    ylabel('MSE');
    subplot(1,2,2);
    test(k) = error_of_ELM; 
    plot(test); 
    title('ELM Training'); 
    hold off
    xlabel('Iteration (n) '); 
    ylabel('ELM'); 
    drawnow;
end
H = logsig(xTrain*wi+repmat(bi,size(xTrain,1),1)); 
Y = logsig(H*wo+repmat(bo,size(xTrain,1),1));
error_of_MLP_BP = E(1000) 
[tmp,Index1] = max(Y,[],2); 
[tmp,Index2] = max(tTrain,[],2); 
fprintf('Training acc : %f\n',mean(mean(Index1 == Index2))*100);
H = logsig(xTest*wi + repmat(bi,size(xTest,1),1)); 
Y = logsig(H*wo + repmat(bo,size(xTest,1),1));
[tmp,Index1]=max(Y,[],2); 
[tmp,Index2]=max(tTest,[],2); 
fprintf('Testing acc : %f\n',mean(mean(Index1==Index2))*100);
tic;