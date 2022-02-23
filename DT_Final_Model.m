clc; close all; clear all;

%% 
% Load the model and test set
load('DT_Final_Model.mat')

%% Testing DT
% Predicted class and posterior probabilities of each class
% Measure test time
tic
[label, score_DT] = predict(Model_DT, Test_X);
toc

%% Confusion matrix
cm = confusionchart(table2cell(Test_y),label)
cm.RowSummary = 'row-normalized';

cm.Title = 'Foetal Health Classification using DT (Testing)';

%% Accuracy, recall, precision and F1 score
% Define the TP, TN, FP and FN
[confMat,order] = confusionmat(table2cell(Test_y),label);
confMat = flip(confMat, 1);
confMat = flip(confMat, 2);
TP = confMat(1, 1); 
TN = confMat(2, 2);
FP = confMat(2, 1);
FN = confMat(1, 2);

% Calculate the accuracy, recall, precision and F1 score
acc = (TP+TN)/(TP+TN+FP+FN);
recall = TP/(TP+FN);
precision = TP/(TP+FP);
F1 = 2*recall*precision/(recall+precision);

table_result = table(acc',recall',precision',F1', 'VariableNames', {'Accuracy','Recall','Precision','F_1'});
display(table_result)

%% ROC curve
% Convert 'abnormal' and 'normal' to 1 and 0 respectively
true_class = table2array(Test_y) =="abnormal";
true_class = double(true_class);

% Plot ROC
[X Y T AUC] = perfcurve(true_class, score_DT(:, 1), 1);

figure;
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC curve for DT');