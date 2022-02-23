% Load the model and test set
load('kNN_Final_Model.mat')

%% Testing kNN
% Predicted class and posterior probabilities of each class
% Measure test time
tic
[label_knn, score_knn] = predict(Model_knn, Test_X_norm);
toc

%% Confusion matrix
figure();
cm = confusionchart(table2cell(Test_y),label_knn);
cm.RowSummary = 'row-normalized';
cm.Title = 'Foetal Health Classification using kNN (Testing)';

%% Accuracy, recall, precision and F1 score
% Define the TP, TN, FP and FN
[confMat,order] = confusionmat(table2cell(Test_y),label_knn);
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
[X_knn Y_knn T_knn AUC_knn] = perfcurve(true_class, score_knn(:, 1), 1);

figure;
plot(X_knn, Y_knn);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC curve for kNN');

%% Plot two ROC curve on the same graph
figure();
plot(X, Y, 'LineWidth', 2)
hold on;
plot(X_knn, Y_knn, 'LineWidth', 2)
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC curve (Testing)');
legend('Decision Tree', 'k-Nearest Neighbours')