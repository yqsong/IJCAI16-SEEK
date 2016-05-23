clear;
tic
load('L_wordnet.mat');
load('groundFitData.mat');
load('knowledgeMatrix.mat'); 
N= size(knowledgeMatrix,1);
%knowledgeMatrix=ones(N);
model = crowd_model(L_wordnet, groundFitData);

result_Pro = ProbWithKnowledge(model,knowledgeMatrix+0.05)
%result_MV = MajorityVote(model)
%result_MWK = MajorityWithKnowledge(model,knowledgeMatrix)
%result_MWW = MajorityWithWeight(model)
toc