clear;
tic
load('L_wordnet.mat');
load('groundFitData.mat');
load('knowledgeMatrix.mat'); 
p1=0.014;alpha=4;
[N,M] = size(knowledgeMatrix);
proMatrix = zeros(N,M);
for i = 1:N
    for j = 1:M
        if knowledgeMatrix(i,j)>0 && i~=j
            p=knowledgeMatrix(i,j)*alpha/(1+knowledgeMatrix(i,j)*alpha);
%            p=alpha/(1+alpha);
            proMatrix(i,j) = log(p/(1-p));
            proMatrix(j,i) = log(p1/(1-p1));
        end
    end
end
model = crowd_model(L_wordnet, groundFitData);


result_SEEK_lnr_norm = SEEK_lnr_norm(model,proMatrix)
result_MWK = MajorityWithKnowledge(model,knowledgeMatrix)
result_MWW = MajorityWithWeight(model)
toc