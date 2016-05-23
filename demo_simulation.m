clear;
Ntask=100;
Nworker=100;
redun=10;
alpha=0.05;
alpha1=0.05;
alpha2=0.75;
confusion=[1-alpha1 alpha2
            alpha1 1-alpha2];
L=zeros(Ntask,Nworker);
ability=(rand(1,Nworker)-0.5)*0.5;
%%≥ı ºªØground
ground=zeros(1,Ntask);
for j=1:Ntask
    if rand(1)<alpha
        ground(j)=2*j-1;
    else
        ground(j)=2*j;
    end    
end

for j=1:Ntask
    r=randperm(Nworker);
    if ground(j)==2*j-1
        for k=1:redun
            if rand(1)*(confusion(1,1)+confusion(2,1)+ability(r(k)))<confusion(1,1)+ability(r(k))
                L(j,r(k))=2*j-1;
            else
                L(j,r(k))=2*j;
            end
        end
    elseif ground(j)==2*j
        for k=1:redun
            if rand(1)*(confusion(1,2)+confusion(2,2)+ability(r(k)))<confusion(1,2)
                L(j,r(k))=2*j-1;
            else
                L(j,r(k))=2*j;
            end
        end
    else
        error='error'
    end
end
knowledgeMatrix=zeros(2*Ntask,2*Ntask);
for j=1:Ntask
    knowledgeMatrix(2*j-1,2*j-1)=confusion(1,1);
    knowledgeMatrix(2*j-1,2*j)=confusion(1,2);
    knowledgeMatrix(2*j,2*j-1)=confusion(2,1);
    knowledgeMatrix(2*j,2*j)=confusion(2,2);
end
model=crowd_model(L,ground);
result_Pro=ProbWithKnowledge(model,knowledgeMatrix)
%result_MWK=MajorityWithKnowledge(model,knowledgeMatrix)
%result_MV=MajorityVote(model) 
%result_MWW=MajorityWithWeight(model)



        