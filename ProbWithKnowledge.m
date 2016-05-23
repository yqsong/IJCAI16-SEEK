function result_Pro=ProbWithKnowledge( model,KnowledgeMatrix)
%% parameters setting
maxIter=20;
TOL=1e-6;
global L NeibTask NeibWork LabelDomain Relation Ntask Nwork Ndom LabelTask LabelWork
Ntask = model.Ntask; 
Nwork = model.Nwork;
Ndom = model.Ndom;
NeibTask = model.NeibTask; 
NeibWork = model.NeibWork;
LabelDomain =model.LabelDomain;
Relation=KnowledgeMatrix;
L=model.L; 
LabelTask=model.LabelTask;
LabelWork=model.LabelWork;
majority=MajorityVote(model);
ans_labels=majority.ans_labels;
result_Pro.majority_anslabel=ans_labels;
if ~isempty(model.true_labels)
    result_Pro.majority_accuracy=sum(~(ans_labels-model.true_labels))/Ntask;
end

% for work_i=1:Nwork 
%     Ability(work_i)=sum(~(model.L(NeibWork{work_i},work_i)-ans_labels(NeibWork{work_i})'))/length(NeibWork{work_i});    
% end
% Ability=(Ability-mean(Ability))*10;

%% main iteration

% oldA=ones(1,Nwork);
% oldS=zeros(1,Ntask)-1.5;
% A=oldA;
% S=oldS;
% likehood=Q(A,oldA,S,oldS)
% [g_Ability,g_Simplicity]=update_diff(A,oldA,S,oldS);
% alpha=1;
% for iter =1:10
%     A=oldA+alpha*g_Ability;
%     S=oldS+alpha*g_Simplicity;
%     likehood_tem=Q(A,oldA,S,oldS)
%     alpha=alpha/2;
% end
% a=1
% for iter =1:10
%     A=oldA-alpha*g_Ability;
%     S=oldS-alpha*g_Simplicity;
%     likehood_tem=Q(A,oldA,S,oldS)
%     alpha=alpha*2;
% end
% result_Pro=likehood;

err = NaN;%初始化误差
Ability=ones(1,Nwork);
Simplicity=zeros(1,Ntask);
likehood=Q(Ability,Ability,Simplicity,Simplicity);
main_likehood=likehood
for iter = 1:maxIter  
    if err<TOL;%误差满足收敛要求
        break;
    elseif iter==maxIter
        info='iter in mainFunction reached to maxIter'
        break;
    else
        [Ability_tem,Simplicity_tem,likehood_tem]=argQ(Ability,Simplicity);
        Ability=Ability_tem;
        Simplicity=Simplicity_tem;
    end
    err=abs(likehood_tem-likehood);
    main_err=err
    likehood=likehood_tem;
    main_likehood=likehood
end
L_anslabel=ones(1,Ntask);
for task_j=1:Ntask
    maxPr=0;
    LjDomain=unique(LabelTask{task_j});
    for k=1:length(LjDomain)
        p_Lt=p_Lt_LjASj(LjDomain(k),task_j,Ability,Simplicity);
        if p_Lt>maxPr
            maxPr=p_Lt;
            L_anslabel(task_j)=LjDomain(k);
        end
    end        
end
if ~isempty(model.true_labels)
    result_Pro.Precision=sum(~(L_anslabel-model.true_labels))/Ntask;
    result_Pro.FaultLabelIndex=find(L_anslabel-model.true_labels);
end
result_Pro.anslabel=L_anslabel;
result_Pro.Simplicity=Simplicity;
result_Pro.Ability=Ability;

end

function p=p_Lij_LtAiSj(work_i,task_j,kth_label,A,S)
global L Relation  LabelTask
uniform=0;
LjDomain=unique(LabelTask{task_j});
for k=1:length(LjDomain)
    uniform=uniform+exp((A(work_i)+S(task_j))*Relation(LjDomain(k),kth_label));
end
if L(task_j,work_i)==0
    errlocal_p_Lij_LtAiSj(1)=work_i
    errlocal_p_Lij_LtAiSj(2)=task_j
end
p=exp((A(work_i)+S(task_j))*Relation(L(task_j,work_i),kth_label))/uniform;
end

function p=p_Lt_LjASj(kth_label,task_j,A,S)
%probability fo true label gived Lj, Ability, Simplicity
global NeibTask LabelTask
uniform=0;
LjDomain=unique(LabelTask{task_j});
for k2th_label=1:length(LjDomain)
    tem=1;
    for i=1:length(NeibTask{task_j})
        tem=tem*p_Lij_LtAiSj(NeibTask{task_j}(i),task_j,k2th_label,A,S);
    end
    uniform=uniform+tem;
end
tem=1;
for i=1:length(NeibTask{task_j})
    tem=tem*p_Lij_LtAiSj(NeibTask{task_j}(i),task_j,kth_label,A,S);
end
p=tem/uniform;
end

function [g_Ability,g_Simplicity]=update_diff(A,oldA,S,oldS)
%辅助函数Q的偏导
global L NeibTask NeibWork  Relation Nwork Ntask  LabelTask
%size(Relation)
g_Ability=zeros(size(oldA));
for work_i=1:Nwork
    sum=0;
    for j=1:length(NeibWork{work_i})
        task_j=NeibWork{work_i}(j);
        LjDomain=unique(LabelTask{task_j});
        for k=1:length(LjDomain)
            nume=0;
            deno=0;
            for k2=1:length(LjDomain)
                nume=nume+Relation(LjDomain(k2),LjDomain(k))*exp(Relation(LjDomain(k2),LjDomain(k))*(A(work_i)+S(task_j)));
                deno=deno+exp(Relation(LjDomain(k2),LjDomain(k))*(A(work_i)+S(task_j)));
            end
            MeanRelation=nume/deno;
            if L(task_j,work_i)==0
                errlocal_Ability(1)=work_i 
                errlocal_Ability(2)=task_j
            end
            sum=sum+p_Lt_LjASj(LjDomain(k),task_j,oldA,oldS)*(Relation(L(task_j,work_i),LjDomain(k))-MeanRelation);
        end
    end
    g_Ability(work_i)=sum;
end
g_Simplicity=zeros(size(oldS));
for task_j=1:Ntask
    sum=0;
    for i=1:length(NeibTask{task_j})
        work_i=NeibTask{task_j}(i);
        LjDomain=unique(LabelTask{task_j});
        for k=1:length(LjDomain)
            nume=0;
            deno=0;
            for k2=1:length(LjDomain)
                nume=nume+Relation(LjDomain(k2),LjDomain(k))*exp(Relation(LjDomain(k2),LjDomain(k))*(oldA(work_i)+oldS(task_j)));
                deno=deno+exp(Relation(LjDomain(k2),LjDomain(k))*(oldA(work_i)+oldS(task_j)));
            end
            MeanRelation=nume/deno;
            if L(task_j,work_i)==0%测试处于错误调用的位置
                errlocal_Simplicity(1)=work_i
                errlocal_Simplicity(2)=task_j
            end
            sum=sum+p_Lt_LjASj(LjDomain(k),task_j,oldA,oldS)*(Relation(L(task_j,work_i),LjDomain(k))-MeanRelation);
        end
    end
    g_Simplicity(task_j)=sum;
end

end

function likehood=Q(A,oldA,S,oldS)
%辅助函数
global NeibWork  Nwork LabelTask
sum=0;
for work_i=1:Nwork
    for j=1:length(NeibWork{work_i})
        task_j=NeibWork{work_i}(j);
        LjDomain=unique(LabelTask{task_j});
        for k=1:length(LjDomain)
            sum=sum+p_Lt_LjASj(LjDomain(k),task_j,oldA,oldS)*log(p_Lij_LtAiSj(work_i,task_j,k,A,S));
        end
    end
end
likehood=sum;
end

function alpha = armijo(Ability, g_Ability, Simplicity,g_Simplicity)
%未使用
%Armijo步长选择参数设定
rho=0.4;
alpha=1;
gamma=0.3;
power= 0; max_power = 3;

while power <= max_power
    new_Ability=Ability+alpha*(gamma^power)*g_Ability;
    new_Simplicity=Simplicity+alpha*(gamma^power)*g_Simplicity;
    if Q(new_Ability,new_Simplicity)>= Q(Ability,Simplicity) + rho * alpha*gamma^power * norm([Ability Simplicity])^2
        break;
    end
    power = power + 1;
end
alpha=alpha*gamma^power;
end

function [A,S,likehood]=argQ(oldA,oldS)
global Nwork Ntask
maxIter=20;
TOL=1e-6;
A=oldA;
S=oldS;
likehood=Q(A,oldA,S,oldS);
[g_Ability,g_Simplicity]=update_diff(A,oldA,S,oldS);%梯度
err=(norm(g_Ability)^2+norm(g_Simplicity)^2)/(Nwork+Ntask);
for iter =1:maxIter
    if err<TOL;%误差满足收敛要求
        break;
    elseif iter>=maxIter
        info='iter in argQ reached to maxIter'
        break;
    else
        alpha=1;
        for iter2=1:maxIter
            if iter2==maxIter
                info='iter2 in argQ reached to maxIter'
                iter=maxIter;
                break;
            else
                A_tem=A+g_Ability*alpha;
                S_tem=S+g_Simplicity*alpha;
                likehood_tem = Q(A_tem,oldA,S_tem,oldS)
                if likehood_tem > likehood
                    A = A_tem;
                    S = S_tem;
                    iter
                    likehood =likehood_tem
                    break;
                else
                    alpha = alpha/2
                end
            end
        end
        [g_Ability,g_Simplicity]=update_diff(A,oldA,S,oldS)%梯度
        err=(norm(g_Ability)^2+norm(g_Simplicity)^2)/(Nwork+Ntask)
    end   
end
end