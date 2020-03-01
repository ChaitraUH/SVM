function [w,bias,margin,alpha,obj,preds,classpred]=SvmCrowd(c)
   %load('/Users/chaitrahegde/Documents/ML CSE 512/Lecture Slides/Homework 4/hw4/q2_1_data.mat');
   D = csvread("TrainFeaturesupdated.csv",1,2);
   trLb = csvread("TrainLabelsupdated.csv",1,2);
   Dval = csvread("ValFeaturesupdated.csv",1,2);
   valLb = csvread("ValLabelsupdated.csv",1,2);
   D=[D;Dval];
   trLb=[trLb;valLb];
   el=size(trLb,1);
   for i=1:4
      Ynew=-ones(el,1);
      for j=1:el
          if (trLb(j)==i)
              Ynew(j)=1;
          end
      end
      [w(:,i),bias(i),margin(i),alpha(:,i),obj(i)]=svmdual(D',Ynew,c);
   end
   %[w,bias,margin,alpha,obj]=svmdual(trD,trLb,c);
   test = csvread("TestFeatures2.csv",1,1);
   preds=predict(test',bias,w);
   classpred=assignclass(preds);
   
end 

function [w,bias,margin,alpha,obj]=svmdual(trD,trLb,c)
    X=trD;
    Y=trLb;
    H=(Y'*Y).*(X'*X);
    n=size(X,2);
    f=-ones(n,1);
    A=[];
    b=[];
    Aeq=Y';
    beq=0;
    lb=zeros(n,1);
    ub=c*ones(n,1);
    [alpha,f]=quadprog(H,f,A,b,Aeq,beq,lb,ub);
    w=X*(Y.*alpha);
    svcount=0; 
    total=0;
    for i=1:n
        val=round(alpha(i),4);
        if (val~=0)
            svcount=svcount+1;
            total=total+(Y(i)-w'*X(:,i));
        end
    end
    bias=total/svcount;
    %bias=Y(i)-(w'*(X(:,i)));
    margin=1/sqrt(sum(abs(alpha)));
    obj=-1*f;
end

function [preds]=predict(X,bias,w)
  for i=1:4
    preds(:,i)=(w(:,i)'*X)+bias(i);
  end
end

function [classpred]=assignclass(preds)
    el=size(preds,1); 
    for i=1:el
        [M,I] = max(preds(i,:));
        classpred(i,:)=I;
    end
end