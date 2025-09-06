function [beta_opt,beta_cvrt,gamma_opt,CV] = cv_SPG_cvrt_py(prob,Y, X, Z, C, CNorm, option,g_idx) 
    % Modified to use Python's sklearn for cross-validation for consistent comparison
    
    if isfield(option, 'nfold')
        nfold=option.nfold;
    else
        nfold=5;
    end  
    
    if isfield(option, 'gammarange')
        gammarange=option.gammarange;
    else
        gammarange=exp(-5:0.1:0);
    end  
    
    nocvrt=isempty(Z);
    [n,p]=size(X);
    [~,q]=size(Z);
    % process C (this is the key to cvrts adjustment)
    C=[C,zeros(size(C,1),q)];
    %
    CV_score=zeros(nfold,length(gammarange));

    % Python part starts (I'm not sure if I need to load python also here)
    KFold = py.importlib.import_module('sklearn.model_selection').KFold;
    cv = KFold(pyargs("n_splits", py.int(nfold), "shuffle", true, "random_state", py.int(0)));
    split = py.list(cv.split(X, Y));

    for ifold=1:nfold
        indices = split{ifold};
        index_test=int64(indices{2}) + 1;
        index_train=int64(indices{1}) + 1;
        
        Xtrain=X(index_train,:);
        Ytrain=Y(index_train,:);
        Xtest=X(index_test,:);
        Ytest=Y(index_test,:);
        if nocvrt
            Ztrain=[];
            Ztest=[];
        else
            Ztrain=Z(index_train,:);
            Ztest=Z(index_test,:);
        end
    
        
        
        for itune=1:length(gammarange)
            gamma=gammarange(itune);
            option.verbose=false;
            if (strcmpi(prob, 'group'))
                [beta,~,~,~,~] = SPG(prob, Ytrain, [Xtrain,Ztrain], gamma, 0, C, CNorm, option,g_idx); 
            else
                [beta,~,~,~,~] = SPG(prob, Ytrain, [Xtrain,Ztrain], gamma, 0, C, CNorm, option);
            end
            
            cvscore=sum((Ytest-[Xtest,Ztest]*beta).^2)/length(Ytest); % MSE
            
            CV_score(ifold,itune)=cvscore;
        end
    end
        
    CV=mean(CV_score,1);
    [~,ind]=min(CV);
    gamma_opt=gammarange(ind);
    if (strcmpi(prob, 'group'))
        [beta_final,~,~,~,~] = SPG(prob, Y, [X,Z], gamma_opt, 0, C, CNorm, option,g_idx); 
    else
        [beta_final,~,~,~,~] = SPG(prob, Y, [X,Z], gamma_opt, 0, C, CNorm, option);
    end
    beta_opt=beta_final(1:p);
    beta_cvrt=beta_final((p+1):end);
    figure(100);clf
    plot(gammarange,CV,'o-');
    xlabel('gamma value');
    ylabel('CV score');