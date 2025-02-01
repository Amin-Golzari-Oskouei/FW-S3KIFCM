function [Cluster_elem,M,Z]=FWCW_FCM(X,M,k,t_max,N,fuzzy_degree,d,q, sigma, beta)
%
%Shirin Khezri, Nasser Aghazadehb, Mahdi Hashemzadehc, Amin Golzari Oskouei, "FW-S3KIFCM: Feature Weighted Safe-Semi-Supervised Kernel-Based 
% Intuitionistic Fuzzy C-Means Clustering Method", Fuzzy Information and Engineering, 2025.
%
%Function Inputs
%===============
%
%X is an Nxd data matrix, where each row corresponds to an instance.
%
%M is a kxd matrix of the initial cluster centers. Each row corresponds to a center.
%
%k is the number of clusters.
%
%t_max is the maximum number of iterations.
%
%beta and sigma are user defined parameter
%
%Function Outputs
%================
%
%Cluster_elem is a kxd matrix containing the final cluster assignments.
%
%M is a kxd matrix of the final cluster centers. Each row corresponds to a center.
%
%z is a kxd matrix of the final weights of each fatuter in each cluster.
%
%Courtesy of A. Golzari Oskouei

if q==0
    error('beta must be a non-zero number');
end

%--------------------------------------------------------------------------
%Weights are uniformly initialized.
Z=ones(k,d)/d;  %initial faeture weights

%Other initializations.
empty=0; %Count the number of iterations for which an empty or singleton cluster is detected.
Iter=1; %Number of iterations.
O_F_old=inf; %Previous iteration objective (used to check convergence).
Cluster_elem_history=[];
z_history=[];
%--------------------------------------------------------------------------

fprintf('\nStart of fuzzy C-means clustering method based on feature-weight and cluster-weight learning iterations\n');
fprintf('----------------------------------\n\n');

%The proposed iterative procedure.
while 1
    
    %Update the cluster assignments.
    for j=1:k
        distance(j,:,:) = 1-exp((-1)*(((X-repmat(M(j,:),N,1)).^2) / (2 * (sigma.^2))));
        WBETA = transpose(Z(j,:).^q);
        WBETA(WBETA==inf)=0;
        dNK(:,j) = reshape(distance(j,:,:),[N,d]) * WBETA   ;
    end
    
    tmp1 = zeros(N,k);
    for j=1:k
        tmp2 = (dNK./repmat(dNK(:,j),1,k)).^(1/(fuzzy_degree-1));
        tmp2(tmp2==inf)=0;
        tmp2(isnan(tmp2))=0;
        tmp1=tmp1+tmp2;
    end
    Cluster_elem = transpose(1./tmp1);
    
    Cluster_elem(isnan(Cluster_elem))=1;
    Cluster_elem(Cluster_elem==inf)=1;
    
    if nnz(dNK==0)>0
        for j=1:N
            if nnz(dNK(j,:)==0)>0
                Cluster_elem(find(dNK(j,:)==0),j) = 1/nnz(dNK(j,:)==0);
                Cluster_elem(find(dNK(j,:)~=0),j) = 0;
            end
        end
    end
    
    %Update the cluster assignments star.
    Cluster_elem_star = 1 - ((1 - (Cluster_elem.^beta)).^ (1/beta));
    
    %Update the cluster assignments star.
    pi = Cluster_elem_star - Cluster_elem;
    
    %Calculate the fuzzy C-means clustering method based on feature-weight and cluster-weight learning objective.
    O_F=object_fun(N,d,k,Cluster_elem,M,fuzzy_degree,Z,q,X, sigma, Cluster_elem_star, pi);
    
    if ~isnan(O_F)
        fprintf('The clustering objective function is %f\n\n',O_F);
    end
    
    %Check for convergence. Never converge if in the current (or previous)
    %iteration empty or singleton clusters were detected.
    if Iter>=t_max || ~isnan(O_F) && ~isnan(O_F_old) && (abs(1-O_F/O_F_old) < 1e-6 )
        
        fprintf('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n');
        fprintf('Converging for after %d iterations.\n',Iter);
        fprintf('The final objective function is =%f.\n',O_F);
        fprintf('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n');
        
        break;
        
    end
    
    O_F_old=O_F;
    
    %Update the cluster centers.
    mf = (Cluster_elem.^fuzzy_degree) + (Cluster_elem_star.^fuzzy_degree);       % MF matrix after exponential modification
    for j=1:k
        M(j,:) = (mf(j,:) * (X .* (exp((-1)*(((X-repmat(M(j,:),N,1)).^2) / (2 * (sigma.^2)))))))./(((mf(j,:)*(exp((-1)*(((X-repmat(M(j,:),N,1)).^2) / (2 * (sigma.^2)))))))); %new center
    end
    
    %Update the feature weights.
    for j=1:k
        distance(j,:,:) = 1-exp((-1)*(((X-repmat(M(j,:),N,1)).^2) / (2 * (sigma.^2))));
        dWkm(j,:) = (mf(j,:)) * reshape(distance(j,:,:),[N,d]);
    end
    
    tmp1 = zeros(k,d);
    for j=1:d
        tmp2 = (dWkm./repmat(dWkm(:,j),1,d)).^(1/(q-1));
        tmp2(tmp2==inf)=0;
        tmp2(isnan(tmp2))=0;
        tmp1=tmp1+tmp2;
    end
    Z = 1./tmp1;
    Z(isnan(Z))=1;
    Z(Z==inf)=1;
    
    if nnz(dWkm==0)>0
        for j=1:k
            if nnz(dWkm(j,:)==0)>0
                Z(j,find(dWkm(j,:)==0)) = 1/nnz(dWkm(j,:)==0);
                Z(j,find(dWkm(j,:)~=0)) = 0;
            end
        end
    end
    
    Iter=Iter+1;
end
end



