function j_fun = object_fun_FWCW_PSSFCM(N,d,k,Cluster_elem,M,fuzzy_degree,z,beta_z,X,alpha_1, alpha_2, pre_Cluster_elem,b,f,Cluster_elem_star, pi, sigma)
mf = (Cluster_elem.^fuzzy_degree) + (Cluster_elem_star.^fuzzy_degree);
for j=1:k
    distance(j,:,:) = 1-exp((-1)*(((X-repmat(M(j,:),N,1)).^2) / (2 * (sigma.^2))));
    WBETA = transpose(z(j,:).^beta_z);
    WBETA(WBETA==inf)=0;
    dNK(:,j) = reshape(distance(j,:,:),[N,d]) * WBETA ;
end

j_fun1 = sum(sum(dNK .* transpose(mf)));

value = pi' .* repmat((exp(1 - ((1/N) * sum(pi', 1)))), N, 1);
j_fun4 = (1/N) * sum(sum(value));

j_fun2 = sum(sum(dNK .* transpose((Cluster_elem-(b.*f)').^fuzzy_degree)));
j_fun3 = sum(sum(dNK .* transpose((Cluster_elem-(b.*pre_Cluster_elem)').^fuzzy_degree)));

j_fun = j_fun1 + (alpha_1 * j_fun2) + (alpha_2 * j_fun3) + j_fun4 ;
end

