function [N] = local_neighbor(X, Num, paraDc)
[num_train,~] = size(X);
distance_matrix = pdist2(X, X,'euclidean');
instance_similar_matrix = exp(-distance_matrix.^2./ paraDc^2);
dist_max = distance_matrix + diag(realmax*ones(1,num_train));

nerghbor = zeros(num_train, num_train);
for i = 1:num_train
    [~,index] = sort(dist_max(i,:));
    neighbor_index = index(1:Num);
    nerghbor(neighbor_index,i) = 1;
end
N = nerghbor.*instance_similar_matrix;
N = N./repmat(sum(N), num_train, 1);
N(find(isnan(N))) = 0;
end