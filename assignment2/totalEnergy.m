function  E  = totalEnergy( X,Y, h, beta, eta )
   %implementation of the complete energy function 
   % based on equation E = h*sum(xi)+beta*sum(xi*xj)+eta*sum(xi*yi)
   
   %calculation of bias
   bias = h*sum(sum(X));
    
   %calculate energy for cliques formed by neighbouring nodes (xi,xj)
   cliqueNeighbours = sum(sum(X(1:size(X,1)-1,:).* X(2:size(X,1),:))) + sum(sum(X(:,1:size(X,2)-1).* X(:,1:size(X,2)-1)));
   cliqueNeighbours = cliqueNeighbours*beta;
   
   %calculate energy for cliques formed by pairs (xi,yi)
   cliqueXY = eta*sum(sum(X.*Y));
    
   %calculate complete energy
   E = bias -  cliqueNeighbours - cliqueXY;
   
end

