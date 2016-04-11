%read the input image 
clearvars;
Y = int8(imread('noisyImage.png'));

%replace 1 with -1 and 0 with 1 in matrix Y and init matrix X with the
%values from Y
Y(Y==0) = -1;
Y = Y*(-1);
X = Y;
imagesc(Y),colormap(gray),title('Initial noisy image');

%parameters for energy calculation
h = 0.0;
beta = 1.0;
eta = 1.0;

%apply the ICM algorithm by repeated raster scanning through image
for i = 1:10
    %take all nodes from X in order and adjust the pixel value
    disp('i=');
    disp(i);
    for r = 1:size(X,1)
        for c = 1:size(X,2)
           %calculate total energy  for xi = 1
            X1 = X;
            X1(r,c) = 1;
            E1 = totalEnergy(X1,Y,h,beta,eta);
            %calculate total energy  for xi = -1
            X2 = X;
            X2(r,c) = -1;
            E2 = totalEnergy(X2,Y,h,beta,eta);
            %keep value for which energy has lower value
            if(E1<E2)
                X(r,c) = 1;
            else
                X(r,c) = -1;
            end
         end
    end
end

finalEnergy = totalEnergy(X,Y,h,beta,eta);
disp(finalEnergy);

figure;
imagesc(X),colormap(gray),title( ['ICM h=' num2str(h) ', beta=' num2str(beta) ', eta=' num2str(eta) ', finalEnergy=' num2str(finalEnergy)] );


%%
%Implementation of ICM optimized, according to answer from Question 3

%read the input image 
clearvars;
Y = int8(imread('noisyImage.png'));

%replace 1 with -1 and 0 with 1 in matrix Y and init matrix X with the
%values from Y
Y(Y==0) = -1;
Y = Y*(-1);
X = Y;
imagesc(Y),colormap(gray),title('Initial noisy image');

%parameters for energy calculation
h = 0.9;
beta = 1.0;
eta = 1.0;
%the two possible values for the binary pixels
x1 = 1;
x2 = -1;

%apply the ICM algorithm by repeated raster scanning through image
for i = 1:3
    %take all nodes from X in order and adjust the pixel value
    for r = 2:(size(X,1)-1)
        for c = 2:(size(X,2)-1)
            %calculate total energy  for xi = 1
            E1 = x1*(h-beta*(X((r-1),c)+X(r+1,c)+X(r,c-1)+X(r,c+1))-eta*Y(r,c));
            %calculate total energy  for xi = -1
            E2 = x2*(h-beta*(X(r-1,c)+X(r+1,c)+X(r,c-1)+X(r,c+1))-eta*Y(r,c));
            %keep value for which energy has lower value
           
            
            
            if(E1<E2)
                X(r,c) = 1;
            else
                X(r,c) = -1;
            end
        end
    end
end


finalEnergy = totalEnergy(X,Y,h,beta,eta);
disp(finalEnergy);

figure;
imagesc(X),colormap(gray),title( ['ICM h=' num2str(h) ', beta=' num2str(beta) ', eta=' num2str(eta) ', finalEnergy=' num2str(finalEnergy)] );

