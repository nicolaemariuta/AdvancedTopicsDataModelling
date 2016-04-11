%Exercise 3: Ancestral Sampling

%choose the number of samples to generate
nrSamples = 10000;

%matrix with the obtaines samples
samples = zeros(nrSamples,7)

%Generate the desired number of samples
for i = 1:nrSamples
    %generate vector of 7 random numbers
    randomNumbers = abs(rand(1,7));
   
    %calculate value of x1 according to the random number generated
    P_x1_0 = 1/(1+exp(-1+2*0));
    if (randomNumbers(1) < P_x1_0)
        x1 = 0;
    else
        x1 = 1;
    end
    
    %calculate value of x2
     P_x2_0 = 1/(1+exp(-2+2*0));
    if (randomNumbers(2) < P_x2_0)
        x2 = 0;
    else
        x2 = 1;
    end
    
    
     %calculate value of x3
     P_x3_0 = 1/(1+exp(-3+2*0));
    if (randomNumbers(3) < P_x3_0)
        x3 = 0;
    else
        x3 = 1;
    end
    
    %calculate value of x4
    P_x4_0 = 1/(1+exp(-4+2*(x1+x2+x3)));
    if (randomNumbers(4) < P_x4_0)
        x4 = 0;
    else
        x4 = 1;
    end
    
    %calculate value of x5
    P_x5_0 = 1/(1+exp(-5+2*(x1+x3)));
    if (randomNumbers(5) < P_x5_0)
        x5 = 0;
    else
        x5 = 1;
    end
    
   %calculate value of x6
    P_x6_0 = 1/(1+exp(-6+2*(x4)));
    if (randomNumbers(6) < P_x6_0)
        x6 = 0;
    else
        x6 = 1;
    end
    
    %calculate value of x7
    P_x7_0 = 1/(1+exp(-7+2*(x4+x5)));
    if (randomNumbers(7) < P_x7_0)
        x7 = 0;
    else
        x7 = 1;
    end
    %insert the obtained sample to the matrix of samples
    samples(i,:) = [x1, x2, x3, x4, x5, x6, x7]; 
end

%marginal probabilities, first row will have the probability that the
%node to have value 0 and second row for the probabilities of value 1
marginalProbabilities = zeros(2,7);

%calculate marginal probablities by counting how many times node has value
%1 and 0 in the samples obtained
for i = 1 : 7
    nrNonZero = nnz(samples(:,i));
    marginalProbabilities(1,i) = nrSamples-nrNonZero;
    marginalProbabilities(2,i) = nrNonZero;
    
end

marginalProbabilities = marginalProbabilities./nrSamples;
disp('Marginal probability for all variables:');
disp(marginalProbabilities);

%calculate marginal probabilities for nodes x1,x4,x7
marginalProbabilities147 = zeros(power(2,3),4);
for i = 1 : power(2,3)
    sample = de2bi(i-1,3);
    
    count = 0;
    for r = 1 : nrSamples
        row = samples(r,:);
        if((row(1) == sample(1)) && (row(4) == sample(2)) && (row(7) == sample(2)))
            count = count+1;
        end
        
    end
    
    marginalProbab = count/nrSamples;
    marginalProbabilities147(i,:) = [sample, marginalProbab];
end

disp('Marginal probability of node triplet (x1,x4,x7):');
disp(marginalProbabilities147);
















