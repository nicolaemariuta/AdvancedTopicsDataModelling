% MCMC for multivariate normal
rng('shuffle')

Sigma = [ 0.3 0.2 ; 0.2 0.2 ] ;
mu = [ 1 ; 1 ];

[V,D]=eig(Sigma);
s=diag(D);

%sampler = 'MH'; % options: {'exact','MH','Gibbs'}
%sampler = 'exact'; 
sampler = 'Gibbs';
L = 500; % number of samples
%stepsize = 0.5; % step-size in M-H proposal density. 
%stepsize = sqrt(s(1))
stepsize = sqrt(s(2))
lagmax = 100; % maximum lag in autocorrelation
burnin = 0; % Burnin time for Gibbs sampler
%burnin = 100; % Burnin time for Gibbs sampler

y = zeros(length(mu),L); % samples
C = chol(Sigma,'lower');

% calculate inverse for computing the normal density
% In general, it is more stable to use Cholesky and \ operator
Sigmainv = inv(Sigma) ; 

% choose initial value - fixed for illustration purposes
y(:,1) = - mu ;
%y(:,1) = mu ;


disp('Order of decorrelation steps: ')
switch sampler
    case 'MH'
        (sqrt(s(2))/sqrt(s(1)))^2
    case 'Gibbs'
        (Sigma(1,1) * Sigmainv(1,1))^2
end



switch sampler
    case 'exact'
        
        y = mu(:,ones(L,1)) + C * randn(length(mu),L) ;
        
    case 'MH'
        
                
        for l=2:L
            yprop = y(:,l-1) + stepsize * randn(length(mu),1) ;
            
            % only compute terms that don't cancel in M-H acceptance
            if rand < exp(-0.5*(yprop-mu)'*Sigmainv*(yprop-mu) ...
                          +0.5*(y(:,l-1)-mu)'*Sigmainv*(y(:,l-1)-mu))
            
                y(:,l) = yprop ;
            else
                y(:,l) = y(:,l-1);
            end
        end
       
    case 'Gibbs'

        
        for l=2:(L+burnin)
            y(1, l) = mu(1) - Sigmainv(1,2) / Sigmainv(1,1) * (y(2,l-1) - mu(2)) + sqrt(1/Sigmainv(1,1)) * randn(1);
            y(2, l) = mu(2) - Sigmainv(1,2) / Sigmainv(2,2) * (y(1,l) - mu(1)) + sqrt(1/Sigmainv(2,2)) * randn(1);
        end
        
        y = y(:, (burnin+1):end);
        
end

disp('Sample mean and variance: ')
mean(y')
cov(y')


% plot contour lines and samples
figure(1)
% con = C * [ cos(0:0.01:2*pi) ; sin(0:0.01:2*pi) ] ; con = con + mu(:,ones(size(con,2),1)) ;
% plot(con(1,:),con(2,:),y(1,:),y(2,:),'*')
dx = -1:0.1:3;
dy = -1:0.1:3;
[ grdx grdy ] = meshgrid( dx, dy );
dens = mvnpdf( [ grdx(:) grdy(:) ], mu', Sigma );
dens = reshape( dens, length( dx ), length( dy ) );
imagesc( dx, dy, dens )
set(gca,'YDir','normal')
hold on
plot(y(1,2:end),y(2,2:end),'w*')
%plot(-mu(1), -mu(2), 'r*')
plot(y(1,1), y(2,1), 'r*')
hold off

% plot auto-correlation function
figure(2)
muhat = mean(y,2);
sigma2hat = std(y,0,2).^2;

R = zeros(length(mu),lagmax) ;
for k=1:lagmax
    for d=1:length(mu)
        R(d,k) = 1/((L-k)*sigma2hat(d)) * (y(d,1:end-k) - muhat(d) ) *  (y(d,1+k:end) - muhat(d) )' ; 
    end
end

for d=1:length(mu)
    subplot(length(mu),1,d)
    plot(1:k,R(d,:),'o' );
    xlabel( 'lag k' ); ylabel( 'corr coeff' )
    grid on
    axis([1 lagmax -1 1])
end
