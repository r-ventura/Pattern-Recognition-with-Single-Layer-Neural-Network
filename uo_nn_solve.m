function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu)

% isd = Search direction (1=GM; 3=BFGS; 7=SGM)
    fprintf(':::::::::::::::::::::::::::::::::::::::::::::::::::\n');
    fprintf('Pattern recognition with neural networks (OM/GCED).\n');
    fprintf('%s\n', datetime('now'));
    fprintf(':::::::::::::::::::::::::::::::::::::::::::::::::::\n');

    % Iniciamos el contador
    tic;


    % Generamos el training data set
    fprintf('Training data set generation.\n');
    fprintf('  num_target = %i\n', num_target);
    fprintf('  tr_freq    = %.2f\n', tr_freq);
    fprintf('  tr_p       = %i\n', tr_p);
    fprintf('  tr_seed    = %i\n', tr_seed);
    
    [Xtr,ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);
    te_freq = tr_freq / 10;
    

    % Generamos el test data set
    fprintf('Test data set generation.\n');
    fprintf('  te_freq    = %.2f\n', te_freq);
    fprintf('  te_q       = %i\n', te_q);
    fprintf('  te_seed    = %i\n', te_seed);
    
    [Xte,yte] = uo_nn_dataset(te_seed, te_q, num_target, te_freq);
    

    fprintf('Optimization\n');
    fprintf('L2 reg. lambda = %4.2f\n', la);
    fprintf('epsG= %6.1d, kmax= %i\n', epsG, kmax);
    fprintf('ils= %i, ialmax= %i, kmaxBLS= %i, epsBLS= %6.1d,\n', ils, ialmax, kmaxBLS, epsal);
    fprintf('c1= %.2f, c2= %.2f, isd= %i\n', c1, c2, isd);
    

    % Declaración de las funciones 
    sig = @(X) 1./(1+exp(-X));
    y = @(X,w) sig(w'*sig(X));
    L = @(w,Xtr,ytr,la) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2)+ (la*norm(w)^2)/2;
    gL = @(w,Xtr,ytr,la) (2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))')/size(ytr,2)+la*w;

    L1 = @(w) L(w,Xtr,ytr,la);
    gL1 = @(w) gL(w,Xtr,ytr,la);


    % Creamos el vector de ceros w
    w = zeros(35,1);
    almax  = 1;

    if isd == 1 % GM
        [wo,fo,niter] = uo_nn_GM(w,L1,gL1,epsG,kmax,ils,almax,ialmax,kmaxBLS,epsal,c1,c2);
    end

    if isd == 3 % BFGS
        [wo,fo,niter] = uo_nn_BFGS(w,L1,gL1,epsG,kmax,ils,almax,ialmax,kmaxBLS,epsal,c1,c2);
    end

    if isd == 7 % SGM
        rng(sg_seed);
        [wo,fo,niter] = uo_nn_SGM(w,la,L,gL,Xtr,ytr,Xte,yte,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest);                 
    end


    % Detenemos el contador
    tex = toc;


    % Representación de wo
    fprintf(']\n');
    fprintf('k= %i\n', niter);

    fprintf('wo=[\n')
    fprintf('   %+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e\n', wo(1:5))
    fprintf('   %+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e\n', wo(6:10))
    fprintf('   %+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e\n', wo(11:15))
    fprintf('   %+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e\n', wo(16:20))
    fprintf('   %+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e\n', wo(21:25))
    fprintf('   %+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e\n', wo(26:30))
    fprintf('   %+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e\n', wo(31:35))
    fprintf('   ]\n')


    % Cálculo i representación de AccuracyTR y AccuracyTE
    syms tr1 tr2;
    tr1 = sym(round(y(Xtr,wo)));
    tr2 = sym(ytr);
    tr_acc = 100/tr_p * sum(kroneckerDelta(tr1,tr2));
    
    syms te1 te2;
    te1 = sym(round(y(Xte,wo)));
    te2 = sym(yte);
    te_acc = 100/te_q * sum(kroneckerDelta(te1,te2));
    
    fprintf('Accuracy\n');
    fprintf('tr_accuracy = %1.3f\n', tr_acc);
    fprintf('te_accuracy = %1.3f\n', te_acc);
    
end