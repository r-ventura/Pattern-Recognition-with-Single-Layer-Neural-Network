function [wo,fo,niter] = uo_nn_SGM(w,la,L,gL,Xtr,ytr,Xte,yte,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest)

    p = length(Xtr);
    m = floor(sg_ga*p); % Tamaño del minibatch
    sg_ke = ceil(p/m);
    sg_kmax = sg_emax*sg_ke;
    e = 0;
    s = 0;
    L_te_best = Inf;
    k = 0;
    
    % Creamos matrices de tamaño n*kmax vacías para acelerar la ejecución
    wk = NaN(size(w,1),sg_kmax); 
    dk = NaN(size(w,1),sg_kmax); 
    alk = NaN(1,sg_kmax);
    wk(:,1) = w;

    while e <= sg_emax && s < sg_ebest
        P = randperm(p); % Permutación de Xtr

        for i = 0:ceil((p/m)-1)
            S = P(i*m+1:min(i*m+m,p));
            XtrS = Xtr(:,S);
            ytrS = ytr(:,S);
            
            % Encontramos el valor de 'd' para esta iteración
            d = -gL(w,Xtr(:,S),ytr(:,S),la);
            dk(:,k+1)= d;
            
            % Encontramos el valor de alpha para esta iteración (learning rate)
            sg_al = 0.01*sg_al0;
            sg_k = floor(sg_be*sg_kmax);
            if k <= sg_k
                al = (1 - (k/sg_k))*sg_al0 + (k/sg_k)*sg_al;
            else
                al = sg_al;
            end
            
            % Calculamos y guardamos los nuevos valores, y contamos una nueva iteración
            alk(k+1) = al;
            w = w + al*d;
            wk(:,k+2)= w;
            k = k + 1;

        end

        e = e + 1; 
        L_te = L(w, Xte, yte, la);

        % Stopping criterion
        if L_te < L_te_best 
            L_te_best = L_te;
            wo = w;
            s = 0;
        else 
            s = s + 1;
        end
    end
    
    wk = wk(:,~all(isnan(wk))); dk = dk(:,~all(isnan(dk))); alk = alk(~isnan(alk));
    fo = L(wo, Xte, yte, la);
    niter = k;
end
