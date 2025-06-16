function [wo,fo,niter] = uo_nn_BFGS(w,L1,gL1,epsG,kmax,ils,almax,ialmax,kmaxBLS,epsal,c1,c2) 
    wo = [w];
    dk = [];
    alk = [];
    I = eye(length(w)); % Matriz identidad
    H = I; Hk(:,:,1) = H;
    k = 1;

    while norm(gL1(w)) > epsG && k < kmax
        % Encontramos el valor de 'd' para esta iteración
        d = -H*gL1(w);
        dk = [dk,d];

        % Encontramos el valor de alpha para esta iteración
        % Si ils = 3, usamos BLSNW32  
        if ils == 3
            if size(alk)~=0 % Tomamos una standard max step length para la primera iteración
                if ialmax == 1
                    almax = (alk(:,end))*(gL1(wo(:,end))'*dk(:,end))/(gL1(w)'*d); % almax1
                else
                    almax = 2*(L1(w)-L1(wo(:,end-1)))/(gL1(w)'*d); % almax2
                end
            end
            
            [al,~] = uo_BLSNW32(L1,gL1,w,d,almax,c1,c2,kmaxBLS,epsal);
            alk = [alk al];
        end

        % Guarda en 'oldw' el valor actual de 'w', que será el de la próxima iteración
        oldw = w;
        w = oldw + al*d;
        wo = [wo,w];

        % Declara 's', 'y' y 'p' para la iteración en curso
        s = w - oldw;
        y = gL1(w) - gL1(oldw);
        p = 1/(y'*s);

        % Calcula y guarda el valor actual de H
        H = (I-(p*s*y'))*H*(I-(p*y*s'))+p*s*s';
        k = k + 1;
        Hk(:,:,k) = H;
    end

    wo = w;
    fo = L1(wo);
    niter = k;
end