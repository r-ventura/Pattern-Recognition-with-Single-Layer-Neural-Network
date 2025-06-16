function [wo,fo,niter] = uo_nn_GM(w,L1,gL1,epsG,kmax,ils,almax,ialmax,kmaxBLS,epsal,c1,c2)
    wo = [w];
    dk = [];
    alk = [];
    k = 1;
    
    while norm(gL1(w)) > epsG && k < kmax
        % Encuentra el valor de 'd' para esta iteración
        d = -gL1(w);
        dk = [dk d];

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
        
        % Calculamos y guardamos el nuevo valor de w, y contamos una nueva iteración
        w = w + al*d;
        wo = [wo w];
        k = k + 1;
    end
    
    wo = w;
    fo = L1(wo);
    niter = k;
end
