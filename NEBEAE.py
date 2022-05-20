"""
Non-linear Extended Blind Endmembers and Abundances Extraction (NEBEAE) Algorithm

"""
import time
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import svds
import scipy.linalg as splin

def performance(fn):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        # print(f'Function {fn.__name__} took {t2-t1} s')
        return t2 - t1, result
    return wrapper

def pca(X,d):
    L, N = X.shape
    xMean = X.mean(axis=1).reshape((L,1))
    xzm = xMean - np.matlib.repmat(xMean, 1, N)
    U, _ , _  = svds( (xzm @ xzm.T)/N , k=d)
    return U

def NFINDR(Y, N):
    """
    N-FINDR endmembers estimation in multi/hyperspectral dataset
    """
    L, K = Y.shape
    # dimention redution by PCA
    U = pca(Y,N)
    Yr = U.T @ Y
    # Initialization
    Po = np.zeros((L,N))
    IDX = np.zeros((1,K))
    TestM = np.zeros((N,N))
    TestM[0,:]=1
    for i in range (N):
        idx = np.floor(float(np.random.rand(1))*K) + 1
        TestM[1:N,i]= Yr[:N-1,i]
        IDX[0,i]=idx
    actualVolume = np.abs(np.linalg.det(TestM))
    it=1
    v1=-1
    v2=actualVolume
    #  Algorithm
    maxit = 3 * N
    while (it<maxit and v2>v1):
        for k in range (N):
            for i in range (K):
                actualSample = TestM[1:N,k]
                TestM[1:N,k] = Yr[:N-1,i]
                volume = np.abs(np.linalg.det(TestM))
                if volume > actualVolume:
                    actualVolume = volume
                    IDX[0,k] = i
                else:
                    TestM[1:N,k]=actualSample
        it = it + 1
        v1 = v2
        v2 = actualVolume
    
    for i in range (N):
        Po[:,i] = Y[:,int(IDX[0,i])].copy()
    return Po

def vca(Y,R):
    #############################################
    # Initializations
    #############################################
    [L, N]=Y.shape   # L number of bands (channels), N number of pixels     
    R = int(R)
    #############################################
    # SNR Estimates
    #############################################  
    y_m = np.mean(Y,axis=1,keepdims=True)
    Y_o = Y - y_m           # data with zero-mean
    Ud  = splin.svd(np.dot(Y_o,Y_o.T)/float(N))[0][:,:R]  # computes the R-projection matrix 
    x_p = np.dot(Ud.T, Y_o)                 # project the zero-mean data onto p-subspace
    P_y     = np.sum(Y**2)/float(N)
    P_x     = np.sum(x_p**2)/float(N) + np.sum(y_m**2)
    SNR = 10*np.log10( (P_x - R/L*P_y)/(P_y - P_x) ) 
    SNR_th = 15 + 10*np.log10(R)+8
    
    #############################################
    # Choosing Projective Projection or 
    #          projection to p-1 subspace
    #############################################
    if SNR < SNR_th:
        d = R-1
        Ud = Ud[:,:d]
        Yp =  np.dot(Ud,x_p[:d,:]) + y_m      # again in dimension L
        x = x_p[:d,:] #  x_p =  Ud.T * Y_o is on a R-dim subspace
        c = np.argmax(np.sum(x**2,axis=0))**0.5
        y = np.vstack(( x, c*np.ones((1,N)) ))
    #############################################
    # VCA algorithm
    #############################################

    indice = np.zeros((R),dtype=int)
    A = np.zeros((R,R))
    A[-1,0] = 1

    for i in range(R):
        w = np.random.rand(R,1);   
        f = w - np.dot(A,np.dot(splin.pinv(A),w))
        f = f / splin.norm(f)
        v = np.dot(f.T,y)
        indice[i] = np.argmax(np.abs(v))
        A[:,i] = y[:,indice[i]]        # same as x(:,indice(i))
    Ae = Yp[:,indice]
    return Ae,indice,Yp


@performance
def abundance(Z, Y, P, D, Lambda, parallel):
    """
    A = abundance(Z,Y,P,D,lambda,parallel)
    Estimation of Optimal Abundances in Nonlinear Mixture Model
    Input Arguments
    Z --> matrix of measurements
    Y --> matrix of normalized measurements
    P --> matrix of end-members
    D --> vector of probabilities of nonlinear mixing
    lambda -->  entropy weight in abundance estimation \in (0,1)
    parallel --> implementation in parallel of the estimation
    Output Argument
    A = abundances matrix 
    Daniel U. Campos-Delgado
    February/2021
    python version march/2022 JNMC
    """
# Check arguments dimensions
    numerr = 0
    M, N = Y.shape
    n = P.shape[1]
    A= np.zeros((n, N))
    if P.shape[0] != M:
        print("ERROR: the number of rows in Y and P does not match for abudances estimation")
        print('P.shape[0]=',P.shape[0])
        print('M=' ,M )
        numerr = 1
        return A, numerr

    for k in range(N):
        yk = np.c_[Y[:,k]]
        zk = np.c_[Z[:,k]]
        byk = float(yk.T@yk)
        dk = np.c_[D[k]]
        deltakn = (1-dk)*np.ones((n,1))+dk*P.T@zk
        Pk = np.multiply(P,((1-dk)*np.ones((M,n))+dk*zk*np.ones((1,n))))
        bk = Pk.T @ yk
        Go = Pk.T @ Pk
        eGo, _ = np.linalg.eig(Go)
        eGo[np.isnan(eGo)]=1e6
        eGo[np.isinf(eGo)]=1e6
        lmin=np.amin(eGo)
        G=Go-np.eye(n)*lmin*Lambda
        Gi=np.linalg.pinv(G)
        #Gi = np.divide(np.eye(n),G)
#Compute Optimal Unconstrained solutions
        sigma = np.divide((deltakn.T@Gi@bk-1),(deltakn.T@Gi@deltakn))
        ak= Gi@ (bk-deltakn*sigma)
#Check for Negative Elements
        if float(sum(ak >= 0)) != n:
            I_set = np.zeros((1, n))
            while float(sum(ak < 0)) != 0:
                I_set = np.where(ak < 0, 1, I_set.T).reshape(1, n)
                L = len(np.where(I_set == 1)[1])
                Q = n+1+L
                Gamma = np.zeros((Q, Q))
                Beta = np.zeros((Q, 1))
                Gamma[:n, :n] = G
                Gamma[:n, n] = (deltakn*byk).reshape(3,)
                Gamma[n, :n] = deltakn.T
                cont = 0
                for i in range(n):
                    if I_set[:,i] != 0:
                        cont += 1
                        ind = i
                        Gamma[ind, n+cont] = 1
                        Gamma[n+cont, ind] = 1
                Beta[:n, :] = bk
                Beta[n, :] = 1
                delta = np.linalg.solve(Gamma, Beta)
                ak = delta[:n]
                ak = np.where(abs(ak) < 1e-9, 0, ak)
        A[:,k] = np.c_[ak].T
    return A, numerr

@performance
def probanonlinear(Z,Y,P,A,parallel):
    """
    D = probanonlinear(Z,Y,P,A,parallel)
    Estimation of Probability of Nonlinear Mixtures 
    Input Arguments
    Z --> matrix of measurements
    Y -> matrix of normalized measurements
    P --> matrix of end-members
    A -->  matrix of abundances
    parallel = implementation in parallel of the estimation
    Output Arguments
    D = Vector of probabilities of Nonlinear Mixtures
    Daniel U. Campos-Delgado February/2021
    Python version march/2022 JNMC
    """
    M, N = Y.shape
    D = np.zeros((N,1))
    #Do = np.zeros((N,1))
    for k in range(N):
        yk = np.c_[Y[:,k]]
        zk = np.c_[Z[:,k]]
        ak = np.c_[A[:,k]]
        ek = P @ ak
        T1 = ek - yk 
        T2 = ek - (ek*zk)
        dk = np.min([1, float(T1.T@T2/(T2.T@T2))])
        D[k]=dk
    return D

@performance 
def endmember(Z, Y, Po, A, D, rho):
    """
    P = endmember(Z,Y,P,A,rho,normalization)
    Estimation of Optimal End-members in Linear Mixture Model
    Input Arguments
    Z --> matrix of measurements
    Y -> matrix of normalized measurements
    P --> matrix of end-members
    A -->  matrix of abundances
    D --> vector of probabilities of nonlinear mixture
    rho = Weighting factor of regularization term
    Output Arguments
    P --> matrix of end-members
    Daniel U. Campos-Delgado Feb/2021
    Python version march/2022 JNMC
    """
    #Compute Gradient of Cost Function
    n, N = A.shape #n=number of endmembers N=pixels
    M, K = Y.shape #M=Bands K= pixels
    numerr = 0
    if Y.shape[1] != N:
        print("ERROR: the number of columns in Y and A does not match")
        numerr = 1
    GradP = np.zeros((M,n))
    R = sum(n-np.array(range(1, n),dtype='object'))
    for k in range(N):
        yk = np.c_[Y[:,k]]
        zk = np.c_[Z[:,k]]
        ak = np.c_[A[:,k]]
        dk = np.c_[D[k]]
        byk = yk.T @ yk
        Mk=np.diag(((1-dk)*np.ones((M,1))+dk*zk).reshape(M,))
        GradP = GradP- (Mk.T @ yk @ ak.T / byk) +( Mk.T @Mk @ Po @ ak @ ak.T) / byk   
    O = n * np.eye(n) - np.ones((n,n))
    GradP = GradP/N + rho * Po @ O/R

    #Compute Optimal Step in Update Rule
    numG = rho * np.trace(GradP @ O @ Po.T + Po @ O @ GradP.T)/R/2
    denG = rho * np.trace(GradP @ O @ GradP.T)/R
        
    for k in range (N):
        yk = np.c_[Y[:,k]]
        zk = np.c_[Z[:,k]]
        ak = np.c_[A[:,k]]
        dk = np.c_[D[k]]
        byk = yk.T @ yk
        Mk=np.diag(((1-dk)*np.ones((M,1))+dk*zk).reshape(M,))
        T1 = Mk @ GradP @ ak
        numG = numG + T1.T @ Mk @ (Po @ ak - yk) / byk / N
        denG = denG + T1.T @ T1 / byk / N
            
        alpha = np.max([0, numG/denG])
        # Compute the Stepest Descent Update of End-members Matrix
        P_est = Po - alpha * GradP
        P_est[P_est<0]=0
        P_est[np.isnan(P_est)]=0
        P_est[np.isinf(P_est)]=0
        P = P_est / np.sum(P_est, axis=0)
    return P, numerr

@performance
def nebeae(Yo,n,parameters,Po,oae):
    """
    nebeae(Yo,N,parameters,Po,oae)
    Input Arguments
    Y = matrix of measurements (LxK)
    n = order of linear mixture model
    parameters  = 8x1 vector of hyper-parameters in BEAE methodology
                = [initicond rho lambda epsilon maxiter downsampling  ...
                    parallel display]
        initcond = initialization of end-members matrix {1,2,3}
                                (1) Maximum cosine difference from mean
                                    measurement (default)
                                (2) Maximum and minimum energy, and
                                    largest distance from them
                                (3) PCA selection + Rectified Linear Unit
                                (4) ICA selection (FOBI) + Rectified
                                Linear Unit
                                (5) N-FINDR endmembers estimation in a 
                                multi/hyperspectral dataset (Winter,1999)
                                (6) Vertex Component Analysis (VCA)
                            (Nascimento and Dias, 2005)
        rho = regularization weight in end-member estimation 
            (default rho=0.1);
        lambda = entropy weight in abundance estimation \in [0,1) 
                (default lambda=0);
        epsilon = threshold for convergence in ALS method 
                (default epsilon=1e-3); 
        maxiter = maximum number of iterations in ALS method
                (default maxiter=20);
        downsampling = percentage of random downsampling in end-member 
                    estimation [0,1) (default downsampling=0.5);
        parallel = implement parallel computation of abundances (0 -> NO or 1 -> YES)
                (default parallel=0);
        display = show progress of iterative optimization process (0 -> NO or 1 -> YES)
                (default display=0);
    Po = initial end-member matrix (LxN)
    oae = only optimal abundance estimation with Po (0 -> NO or 1 -> YES)
        (default oae = 0)
    Output Arguments
    P = matrix of end-members (LxN)
    A  = abudances matrix (NxK)
    Ds = vector of nonlinear interaction levels (Kx1)
    S  = scaling vector (Kx1)
    Yh = estimated matrix of measurements (LxK) --> Yh = P*A*diag(S)
    Daniel U. Campos Delgado August/2021
    Python ver march 2021 JNMC
    """
    # Default parameters
    initcond = 1
    rho = 0.1
    Lambda = 0
    epsilon = 1e-3
    maxiter = 20
    downsampling = 0.5
    parallel = 0
    display = 0
    numerr = 0
    #Checking consistency of imput arguments
    nargin = 0
    if type(Yo) == np.ndarray:
        nargin += 1
    if type(n) == int:
        nargin += 1
    if type(parameters) == list:
        nargin += 1
    if type(Po) == np.ndarray:
        nargin += 1
    if type(oae) == int:
        nargin += 1
    if nargin != 5:
        oae = 0
    if nargin == 0:
        print("The measurement matrix Y has to be used as argument!!")
    elif nargin == 1:
        n = 2
    elif nargin == 3 or nargin == 4 or nargin == 5:
        if len(parameters) != 8:
            print("The length of parameters vector is not 8 !!")
            print("Default values of hyper-parameters are used instead")
        else:
            #print("se usaran los valores proporcionados")
            initcond, rho, Lambda, epsilon, maxiter, downsampling, parallel, display = parameters
            #print(parameters)
            
            if initcond != 1 and initcond != 2 and initcond != 3  and initcond != 4 and initcond != 5 and initcond != 6:
                print("The initialization procedure of endmembers matrix is 1,2,3,4,5 or 6!")
                print("The default value is considered!")
                initcond = 1
            if rho < 0:
                print("The regularization weight rho cannot be negative")
                print("The default value is considered!")
                rho = 0.1
            if Lambda < 0 or Lambda >= 1:
                print("The entropy weight lambda is limited to [0,1)")
                print("The default value is considered!")
                Lambda = 0
            if epsilon < 0 or epsilon > 0.5:
                print("The threshold epsilon can't be negative or > 0.5")
                print("The default value is considered!")
                epsilon = 1e-3
            if maxiter < 0 and maxiter < 100:
                print("The upper bound maxiter can't be negative or >100")
                print("The default value is considered!")
                maxiter = 20
            if 0 > downsampling > 1:
                print("The downsampling factor cannot be negative or >1")
                print("The default value is considered!")
                downsampling = 0.5
            if parallel != 0 and parallel != 1:
                print("The parallelization parameter is 0 or 1")
                print("The default value is considered!")
                parallel = 0
            if display != 0 and display != 1:
                print("The display parameter is 0 or 1")
                print("The default value is considered")
                display = 0
        if n < 2:
            print("The order of the linear mixture model has to be greater than 2!")
            print("The default value n=2 is considered!")
            n = 2
    if nargin == 4 or nargin == 5:
        if type(Po) != np.ndarray:
            print("The initial end-members Po must be a matrix !!")
            print("The initialization is considered by the maximum cosine difference from mean measurement")
            initcond = 1
        else:
            if Po.shape[0] == Yo.shape[0] and Po.shape[1] == n:
                initcond = 0
            else:
                print("The size of Po must be M x n!!")
                print("The initialization is considered based on the input dataset")
                initcond = 1
    if nargin == 5:
        if oae != 0 and oae != 1:
            print("The assignment of oae is incorrect!!")
            print("The initial end-members Po will be improved iteratively from a selected sample")
            oae = 0
        elif oae == 1 and initcond != 0:
            print("The initial end-members Po is not defined properly!")
            print("Po will be improved iteratively from a selected sample")
            oae = 0
    if nargin >= 6:
        print("The number of input arguments is 5 maximum")
        print("Please check the help documentation.")
    
    # Random downsampling
    if type(Yo) != np.ndarray:
        print("The measurements matrix Y has to be a matrix")
    M, No = Yo.shape
    if M > No:
        print("The number of spatial measurements has to be larger to the number of time samples!")
    
    
    #downsanmpling
    N = round(No*(1-downsampling))
    I = np.array(range(No))
    Is = np.random.choice(No, N, replace=False)
    Y = Yo[:, Is-1]
    # Normalization
    mYm = np.sum(Y, 0)
    mYmo = np.sum(Yo, 0)
    Ym = Y / np.tile(mYm, [M, 1])
    Ymo = Yo / np.tile(mYmo, [M, 1])
    NYm = np.linalg.norm(Ym, 'fro')

    # Selection of Initial Endmembers Matrix
    if initcond == 1 or initcond == 2:
        if initcond == 1:
            Po = np.zeros((M, 1))
            index = 1
            p_max = np.mean(Yo, axis=1)
            Yt = Yo
            Po[:, index-1] = p_max
        elif initcond == 2:
            index = 1
            Y1m = np.sum(abs(Yo), 0)
            y_max = np.max(Y1m)
            Imax = np.argwhere(Y1m == y_max)[0][0]
            y_min = np.min(Y1m)
            I_min = np.argwhere(Y1m == y_min)[0][0]
            p_max = Yo[:, Imax]
            p_min = Yo[:, I_min]
            K = Yo.shape[1]
            II = np.arange(1, K+1)
            condition = np.logical_and(II != II[Imax], II != II[I_min])
            II = np.extract(condition, II)
            Yt = Yo[:, II-1]
            Po = p_max
            index += 1
            Po = np.c_[Po, p_min]
        while index < n:
            y_max = np.zeros((1, index))
            Imax = np.zeros((1, index), dtype=int)
            for j in range(index):
                if j == 0:
                    for i in range(index):
                        e1m = np.around(np.sum(Yt*np.tile(Po[:, i], [Yt.shape[1], 1]).T, 0) /
                                        np.sqrt(np.sum(Yt**2, 0))/np.sqrt(np.sum(Po[:, i]**2, 0)), 4)
                        y_max[j][i] = np.around(np.amin(abs(e1m)), 4)
                        Imax[j][i] = np.where(e1m == y_max[j][i])[0][0]
            ym_max = np.amin(y_max)
            Im_max = np.where(y_max == ym_max)[1][0]
            IImax = Imax[0][Im_max]
            p_max = Yt[:, IImax]
            index += 1
            Po = np.c_[Po, p_max]
            II = np.arange(1, Yt.shape[1]+1)
            II = np.extract(II != IImax+1, II)
            Yt = Yt[:, list(II-1)]
    elif initcond == 3:
        UU, s, VV = np.linalg.svd(Ym.T, full_matrices=False)
        W = VV.T[:, :n]
        Po = W * np.tile(np.sign(W.T@np.ones((M, 1))).T, [M, 1])
    elif initcond == 4:
        Yom = np.mean(Ym, axis=1)
        Yon = Ym-np.tile(Yom, [n, 1]).T
        UU, s, VV = np.linalg.svd(Yon.T, full_matrices=False)
        S = np.diag(s)
        Yo_w = np.linalg.pinv(linalg.sqrtm(S)) @ VV @ Ym
        V, s, u = np.linalg.svd((np.tile(sum(Yo_w * Yo_w), [M, 1]) * Yo_w) @ Yo_w.T, full_matrices=False)
        W = VV.T @ linalg.sqrtm(S)@V[:n, :].T
        Po = W*np.tile(np.sign(W.T@np.ones((M, 1))).T, [M, 1])
    elif initcond == 5:
        Po = NFINDR(Yo,n)
    elif initcond == 6:
        Po,_,_ = vca(Yo,n)
    else:
        P=Po.copy()
        
    Po = np.where(Po < 0, 0, Po)
    Po = np.where(np.isnan(Po), 0, Po)
    Po = np.where(np.isinf(Po), 0, Po)
    mPo=Po.sum(axis=0,keepdims=True)
    P=Po/mPo
    # alternated Least Squares Procedure
    ITER = 1
    J = 1e5
    Jp = 1e6
    a_Time = 0
    p_Time = 0
    d_Time = 0
    Dm = np.zeros((N,1))
    tic = time.time()
    
    while (Jp-J)/Jp >= epsilon and ITER < maxiter and oae == 0 and numerr == 0:
        t_A,  outputs_a = abundance(Y,Ym,P,Dm,Lambda, parallel)
        a_Time += t_A
        Am, numerr = outputs_a
        t_D, Dm = probanonlinear(Y,Ym,P,Am,parallel)
        d_Time += t_D
        Pp = P.copy()
        if numerr==0:
            t_P, outputs_e = endmember(Y,Ym,Pp,Am,Dm,rho)
            P, numerr = outputs_e
            p_Time += t_P
        Jp = J
        a = np.multiply(np.tile((1-Dm).T, [M,1]),P@Am)
        b = np.multiply(np.tile(Dm.T,[M,1]),np.multiply(P@Am,Y))
        J = np.linalg.norm(Ym-a-b,ord='fro')
        if J > Jp:
            P = Pp.copy()
            break
        if display == 1:
            print(f"Number of iteration = {ITER}")
            print(f"Percentage Estimation Error = {(100*J)/NYm} %")
            print(f"Abundance estimation took {t_A}")
            print(f"Endmember estimation took {t_P}")
        ITER += 1
    if numerr == 0:
        if oae==1:
            J=1e5
            Jp=1e6
            D=np.zeros((N,1))
            ITER = 1
            while (Jp-J)/Jp >= epsilon and ITER < maxiter:
                t_a,outputs_A= abundance(Yo,Ymo,D,Lambda,parallel)
                A, numerr = outputs_A
                t_D, D = probanonlinear(Yo,Ymo,P,A,parallel)
                Jp=J
                a = np.multiply(np.tile((1-D).T, [M,1]),P@A)
                b = np.multiply(np.tile(D.T,[M,1]),np.multiply(P@A,Yo))
                J = np.linalg.norm(Ymo-a-b,ord='fro')
                ITER += 1
            print('Percentage Estimation Error = ',100*J/NYm,'%')
        else:
            Ins = np.setdiff1d(I,Is)
            J = 1e5
            Jp = 1e6
            Dms=np.zeros((len(Ins),1))
            ITER = 1
            Ymos = Ymo[:,Ins]
            Yos = Yo[:,Ins]

            while(Jp-J)/Jp >= epsilon and ITER < maxiter:
                t_Ams, outp_Ams = abundance(Yos,Ymos,P,Dms,Lambda,parallel)
                Ams,numerr = outp_Ams
                t_dms, Dms = probanonlinear(Yos, Ymos,P,Ams,parallel)
                Jp = J
                a = np.multiply(np.tile((1-Dms).T, [M,1]),P@Ams)
                b = np.multiply(np.tile(Dms.T,[M,1]),np.multiply(P@Ams,Yos))
                J = np.linalg.norm(Ymos-a-b,ord='fro')
                ITER += 1
            A = np.concatenate((Am,Ams),axis=1)
            D = np.concatenate((Dm,Dms),axis=0)
            II = np.concatenate((Is,Ins))
            Index=np.argsort(II)
            A = A[:,Index]
            D = D[Index,:]
        toc = time.time()
        elap_time = toc-tic
        if display == 1:
            print(f"Elapsed Time = {elap_time} seconds")
        S = mYmo.T
        AA = np.multiply(A, np.tile(mYmo,[n,1]))
        a = np.multiply(np.tile((1-D).T,[M,1]),P@AA)
        b = np.multiply(np.tile(D.T,[M,1]),P@AA)
        Yh = a + np.multiply(b,Yo)
        t_ds, Ds = probanonlinear(Yo,Yo,P,A,parallel)
    else:
        print("Please revier the problem formulation, nor realiable results")
        P=np.array([])
        Ds=np.array([])
        S=np.array([])
        A=np.array([])
        Yh=np.array([])
    
    return P,A,Ds,S,Yh



