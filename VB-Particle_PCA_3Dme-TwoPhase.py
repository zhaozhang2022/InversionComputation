
import sys
import torch
import torchvision
import torch.utils.data
import math
import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
from scipy.sparse.linalg import minres
from scipy.sparse.linalg import cg
import random
import copy
import time
from scipy.stats import multivariate_normal as mvnorm

qwt_real=np.loadtxt('realqwt_imp_noisefixedsigma_dt100knt1440.txt')

class NODE:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

class CELL:
    def __init__(self): # 不加self就变成了对所有类对象同时更改
        self.vertices = [-1, -1, -1, -1, -1, -1, -1, -1]
        self.neighbors = [-1, -1, -1, -1, -1, -1]
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.volume = 0
        self.xc = 0
        self.yc = 0
        self.zc = 0
        self.porosity = 0
        self.kx = 0
        self.ky = 0
        self.kz = 0
        self.trans = [0, 0, 0, 0, 0, 0]
        self.transw = [0, 0, 0, 0, 0, 0]
        self.markbc = 0
        self.press = 0
        self.Sw = 0
        self.markwell=0
        self.mobiw=0
        self.mobio=0
        self.mobit=0

def computemobi():
    for ie in range(0, ncell):
        sw=celllist[ie].Sw
        a=(1-sw)/(1-Siw)
        b=(sw-Siw)/(1-Siw)
        kro=a*a*(1-b*b)
        krw=b*b*b*b
        vro=kro/mu_o
        vrw=krw/mu_w
        celllist[ie].mobio=vro
        celllist[ie].mobiw=vrw
        celllist[ie].mobit=vro+vrw


class CallingCounter(object):
    def __init__ (self, func):
        self.func = func
        self.count = 0

    def __call__ (self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

@CallingCounter
def twophase_impes(kvec): #homogeneous reservoir input k output transient p of bottomleft corner producer
    chukvec = np.dot(kvec, selected_eigenvectors.T)
    chukvec = chukvec * pstd + pmeanx
    chukvec = 2 ** chukvec * 0.1 * 1e-15
    for i in range(0, ncell):  # set chuk
        celllist[i].kx = chukvec[i]
        celllist[i].ky = chukvec[i]
        celllist[i].kz = chukvec[i]*0.1
    for ie in range(0, ncell):  # compute transmissibility
        dx1 = celllist[ie].dx
        dy1 = celllist[ie].dy
        dz1 = celllist[ie].dz
        for j in range(0, 6):
            je = celllist[ie].neighbors[j]
            if je >= 0:
                dx2 = celllist[je].dx
                dy2 = celllist[je].dy
                dz2 = celllist[je].dz
                mt1 = 1.0 / mu_o
                mt2 = 1.0 / mu_o
                if j == 0 or j == 1:
                    mt1 = mt1 * dy1 * dz1
                    mt2 = mt2 * dy2 * dz2
                    k1 = celllist[ie].kx
                    k2 = celllist[je].kx
                    dd1 = dx1 / 2.
                    dd2 = dx2 / 2.
                elif j == 2 or j == 3:
                    mt1 = mt1 * dx1 * dz1
                    mt2 = mt2 * dx2 * dz2
                    k1 = celllist[ie].ky
                    k2 = celllist[je].ky
                    dd1 = dy1 / 2.
                    dd2 = dy2 / 2.
                else:
                    mt1 = mt1 * dx1 * dy1
                    mt2 = mt2 * dx2 * dy2
                    k1 = celllist[ie].kz
                    k2 = celllist[je].kz
                    dd1 = dz1 / 2.
                    dd2 = dz2 / 2.
                t1 = mt1 * k1 / dd1
                t2 = mt2 * k2 / dd2
                tt = 1 / (1 / t1 + 1 / t2)
                celllist[ie].trans[j] = tt
    qwt=np.zeros((4,nt))   # record flow rate with time
    for i in range(0, ncell):  # initial condition
        celllist[i].press = p_init
        celllist[i].Sw = Siw
    presslast = np.zeros(ncell)
    for t in range(nt):
        # print('iteration: ', t)
        computemobi()
        # implicit pressure
        if t % dtscale == 0:
            Acoef = np.zeros((ncell, ncell))
            RHSvec = np.zeros(ncell)
            for ie in range(ncell):
                Sw = celllist[ie].Sw
                So = 1.0 - Sw
                Ct = Co * So + Cw * Sw
                p_i = celllist[ie].press
                Acoef[ie, ie] = celllist[ie].porosity * Ct * celllist[ie].volume / dt_p
                RHSvec[ie] = celllist[ie].porosity * Ct * celllist[ie].volume / dt_p * p_i
                if celllist[ie].markbc == -2:  # 生产
                    PI = PIcoef * celllist[ie].mobit * celllist[ie].kx
                    qt = PI * (celllist[ie].press - bhp_constant)
                    RHSvec[ie] = RHSvec[ie] - qt
                elif celllist[ie].markbc == -1:  # 注水
                    RHSvec[ie] = RHSvec[ie] + qw_fixed
                for j in range(6):
                    je = celllist[ie].neighbors[j]
                    if je >= 0:
                        Tij = celllist[ie].trans[j]
                        if celllist[ie].press > celllist[je].press:
                            mobi = celllist[ie].mobit
                        elif celllist[ie].press < celllist[je].press:
                            mobi = celllist[je].mobit
                        else:
                            mobi = (celllist[ie].mobit + celllist[je].mobit) * 0.5
                        Acoef[ie, je] = -Tij * mobi
                        Acoef[ie, ie] = Acoef[ie, ie] + Tij * mobi
            press, exit_code = cg(Acoef, RHSvec, x0=None, tol=1e-05)
            for ie in range(ncell):
                celllist[ie].press = press[ie]
                if press[ie] < 0:
                    print('negative press at ', ie, press[ie])
            for ie in range(ncell):
                presslast[ie] = celllist[ie].press
        # explicit saturation
        for ie in range(ncell):
            RHS = 0
            if celllist[ie].markbc == -2:
                PI = PIcoef * celllist[ie].mobit * celllist[ie].kx
                qt = PI * (celllist[ie].press - bhp_constant)
                qwt[celllist[ie].markwell - 1, t] = qt
                PIw = PIcoef * celllist[ie].mobiw * celllist[ie].kx
                qw = PIw * (celllist[ie].press - bhp_constant)
                RHS = RHS - qw
            elif celllist[ie].markbc == -1:
                RHS = RHS + qw_fixed
            pi = celllist[ie].press
            for j in range(6):
                je = celllist[ie].neighbors[j]
                if je >= 0:
                    pj = celllist[je].press
                    Tij = celllist[ie].trans[j]
                    if pi > pj:
                        mobiw = celllist[ie].mobiw
                    elif pi < pj:
                        mobiw = celllist[je].mobiw
                    else:
                        mobiw = (celllist[ie].mobiw + celllist[je].mobiw) * 0.5
                    RHS = RHS - mobiw * Tij * (pi - pj)
            RHS = RHS - (pi - presslast[ie]) / dt * poro * celllist[ie].volume * celllist[ie].Sw * Cw
            celllist[ie].Sw = celllist[ie].Sw + RHS * dt / poro / celllist[ie].volume
    qwtall = np.reshape(qwt, (-1))
    sw=np.zeros(ncell)
    pressvec=np.zeros(ncell)
    for i in range(ncell):
        sw[i]=celllist[i].Sw
    return qwtall, sw, pressvec

def loglikelihoodprob(qwt_obs, qwt_sim):
    pp=0.0
    if len(qwt_obs)!=len(qwt_sim):
        print('observed and simulated sequences of different lengths')
        sys.exit(0)
    for i in range(len(qwt_obs)):
        p1=qwt_obs[i]
        p2=qwt_sim[i]
        sigma_=obsigma_
        z=(p2-p1)/sigma_
        f=1.0/math.sqrt(2*math.pi)*math.exp(-z**2/2)
        if f<1e-100:
            f=1e-100
        pp=pp+math.log(f)
    return pp


def regularizer(meanvec, covmatrix, pmean, pcov):  # regularizer
    qdis = torch.distributions.MultivariateNormal(torch.tensor(meanvec), torch.tensor(covmatrix))
    pdis = torch.distributions.MultivariateNormal(torch.tensor(pmean), torch.tensor(pcov))
    result = torch.distributions.kl_divergence(qdis, pdis).mean()
    return result
def objective(kvec, pmean,pcov, qwt_sim, qwt_real, cov_ob): #用概率的话，perm分布的正则化项概率太小。
    kvec1=kvec-pmean
    kvec1t=kvec1.T
    # covinv=inv(pcov)
    ob1=np.dot(np.dot(kvec1t, inv(pcov)),kvec1)
    obm=(qwt_sim-qwt_real)
    obmt=obm.T
    ob2=np.dot(np.dot(obmt, inv(cov_ob)),obm)
    return -ob1-ob2 #变为最大化,也是正比于log概率

print("build Grid")
dxvec=[0]
for i in range(0, 20):
    dxvec.append(15)

dyvec=[0]
for i in range(0, 20):
    dyvec.append(15)
dzvec=[0]
for i in range(0, 5):
    dzvec.append(6)

nx=len(dxvec)-1
ny=len(dyvec)-1
nz=len(dzvec)-1
nodelist=[]
llz = 0
for k in range(0, nz+1):
    llz = llz + dzvec[k]
    lly=0
    for j in range(0, ny+1):
        lly = lly + dyvec[j]
        llx = 0
        for i in range(0, nx+1):
            llx = llx + dxvec[i]
            node=NODE()
            node.x=llx
            node.y=lly
            node.z=llz
            nodelist.append(node)

# build connectivity and neighbors
celllist=[]

for k in range(0, nz):
    for j in range(0, ny):
        for i in range(0, nx):
            id = k * nx * ny + j * nx + i
            nc=id
            cell = CELL()
            if i>0:
                cell.neighbors[0] = nc - 1
            if i<nx-1:
                cell.neighbors[1] = nc + 1
            if j>0:
                cell.neighbors[2] = nc - nx
            if j<ny-1:
                cell.neighbors[3] = nc + nx
            if k>0:
                cell.neighbors[4] = nc - nx*ny
            if k<nz-1:
                cell.neighbors[5] = nc + nx * ny
            i0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
            i1 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1
            i2 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
            i3 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i + 1
            i4 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i
            i5 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1
            i6 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
            i7 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i + 1
            cell.dx = nodelist[i1].x - nodelist[i0].x
            cell.dy = nodelist[i2].y - nodelist[i0].y
            cell.dz = nodelist[i4].z - nodelist[i0].z
            cell.vertices[0] = i0
            cell.vertices[1] = i1
            cell.vertices[2] = i2
            cell.vertices[3] = i3
            cell.vertices[4] = i4
            cell.vertices[5] = i5
            cell.vertices[6] = i6
            cell.vertices[7] = i7
            cell.xc = 0.125 * (nodelist[i0].x+nodelist[i1].x+nodelist[i2].x+nodelist[i3].x+nodelist[i4].x+nodelist[i5].x+nodelist[i6].x+nodelist[i7].x)
            cell.yc = 0.125 * (nodelist[i0].y + nodelist[i1].y + nodelist[i2].y + nodelist[i3].y + nodelist[i4].y + nodelist[i5].y + nodelist[i6].y + nodelist[i7].y)
            cell.zc = 0.125 * (nodelist[i0].z + nodelist[i1].z + nodelist[i2].z + nodelist[i3].z + nodelist[i4].z + nodelist[i5].z + nodelist[i6].z + nodelist[i7].z)
            cell.volume=cell.dx*cell.dy*cell.dz
            celllist.append(cell)

cellvolume=celllist[0].volume

ncell=len(celllist)
print("define properties")
mu_o = 1.8e-3
mu_w = 1e-3
poro = 0.2
Siw=0.2
Cw = 4 * 1e-6 / 6894
Co = 100 * 1e-6 / 6894
for i in range(0, ncell):
    celllist[i].porosity = poro
print("define initial condition")
p_init=30.0*1e6
print("define well condition")
rw = 0.05
SS = 3
length = 3000
bhp_constant = 25e6 #bottom-hole pressure
qw_fixed = 10.0/86400
ddx = dxvec[1]-dxvec[0]
re = 0.14*(ddx*ddx + ddx*ddx)**0.5
PIcoef = 2 * 3.14*ddx / (math.log(re / rw) + SS)
nwell=4
for k in range(0, nz):
    for j in range(0, ny):
        for i in range(0, nx):
            id = k * nx * ny + j * nx + i
            if i==0 and j==0:
                celllist[id].markwell = 1
                celllist[id].markbc = -2 #生产井
            elif i==19 and j==0:
                celllist[id].markwell = 2
                celllist[id].markbc = -2
            elif i==0 and j==19:
                celllist[id].markwell = 3
                celllist[id].markbc = -2
            elif i==19 and j==19:
                celllist[id].markwell = 4
                celllist[id].markbc = -2
            elif i==9 and j==9:
                celllist[id].markwell = 5
                celllist[id].markbc = -1 #注水井

# Build Prior Model /////////////////////////////////////////////////////////////////////////////////////////////////
perms_sgsim = np.loadtxt('perms3dme.txt',skiprows=0)
perms_sgsim = perms_sgsim[:,1:1999]
# 用的时候再permvec_init[:,:]=2**permvec_init[:,:]*0.1*1e-15序贯高斯模拟的是h，直接h降维更有效果
permvec_init=perms_sgsim.copy()
permvec_init = permvec_init.T
pmeanx=np.mean(permvec_init,axis=0)
# permvec_init规范化后主成分分析
pstd = np.std(permvec_init,axis=0)
pstd=np.where(pstd<1e-10, 1, pstd)
permvec_init_std = (permvec_init - pmeanx)/pstd
pcovx=np.cov(permvec_init_std.T)
eigenvalues,eigenvectors = np.linalg.eig(pcovx)
sorted_ind = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_ind]
sorted_eigenvectors = eigenvectors[:,sorted_ind]
tol = 0.9999
eig_cum = np.cumsum(sorted_eigenvalues)
n_components = np.where(eig_cum/eig_cum[-1]>tol)[0]
n_components = n_components[0]
selected_eigenvalues = sorted_eigenvalues[:n_components].real
selected_eigenvectors = sorted_eigenvectors[:,:n_components].real
permvec_init_pca = np.dot(permvec_init_std,selected_eigenvectors) # Y_t=X_tV
# permvec_init_stdapprox=np.dot(permvec_init_pca,np.linalg.pinv(selected_eigenvectors))
# permvec_init_stdapprox2=np.dot(permvec_init_pca,selected_eigenvectors.T)
ndim=n_components
# qcovy = np.diag(vary)
# pcov=np.cov(permvec_init_pca.T)  #先验y协方差 几乎等于斜对角，但有小负数导致不正定
pcov=np.diag(selected_eigenvalues)
pmean=np.mean(permvec_init_pca,axis=0) #先验y均值

print("simulation settings")
nt = 1440
dt=100000
dtscale = 5
dt_p=dt*dtscale
starttime=time.time()
# start optimization
nsamples=100
maxupdate=10
obsigma_=2e-6
C_D=np.diag(np.ones(len(qwt_real))*obsigma_**2)
df=np.zeros((nsamples, nt*nwell))  # forecast data
mf=permvec_init_pca[:nsamples]  # n*m
datavec=np.zeros(nsamples)
qmean = np.mean(mf, axis=0)
qcov = np.diag(np.var(mf, axis=0))
regular = np.float(regularizer(qmean, qcov, pmean, pcov))
allswprior=np.zeros((nsamples, ncell))
allpressprior=np.zeros((nsamples, ncell))
for i in range(nsamples):
    isample=mf[i]
    df[i], allswprior[i], allpressprior[i]=twophase_impes(isample)
    loglike = loglikelihoodprob(qwt_real, df[i])
    datavec[i]=loglike
data_term=np.mean(datavec)
nelbo=regular-data_term
print('initial negativeelbo: ', nelbo, regular, -1*data_term)
f = open('writeELBO_pca100_100samples-10updates.txt','w')
f.write("%e %e %e \n" % (nelbo, regular, -1*data_term))

cov_search=pcov*0.01
allsw=np.zeros((nsamples, ncell))
allpress=np.zeros((nsamples, ncell))
for iupdate in range(maxupdate):
    print('Iteration ', iupdate)
    if iupdate>0:
        # 通过转移概率生成新样本
        for i in range(nsamples):
            sample = np.random.multivariate_normal(mf2[i], cov_search)
            simqwt, allsw[i], allpress[i] = twophase_impes(sample)
            newlogp = objective(sample, pmean, pcov, simqwt, qwt_real, C_D)
            oldlogp = objective(mf2[i], pmean, pcov, df2[i], qwt_real, C_D)
            logpp = newlogp - oldlogp  # 转移矩阵对称
            if logpp < -100:
                mf[i] = mf2[i]
                df[i] = df2[i]
            elif logpp >= 0:
                mf[i] = sample
                df[i] = simqwt
            else:
                pp = 2.71828 ** logpp
                alpha = min(1, pp)
                u = random.uniform(0, 1)
                if u < alpha:
                    mf[i] = sample
                    df[i] = simqwt
                else:
                    mf[i] = mf2[i]
                    df[i] = df2[i]
        # （或通过esmda生成新样本）
        print('Compute datavec for new samples')
        for i in range(nsamples):
            loglike = loglikelihoodprob(qwt_real, df[i])
            datavec[i] = loglike
    print('Compute Weight')
    weightvec=-1/datavec # weightvec = 1.1**datavec
    weightvec=weightvec/np.sum(weightvec)
    # 重采样保证样本大小不变
    sumnew0 = random.uniform(0, 1 / nsamples)
    jj = 0
    mf2 = np.zeros((nsamples, ndim))
    df2 = np.zeros((nsamples, nt * nwell))
    cdf = np.cumsum(weightvec)
    for i in range(nsamples):
        sumnew = sumnew0 + i / nsamples
        while sumnew > cdf[jj]:
            jj = jj + 1
        mf2[i] = mf[jj]
        df2[i] = df[jj]
    print('evaluate ELBO of resampled realisations')
    qmean = np.mean(mf2, axis=0)
    qcov = np.diag(np.var(mf2, axis=0))
    regular = np.float(regularizer(qmean, qcov, pmean, pcov))
    for i in range(nsamples):
        isample = mf[i]
        loglike = loglikelihoodprob(qwt_real, df2[i])
        datavec[i] = loglike
    data_term = np.mean(datavec)
    nelbo = regular - data_term
    print('negativeelbo: ', nelbo, regular, -1 * data_term)
    f.write("%e %e %e \n" % (nelbo, regular, -1 * data_term))
    print(datavec)


endtime=time.time()

f.close()
print(endtime-starttime)
print(twophase_impes.count)
f = open('distributions_pca100_100samples_10updates.txt','w')
f.write("result cov diagonal: \n")
for i in range(ndim):
    for j in range(ndim):
        f.write("%e " % (qcov[i,j]))
    f.write("\n")
f.write("Maxupdate is %d\n" % (maxupdate))
f.write("Time is %e\n" % (endtime-starttime))
f.write("forward count is %e\n" % twophase_impes.count)
f.write("-ELBO is %e\n" % nelbo)
f.close()

chukmean=np.zeros(ncell)
f = open('allperm_pca100_100samples_10updates.txt','w')
for i in range(nsamples):
    chukvec = np.dot(mf[i], selected_eigenvectors.T)
    chukvec = chukvec * pstd + pmeanx
    chukvec = 2 ** chukvec * 0.1
    chukmean=chukmean+chukvec
    for j in range(ncell):
        f.write("%e " % (chukvec[j]))
    f.write("\n")
f.close()
chukmean=chukmean/nsamples
f = open('allY_pca100_100samples_10updates.txt','w')
for i in range(nsamples):
    for j in range(ndim):
        f.write("%e " % (mf[i, j]))
    f.write("\n")
f.close()
swmean=np.mean(allsw, axis=0)
pressmean=np.mean(allpress, axis=0)
swmeanprior=np.mean(allswprior, axis=0)
pressmeanprior=np.mean(allpressprior, axis=0)
print("output to vtk")
f = open('result_3dme-TwoPhase_PCA100_100samples_10updates.vtk','w')
f.write("# vtk DataFile Version 2.0\n")
f.write( "Unstructured Grid\n")
f.write( "ASCII\n")
f.write("DATASET UNSTRUCTURED_GRID\n")
f.write("POINTS %d double\n" % (len(nodelist)))
for i in range(0, len(nodelist)):
    f.write("%0.3f %0.3f %0.3f\n" % (nodelist[i].x, nodelist[i].y, nodelist[i].z))
f.write("\n")
f.write("CELLS %d %d\n" % (len(celllist), len(celllist)*9))
for i in range(0, len(celllist)):
    f.write("%d %d %d %d %d %d %d %d %d\n" % (8, celllist[i].vertices[0], celllist[i].vertices[1], celllist[i].vertices[3], celllist[i].vertices[2], celllist[i].vertices[4], celllist[i].vertices[5], celllist[i].vertices[7], celllist[i].vertices[6]))
f.write("\n")
f.write("CELL_TYPES %d\n" % (len(celllist)))
for i in range(0, len(celllist)):
    f.write("12\n")
f.write("\n")
f.write("CELL_DATA %d\n" % (len(celllist)))
f.write("SCALARS Permeability_mD double\n")
f.write("LOOKUP_TABLE default\n")
for i in range(0, len(celllist)):
    f.write("%0.3f\n" % (celllist[i].kx*1e15))
f.write("SCALARS Pressure double\n")
f.write("LOOKUP_TABLE default\n")
for i in range(0, len(celllist)):
    f.write("%0.3f\n" % (celllist[i].press/10**6))
f.write("SCALARS Sw double\n")
f.write("LOOKUP_TABLE default\n")
for i in range(0, len(celllist)):
    f.write("%0.3f\n" % (celllist[i].Sw))
f.write("SCALARS MeanPressure double\n")
f.write("LOOKUP_TABLE default\n")
for i in range(0, len(celllist)):
    f.write("%0.3f\n" % (pressmean[i]/10**6))
f.write("SCALARS MeanSw double\n")
f.write("LOOKUP_TABLE default\n")
for i in range(0, len(celllist)):
    f.write("%0.3f\n" % (swmean[i]))
f.write("SCALARS MeanPerm double\n")
f.write("LOOKUP_TABLE default\n")
for i in range(0, len(celllist)):
    f.write("%0.3f\n" % (chukmean[i]))
f.write("SCALARS MeanSwPrior double\n")
f.write("LOOKUP_TABLE default\n")
for i in range(0, len(celllist)):
    f.write("%0.3f\n" % (swmeanprior[i]))
f.write("SCALARS MeanPressPrior double\n")
f.write("LOOKUP_TABLE default\n")
for i in range(0, len(celllist)):
    f.write("%0.3f\n" % (pressmean[i]/10**6))
f.close()




