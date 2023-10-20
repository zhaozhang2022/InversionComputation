
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

qwt_real=np.loadtxt('realqwt_imp_sgsim100k_noisefixedsigma_dt100knt120.txt')

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
        self.markbc = 0
        self.markwell = 0
        self.press = 0


class CallingCounter(object):
    def __init__ (self, func):
        self.func = func
        self.count = 0

    def __call__ (self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

@CallingCounter
def singlephase_unsteady_imp(kvec): #homogeneous reservoir input k output transient p of bottomleft corner producer
    chukvec = np.dot(kvec, np.linalg.pinv(selected_eigenvectors))
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
        for j in range(0, 4):
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
                elif j == 4 or j == 5:
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
    qwt=np.zeros(nt)   # record flow rate with time
    for i in range(0, ncell):  # initial condition
        celllist[i].press = p_init
    Acoef = np.zeros((ncell, ncell))
    RHS = np.zeros(ncell)
    for t in range(nt):
        for ie in range(ncell):
            icell = ie
            p_i = celllist[icell].press
            Acoef[ie, ie] = celllist[icell].porosity * ct * celllist[icell].volume / dt
            RHS[ie] = celllist[icell].porosity * ct * celllist[icell].volume / dt * p_i
            if celllist[icell].markwell > 0:
                qw = PI * (celllist[icell].press - pwf)
                RHS[ie] = RHS[ie] - qw
                qwt[t] = qw
            for j in range(6):
                je = celllist[icell].neighbors[j]
                if je >= 0:
                    Acoef[ie, je] = -celllist[ie].trans[j]
                    Acoef[ie, ie] = Acoef[ie, ie] + celllist[ie].trans[j]
        press, exit_code = cg(Acoef, RHS, x0=None, tol=1e-05)
        for ie in range(ncell):
            icell = ie
            celllist[icell].press = press[ie]
    return qwt

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
    ob1=np.dot(np.dot(kvec1t, inv(pcov)),kvec1)
    obm=(qwt_sim-qwt_real)
    obmt=obm.T
    ob2=np.dot(np.dot(obmt, inv(cov_ob)),obm)
    return -ob1-ob2 #变为最大化,也是正比于log概率
print("build Grid")
dxvec=[0]
for i in range(0, 10):
    dxvec.append(10)

dyvec=[0]
for i in range(0, 10):
    dyvec.append(10)
dzvec=[0,5]

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
mu_o = 2e-3
ct = 5e-8
poro = 0.1
for i in range(0, ncell):
    celllist[i].porosity = poro
print("define well condition")
rw = 0.05
SS = 3
length = 3000
cs = ct * 3.14 * rw * rw * length
ddx = 10
re = 0.14 * (ddx * ddx + ddx * ddx) ** 0.5
PI = 2 * 3.14 * ddx * 2.5e-15 / mu_o / (math.log(re / rw) + SS)
pwf = 20e6  # bottom-hole pressure
celllist[0].markwell = 1
# simulation settings same as the qwt_real which is synthetic
p_init=30.0*1e6
nt = 120
dt = 100000

# prior***********
starttime=time.time()
# Build Initial Population /////////////////////////////////////////////////////////////////////////////////////////////////
perms_sgsim = np.loadtxt('sgsim1010_1001.txt',skiprows=0)
perms_sgsim = perms_sgsim[:,1:1001]
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
# check=np.dot(eigenvectors, np.dot(np.diag(eigenvalues),eigenvectors.T))
sorted_ind = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_ind]
sorted_eigenvectors = eigenvectors[:,sorted_ind]
tol = 0.6
eig_cum = np.cumsum(sorted_eigenvalues)
n_components = np.where(eig_cum/eig_cum[-1]>tol)[0]
n_components = n_components[0]
selected_eigenvalues = sorted_eigenvalues[:n_components]
selected_eigenvectors = sorted_eigenvectors[:,:n_components]
permvec_init_pca = np.dot(permvec_init_std,selected_eigenvectors) # Y_t=X_tV
covcheck=np.cov(permvec_init_pca.T)
# permvec_init_stdapprox=np.dot(permvec_init_pca,np.linalg.pinv(selected_eigenvectors))
# permvec_init_stdapprox2=np.dot(permvec_init_pca,selected_eigenvectors.T)
ndim=n_components
# qcovy = np.diag(vary)
# pcov=np.cov(permvec_init_pca.T)  #先验y协方差 几乎等于斜对角，但有小负数导致不正定


pcov=np.diag(selected_eigenvalues)
pmean=np.mean(permvec_init_pca,axis=0) #先验y均值


# start optimization
nsamples=50
maxupdate=10
obsigma_=2e-6
C_D=np.diag(np.ones(len(qwt_real))*obsigma_**2)
# C_D=maxupdate*C_D
df=np.zeros((nsamples, nt))  # forecast data
mf=permvec_init_pca[:nsamples]  # n*m
datavec=np.zeros(nsamples)
qmean = np.mean(mf, axis=0)
qcov = np.diag(np.var(mf, axis=0))
regular = np.float(regularizer(qmean, qcov, pmean, pcov))
for i in range(nsamples):
    isample=mf[i]
    df[i]=singlephase_unsteady_imp(isample)
    loglike = loglikelihoodprob(qwt_real, df[i])
    datavec[i]=loglike
data_term=np.mean(datavec)
nelbo=regular-data_term
print('initial negativeelbo: ', nelbo, regular, -1*data_term)
f = open('writeELBO_pca60_50samples-10updates.txt','w')
f.write("%e %e %e \n" % (nelbo, regular, -1*data_term))

cov_search=pcov*0.01
for iupdate in range(maxupdate):
    print('Iteration ', iupdate)
    if iupdate>0:
        # 通过转移概率生成新样本
        for i in range(nsamples):
            sample = np.random.multivariate_normal(mf2[i], cov_search)
            simqwt = singlephase_unsteady_imp(sample)
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
    sumnew0=random.uniform(0, 1/nsamples)
    jj=0
    mf2 = np.zeros((nsamples, ndim))
    df2 = np.zeros((nsamples, nt))
    cdf=np.cumsum(weightvec)
    for i in range(nsamples):
        sumnew = sumnew0 + i / nsamples
        while sumnew>cdf[jj]:
            jj=jj+1
        mf2[i]=mf[jj]
        df2[i]=df[jj]
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
print(singlephase_unsteady_imp.count)
f = open('distributions_pca60_50samples-10updates.txt','w')
f.write("result mean: \n")
for i in range(ndim):
    f.write("%e " % (qmean[i]))
f.write("\n")
f.write("result cov diagonal: \n")
for i in range(ndim):
    f.write("%e " % (qcov[i,i]))
f.write("\n")
f.write("Maxupdate is %d\n" % (maxupdate))
f.write("Time is %e\n" % (endtime-starttime))
f.write("forward count is %e\n" % singlephase_unsteady_imp.count)
f.write("-ELBO is %e\n" % nelbo)
f.close()

f = open('allperm_pca60_50samples_10updates.txt','w')
chukmean=np.zeros(ncell)
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
f = open('allY_pca60_50samples_10updates.txt','w')
for i in range(nsamples):
    for j in range(ndim):
        f.write("%e " % (mf[i, j]))
    f.write("\n")
f.close()
print("output to vtk")
f = open('result_PCA60_150samples_10updates.vtk','w')
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
f.write("SCALARS MeanPerm_mD double\n")
f.write("LOOKUP_TABLE default\n")
for i in range(0, len(celllist)):
    f.write("%0.3f\n" % (chukmean[i]))
f.close()









