from dipy.data import default_sphere
from dipy.tracking.local_tracking import LocalTracking
import numpy as np
from scipy.interpolate import griddata
from dipy.tracking.streamline import Streamlines
import copy
from coordinates import toInds


def pointsPerLine(streamlines):
    N_lines=len(streamlines)
    ppline=np.zeros(N_lines)
    for l in range(0,N_lines):
        ppline[l]=len(streamlines[l])
    return ppline.astype(int)

def allLines2Lines(allLines,pointsPerLine):
    streamlines=[]
    first=0
    for i in range(0,len(pointsPerLine)):
        templine=[]
        for p in range(0,pointsPerLine[i]):
            point=allLines[first+i]
            if( np.isnan(np.sum(point))==0):
                templine.append(point)
        if(len(templine)>1):
            streamlines.append(templine)
        first=pointsPerLine[i]
    return streamlines
 
def connectedInds(line,mask_nii):
    cinds=[]
    stop=mask_nii.get_fdata()
    for p in range(0,len(line)):
        worlds = np.asarray([line[p,0], line[p,1], line[p,2]])
        inds = toInds(mask_nii, [worlds])
        inds = np.asarray(inds[0])
        inds = inds.round()
        inds = inds.astype(int)
        condition = inds[0] >= stop.shape[0] or inds[1] >= stop.shape[1] or inds[2] >= stop.shape[2]
        if condition==False:
            cinds.append(list(inds)) if list(inds) not in cinds else cinds
    return cinds

class trueTracts:
    def __init__(self,mask_nii,U_nii,V_nii,W_nii,phi,phiInv):
        self.mask_nii=mask_nii
        self.U_nii=U_nii
        self.V_nii=V_nii
        self.W_nii=W_nii
        self.phi=phi
        self.phiInv=phiInv
        self.affine=U_nii.affine

        ua, va, wa = self.phi(0, 0, 0)
        uh, vh, wh = self.phi(abs(self.affine[0][0]), self.affine[0][0], 0)

        self.delU = abs(uh - ua) / 10
        self.delV = abs(vh - va) / 10

    def connectedInds(self,stop_mask_nii,stopval,seed,const_coord):
        stop=stop_mask_nii.get_fdata()
        U = self.U_nii.get_fdata()
        V = self.V_nii.get_fdata()


        #lines = [] #all the streamlines from seed
        allinds=[]

        #positive streamlines
        line=[]
        u1, v1, w1 = self.phi(seed[0], seed[1], seed[2])
        l = 0
        v1p = v1
        u1p = u1
        if const_coord == 'u':
            v1n = v1p + self.delV
            lmax = (np.nanmax(V) - v1n) / self.delV + 1
        else:
            u1n = u1p + self.delU
            lmax = (np.nanmax(U) - u1n) / self.delU + 1
        while l < lmax:
            xp, yp, zp = self.phiInv(u1p, v1p, w1)
            worlds = np.asarray([xp, yp, zp])
            inds = toInds(self.mask_nii, [worlds])
            inds = np.asarray(inds[0])
            inds = inds.round()
            inds = inds.astype(int)
            condition=inds[0]>=stop.shape[0] or inds[1]>=stop.shape[1] or inds[2]>=stop.shape[2]
            if condition == True:
                break
            #add line to keep inds within image size.
            if (stop[inds[0], inds[1], inds[2]] != stopval):
                #print(stop[inds[0], inds[1], inds[2]])
                break
            else:
                line.append(worlds)
                allinds.append(list(inds)) if list(inds) not in allinds else allinds
                if const_coord == 'u':
                    v1p = v1p + self.delV
                else:
                    u1p = u1p + self.delU
                l = l + 1
        #if (len(line) > 0):
            #line.append(line)
        line.reverse()
        allinds.reverse()
        nline = []
        l = 0
        v1n = v1
        u1n = u1
        if const_coord == 'u':
            v1n = v1n - self.delV
            lmax = (-np.nanmin(V) + v1n) / self.delV + 1
        else:
            u1n = u1n - self.delU
            lmax = (-np.nanmin(U) + u1n) / self.delU + 1
        while l < lmax:
            xn, yn, zn = self.phiInv(u1n, v1n, w1)
            worlds = np.asarray([xn, yn, zn])
            inds = toInds(self.mask_nii, [worlds])
            inds = np.asarray(inds[0])
            inds = inds.round()
            inds = inds.astype(int)
            condition=inds[0]>=stop.shape[0] or inds[1]>=stop.shape[1] or inds[2]>=stop.shape[2]
            if condition == True:
                break
            if (stop[inds[0], inds[1], inds[2]] != stopval):
                # pass
                break
            else:
                line.append(worlds)
                allinds.append(list(inds)) if list(inds) not in allinds else allinds
                if const_coord == 'u':
                    v1n = v1n - self.delV
                else:
                    u1n = u1n - self.delU
                l = l + 1
        #if (len(nline) > 0):
            #line.append(nline)
        return allinds,line

class tracking:
    def __init__(self,peaks,stopping_criterion,seeds,affine,
                 graddev=None,sphere=default_sphere.subdivide(),seed_density=1):
        self.peaks=peaks
        self.stopping_criterion=stopping_criterion
        self.seeds=seeds
        self.affine=affine
        self.graddev=graddev
        self.sphere=sphere
        self.streamlines=[]
        self.NpointsPerLine=[]
        self.seed_density=seed_density

        #adding some key direction to the sphere
        self.sphere.vertices = np.append(self.sphere.vertices, [[1, 0, 0]])
        self.sphere.vertices = np.append(self.sphere.vertices, [[-1, 0, 0]])
        self.sphere.vertices = np.append(self.sphere.vertices, [[0, 1, 0]])
        self.sphere.vertices = np.append(self.sphere.vertices, [[0, -1, 0]])
        self.sphere.vertices = self.sphere.vertices.reshape([-1, 3])

    def localTracking(self):
        if self.graddev is None:


            #multiply by the jacobian (zero out z-direction)
            graddev=np.zeros([3,3])
            graddev[0, 0] = 1
            graddev[1, 1] = 1
            graddev[2, 2] = 1

            new_peak_dirsp = np.einsum('ab,ijkvb->aijkv',
                                      graddev, self.peaks.peak_dirs)
            shape=new_peak_dirsp.shape
            new_peak_dirsp=new_peak_dirsp.reshape(3,-1)
            new_peak_dirs=copy.deepcopy(new_peak_dirsp)
            for i in range(0,new_peak_dirs.shape[-1]):
                norm=np.linalg.norm(new_peak_dirsp[:, i])
                if norm!=0:
                    new_peak_dirs[:,i]=new_peak_dirsp[:,i]/norm
            new_peak_dirs = new_peak_dirs.reshape(shape)
            new_peak_dirs = np.moveaxis(new_peak_dirs,0,-1)
            new_peak_dirs = new_peak_dirs.reshape([-1, self.peaks.peak_indices.shape[-1], 3])
            #update self.peaks.peak_indices
            peak_indices=np.zeros(self.peaks.peak_indices.shape)
            peak_indices=peak_indices.reshape([-1,self.peaks.peak_indices.shape[-1]])

            for i in range(0, peak_indices.shape[0]):
                for k in range(0, self.peaks.peak_indices.shape[-1]):
                    peak_indices[i, k] = self.sphere.find_closest(new_peak_dirs[i, k, :])

            self.peaks.peak_indices = peak_indices.reshape(self.peaks.peak_indices.shape)



            streamlines_generator=LocalTracking(self.peaks,
                                           self.stopping_criterion,
                                           self.seeds,
                                           self.affine,
                                           step_size=abs(self.affine[0,0]/6))
            self.streamlines=Streamlines(streamlines_generator)


        else:
            shape=self.graddev.shape
            self.graddev=self.graddev.reshape(shape[0:3]+ (3, 3), order='F')
            #self.graddev[:, :, :, :, 2] = 0
            #self.graddev[:, :, :, 2, :] = 0
            #self.graddev[:, :, :, 2, 2] = -1

            self.graddev=(self.graddev.reshape([-1,3,3])+np.eye(3))
            self.graddev=self.graddev.reshape(shape[0:3]+(3,3))

            #multiply by the jacobian
            new_peak_dirsp = np.einsum('ijkab,ijkvb->aijkv',
                                      self.graddev, self.peaks.peak_dirs)
            shape=new_peak_dirsp.shape
            new_peak_dirsp=new_peak_dirsp.reshape(3,-1)
            new_peak_dirs=copy.deepcopy(new_peak_dirsp)
            for i in range(0,new_peak_dirs.shape[-1]):
                norm=np.linalg.norm(new_peak_dirsp[:, i])
                if norm!=0:
                    new_peak_dirs[:,i]=new_peak_dirsp[:,i]/norm
            new_peak_dirs = new_peak_dirs.reshape(shape)
            new_peak_dirs = np.moveaxis(new_peak_dirs,0,-1)
            new_peak_dirs = new_peak_dirs.reshape([-1, self.peaks.peak_indices.shape[-1], 3])
            #update self.peaks.peak_indices
            peak_indices=np.zeros(self.peaks.peak_indices.shape)
            peak_indices=peak_indices.reshape([-1,self.peaks.peak_indices.shape[-1]])

            for i in range(0, peak_indices.shape[0]):
                for k in range(0, self.peaks.peak_indices.shape[-1]):
                    peak_indices[i, k] = self.sphere.find_closest(new_peak_dirs[i, k, :])

            self.peaks.peak_indices = peak_indices.reshape(self.peaks.peak_indices.shape)

            streamlines_generator= LocalTracking(self.peaks,
                                             self.stopping_criterion,
                                             self.seeds,
                                             self.affine,
                                             step_size=self.affine[0, 0]/6)

            self.streamlines=Streamlines(streamlines_generator)
            self.NpointsPerLine=pointsPerLine(self.streamlines)


    def plot(self):
        if has_fury:
            # Prepare the display objects.
            color = colormap.line_colors(self.streamlines)

            streamlines_actor = actor.line(self.streamlines,
                                           colormap.line_colors(self.streamlines))

            # Create the 3D display.
            scene = window.Scene()
            scene.add(streamlines_actor)

            # Save still images for this static example. Or for interactivity use
            window.show(scene)


class unfoldStreamlines:
    def __init__(self,nativeStreamlines,unfoldStreamlines,nppl,uppl,coords):
        self.nativeStreamlines=nativeStreamlines
        self.unfoldStreamlines=unfoldStreamlines
        self.coords=coords
        self.streamlinesFromUnfold=[]
        self.streamlinesFromNative=[]
        self.nppl=nppl
        self.uppl=uppl

    def moveStreamlines2Native(self):
        #assuming that the coordinates are in world coordinates.

        points=np.asarray([self.coords.Ua,
                           self.coords.Va,
                           self.coords.Wa]).transpose()

        allLines=self.unfoldStreamlines.get_data()
        x = griddata(points, self.coords.X, allLines)
        y = griddata(points, self.coords.Y, allLines)
        z = griddata(points, self.coords.Z, allLines)

        allLines=np.asarray([x,y,z]).T

        self.streamlinesFromUnfold=allLines2Lines(allLines,
                                                  self.uppl)


    def moveStreamlines2Unfold(self):

        points = np.asarray([self.coords.X,
                             self.coords.Y,
                             self.coords.Z]).transpose()

        allLines = self.nativeStreamlines.get_data()
        ua = griddata(points, self.coords.Ua, allLines)
        va = griddata(points, self.coords.Va, allLines)
        wa = griddata(points, self.coords.Wa, allLines)

        allLines = np.asarray([ua, va, wa]).T

        self.streamlinesFromNative = allLines2Lines(allLines,
                                                    self.nppl)

