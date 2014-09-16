import numpy
import vigra
import h5py
import os

def iterateImagesFromFolder(path,timesteps = None):
    #for working with data sets from folders, not relevant in ilastik/pgmlink
    Files = [file for file in os.listdir(path) if file.endswith(".h5")]
    Files.sort()
    if timesteps is None:
        timesteps = range(len(Files)-1)
        
    for t in timesteps:
        f = h5py.File(path+"/"+Files[t],'r')
        img = f["segmentation/labels"].__array__()
        if "Moves" in f["tracking"]:
            moves = f["tracking/Moves"].__array__()
        else:
            moves = []
        yield img,moves

def estimateCovarianceFromFolder(path,timesteps = None):
    previous_rc = None
    
    Samples = []
    
    for img,moves in iterateImagesFromFolder(path,timesteps):
        img = img.view(vigra.VigraArray)
        img.axistags = vigra.defaultAxistags("xyz")
        FA = vigra.analysis.extractRegionFeatures(img.astype(numpy.float32),img.astype(numpy.uint32),features = "RegionCenter")
        rc = FA["RegionCenter"]

        if previous_rc is not None:#not first timestep
            num_moves = moves.shape[0]
            for i in xrange(num_moves):
                m1,m2 = moves[i,0],moves[i,1]
                Samples.append(rc[m2]-previous_rc[m1])
        previous_rc = rc
    dim = img.shape.__len__()
    n = len(Samples)
    mu = sum(Samples,numpy.zeros((1,dim)))/n
        
    Q = numpy.zeros((dim,dim))
    Q = sum((numpy.transpose(x-mu)*(x-mu) for x in Samples),Q)/(n-1)
    return mu,Q

def estimateVarianceFromFolder(path):
    previous_rc = None
    
    for img,moves in iterateImagesFromFolder(path):
        img = img.view(vigra.VigraArray)
        img.axistags = vigra.defaultAxistags("xyz")
        FA = vigra.analysis.extractRegionFeatures(img.astype(numpy.float32),img.astype(numpy.uint32),features = "RegionCenter")
        rc = FA["RegionCenter"]
        s = 0.
        total = 0.
        if previous_rc is not None:#not first timestep
            num_moves = moves.shape[0]
            for i in xrange(num_moves):
                m1,m2 = moves[i,0],moves[i,1]
                s+=numpy.linalg.norm(rc[m2]-previous_rc[m1])
            total+=num_moves
            
        previous_rc = rc
    return s/total
    
if __name__ == "__main__":
    print estimateCovarianceFromFolder("/home/bheuer/Documents/ilastik_files/some_tracking_result")
    print estimateVarianceFromFolder("/home/bheuer/Documents/ilastik_files/drosophila_ilastik_export")
    print estimateCovarianceFromFolder("/home/bheuer/Documents/ilastik_files/drosophila_ilastik_export")