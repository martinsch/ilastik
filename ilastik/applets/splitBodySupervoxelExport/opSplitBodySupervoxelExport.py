###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# In addition, as a special exception, the copyright holders of
# ilastik give you permission to combine ilastik with applets,
# workflows and plugins which are not covered under the GNU
# General Public License.
#
# See the LICENSE file for details. License information is also available
# on the ilastik web site at:
#		   http://ilastik.org/license.html
###############################################################################
from functools import partial
import numpy
import h5py
from lazyflow.request import Request
from lazyflow.graph import Operator, InputSlot, OutputSlot, OperatorWrapper
from lazyflow.operators import OpCompressedCache, OpVigraLabelVolume, OpFilterLabels, OpSelectLabel, OpMaskedSelect, OpDtypeView
from lazyflow.operators.ioOperators import OpH5WriterBigDataset
from lazyflow.operators.opReorderAxes import OpReorderAxes

from lazyflow.utility import PathComponents
from ilastik.utility import bind, log_exception
from ilastik.applets.splitBodyPostprocessing.opSplitBodyPostprocessing import OpAccumulateFragmentSegmentations, OpMaskedWatershed

import logging
logger = logging.getLogger(__name__)

class OpSplitBodySupervoxelExport(Operator):

    DatasetInfos = InputSlot(level=1) # Used to extract the other datasets from the segmentation file.
    WorkingDirectory = InputSlot()
    
    RawData = InputSlot() # (Display only)
    InputData = InputSlot() # The membrane probabilities
    RavelerLabels = InputSlot()
    Supervoxels = InputSlot()
    AnnotationBodyIds = InputSlot() # The list of bodies actually edited
                                    # (Must be connected to ensure that setupOutputs will be 
                                    #   called resize the multi-slots when necessary)

    # For these multislots, N = number of raveler bodies that were edited
    EditedRavelerBodies = OutputSlot(level=1)
    MaskedSupervoxels = OutputSlot(level=1)
    RelabeledSupervoxels = OutputSlot(level=1)
    FilteredMaskedSupervoxels = OutputSlot(level=1)
    HoleFilledSupervoxels = OutputSlot(level=1)
    
    FinalSupervoxels = OutputSlot()
    SupervoxelMapping = OutputSlot()

    # RavelerLabels ------>------------------------------------------------------------------------------------------------------------------------------------------------>------------------------------------------------------------------------------------------------------------------------------------
    #                      \                                                                                                                                                \                                                                                                                                   \
    #                       \       Supervoxels --                   MaskedSupervoxels                                                                                       \                                                                                                                                   \
    #                        \                    \                 /                                                                                                         \                                                                                                                                   \
    # AnnotationBodyIds ----> opSelectLabel[n] --> opMaskedSelect[n] --> opRelabelMaskedSupervoxels[n] --> opRelabeledMaskedSupervoxelsCaches[n] --> opSmallLabelFilter[n] --> opMaskedWatershed[n] --> opMaskedWatershedCaches[n] --> opRelabelMergedSupervoxels[n] --> opRelabeledMergedSupervoxelsCaches[n] --> opAccumulateFinalImage --> opFinalCache --> FinalSupervoxels
    #                                         \                                                                                                                               /                                                                                                                               \                          \
    #                                          -------------------------------------------------------------------------------------------------------------------------------                                                                                                                                 RelabeledSupervoxels       (SupervoxelMapping)

    def __init__(self, *args, **kwargs):
        super( OpSplitBodySupervoxelExport, self ).__init__(*args, **kwargs)

        # HACK: Be sure that the output slots are resized if the raveler body list changes
        self.AnnotationBodyIds.notifyDirty( bind(self._setupOutputs) )

        # Prepare a set of OpSelectLabels for easy access to raveler object masks
        self._opSelectLabel = OperatorWrapper( OpSelectLabel, parent=self, broadcastingSlotNames=['Input'] )
        self._opSelectLabel.Input.connect( self.RavelerLabels )
        self.EditedRavelerBodies.connect( self._opSelectLabel.Output )

        # Mask in the body of interest
        self._opMaskedSelect = OperatorWrapper( OpMaskedSelectUint32, parent=self, broadcastingSlotNames=['Input'] )
        self._opMaskedSelect.Input.connect( self.Supervoxels )
        self._opMaskedSelect.Mask.connect( self._opSelectLabel.Output )
        self.MaskedSupervoxels.connect( self._opMaskedSelect.Output )        

        # Must run CC before filter, to ensure that discontiguous labels can't avoid the filter.
        self._opRelabelMaskedSupervoxels = OperatorWrapper( OpVigraLabelVolume, parent=self )
        self._opRelabelMaskedSupervoxels.Input.connect( self._opMaskedSelect.Output )
        
        self._opRelabeledMaskedSupervoxelCaches = OperatorWrapper( OpCompressedCache, parent=self )
        self._opRelabeledMaskedSupervoxelCaches.Input.connect( self._opRelabelMaskedSupervoxels.Output )

        # Filter out the small CC to eliminate tiny pieces of supervoxels that overlap the mask boundaries
        self._opSmallLabelFilter = OperatorWrapper( OpFilterLabels, parent=self, broadcastingSlotNames=['MinLabelSize'] )
        self._opSmallLabelFilter.MinLabelSize.setValue( 10 )
        self._opSmallLabelFilter.Input.connect( self._opRelabeledMaskedSupervoxelCaches.Output )

        self._opSmallLabelFilterCaches = OperatorWrapper( OpCompressedCache, parent=self )
        self._opSmallLabelFilterCaches.Input.connect( self._opSmallLabelFilter.Output )
        self.FilteredMaskedSupervoxels.connect( self._opSmallLabelFilterCaches.Output )

        # Re-fill the holes left by the filter using region growing (with a mask)
        self._opMaskedWatersheds =  OperatorWrapper( OpMaskedWatershed, parent=self )
        self._opMaskedWatersheds.Input.connect( self.InputData )
        self._opMaskedWatersheds.Mask.connect( self._opSelectLabel.Output )
        self._opMaskedWatersheds.Seeds.connect( self._opSmallLabelFilterCaches.Output )

        # Cache is necessary because it ensures that the entire volume is used for watershed.
        self._opMaskedWatershedCaches = OperatorWrapper( OpCompressedCache, parent=self )
        self._opMaskedWatershedCaches.Input.connect( self._opMaskedWatersheds.Output )
        self.HoleFilledSupervoxels.connect( self._opMaskedWatershedCaches.Output )

        # Relabel the supervoxels in the mask to ensure contiguous supervoxels (after mask) and consecutive labels
        self._opRelabelMergedSupervoxels = OperatorWrapper( OpVigraLabelVolume, parent=self )
        self._opRelabelMergedSupervoxels.Input.connect( self._opMaskedWatershedCaches.Output )
        
        self._opRelabeledMergedSupervoxelCaches = OperatorWrapper( OpCompressedCache, parent=self )
        self._opRelabeledMergedSupervoxelCaches.Input.connect( self._opRelabelMergedSupervoxels.Output )
        self.RelabeledSupervoxels.connect( self._opRelabeledMergedSupervoxelCaches.Output )

        self._opAccumulateFinalImage = OpAccumulateFragmentSegmentations( parent=self )
        self._opAccumulateFinalImage.RavelerLabels.connect( self.RavelerLabels )
        self._opAccumulateFinalImage.FragmentSegmentations.connect( self._opRelabeledMergedSupervoxelCaches.Output )
        
        self._opFinalCache = OpCompressedCache( parent=self )
        self._opFinalCache.Input.connect( self._opAccumulateFinalImage.Output )
        self.FinalSupervoxels.connect( self._opFinalCache.Output )
        self.SupervoxelMapping.connect( self._opAccumulateFinalImage.Mapping )
        
    def setupOutputs(self):
        raveler_bodies = self.AnnotationBodyIds.value
        num_bodies = len(raveler_bodies)

        # Map raveler body ids to the subslots that need them.
        self._opSelectLabel.SelectedLabel.resize( num_bodies )
        for index, raveler_body_id in enumerate(raveler_bodies):
            self._opSelectLabel.SelectedLabel[index].setValue( raveler_body_id )

    def execute(self, slot, subindex, roi, result):
        assert False, "Can't execute slot {}.  All slots should be connected to internal operators".format( slot.name )

    def propagateDirty(self, slot, subindex, roi):
        # If anything is dirty, the entire output is dirty
        self.FinalSupervoxels.setDirty()

    def exportFinalSupervoxels(self, outputPath, axisorder, progressCallback=None):
        """
        Executes the export process within a request.
        The (already-running) request is returned, in case you want to wait for it or monitor its progress.
        """
        assert self.FinalSupervoxels.ready(), "Can't export yet: The final segmentation isn't ready!"

        logger.info("Starting Final Segmentation Export...")
        
        opTranspose = OpReorderAxes( parent=self )
        opTranspose.AxisOrder.setValue( axisorder )
        opTranspose.Input.connect( self.FinalSupervoxels )
        
        f = h5py.File(outputPath, 'w')
        opExporter = OpH5WriterBigDataset(parent=self)
        opExporter.hdf5File.setValue( f )
        opExporter.hdf5Path.setValue( 'stack' )
        opExporter.Image.connect( opTranspose.Output )
        if progressCallback is not None:
            opExporter.progressSignal.subscribe( progressCallback )
        
        req = Request( partial(self._runExporter, opExporter) )

        def cleanOps():
            opExporter.cleanUp()
            opTranspose.cleanUp()
        
        def handleFailed( exc, exc_info ):
            cleanOps()        
            f.close()
            msg = "Final Supervoxel export FAILED due to the following error:\n{}".format( exc )
            log_exception( logger, msg, exc_info )

        def handleFinished( result ):
            # Generate the mapping transforms dataset
            mapping = self._opAccumulateFinalImage.Mapping.value
            num_labels = mapping.keys()[-1][1]
            transform = numpy.zeros( shape=(num_labels, 2), dtype=numpy.uint32 )
            for (start, stop), body_id in mapping.items():
                for supervoxel_label in range(start, stop):
                    transform[supervoxel_label][0] = supervoxel_label
                    if body_id == -1:
                        # Special case: -1 means "identity transform" for this supervoxel
                        # (Which is really an untouched raveler body)
                        transform[supervoxel_label][1] = supervoxel_label
                    else:
                        transform[supervoxel_label][1] = body_id

            # Save the transform before closing the file
            f.create_dataset('transforms', data=transform)

            # Copy all other datasets from the original segmentation file.
            ravelerSegmentationInfo = self.DatasetInfos[2].value
            pathComponents = PathComponents(ravelerSegmentationInfo.filePath, self.WorkingDirectory.value)
            with h5py.File(pathComponents.externalPath, 'r') as originalFile:
                for k,dset in originalFile.items():
                    if k not in ['transforms', 'stack']:
                        f.copy(dset, k)
            
            try:
                cleanOps()
                logger.info("FINISHED Final Supervoxel Export")
            finally:
                f.close()

        def handleCancelled():
            cleanOps()
            f.close()
            logger.info( "Final Supervoxel export was cancelled!" )

        req.notify_failed( handleFailed )
        req.notify_finished( handleFinished )
        req.notify_cancelled( handleCancelled )
        
        req.submit()
        return req # Returned in case the user wants to cancel it.

    def _runExporter(self, opExporter):
        # Trigger the export
        success = opExporter.WriteImage.value
        assert success
        return success

class OpMaskedSelectUint32(Operator):
    # Upstream watershed is output as signed int32.
    # We must produce uint32 for the label op.
    Input = InputSlot()
    Mask = InputSlot()
    Output = OutputSlot()
    
    def __init__(self, *args, **kwargs):
        super( OpMaskedSelectUint32, self ).__init__( *args, **kwargs )
        
        self._opMaskedSelect = OpMaskedSelect( parent=self )
        self._opMaskedSelect.Input.connect( self.Input )
        self._opMaskedSelect.Mask.connect( self.Mask )
        
        self._opConvertDtype = OpDtypeView( parent=self )
        self._opConvertDtype.Input.connect( self._opMaskedSelect.Output )
        self._opConvertDtype.OutputDtype.setValue( numpy.uint32 )
        self.Output.connect( self._opConvertDtype.Output )

    def setupOutputs(self):
        pass
    
    def execute(self, slot, subindex, roi, result):
        pass

    def propagateDirty(self, slot, subindex, roi):
        pass



