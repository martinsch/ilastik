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
from lazyflow.graph import Operator, InputSlot, OutputSlot, OperatorWrapper

from ilastik.utility.operatorSubView import OperatorSubView
from lazyflow.operators import OpBlockedSparseLabelArray, OpValueCache, OpTrainRandomForestBlocked, \
                               OpPredictRandomForest, OpSlicedBlockedArrayCache, OpMultiArraySlicer2, \
                               OpPrecomputedInput, Op50ToMulti, OpArrayPiper, OpMultiArrayStacker

try:
    from context.operators.contextVariance import OpContextVariance
except ImportError:
    import os.path
    import warnings
    warnings.warn("Module 'context' not found. "
          "Module %s is not fully functional." % os.path.basename(__file__)) 
                               
class OpAutocontextClassification( Operator ):
    """
    Top-level operator for classification with autocontext
    """
    
    name = "OpAutocontextClassification"
    category = "Top-level"
    
    # Graph inputs
    
    InputImages = InputSlot(level=1) # Original input data.  Used for display only.

    LabelsAllowedFlags = InputSlot(stype='bool', level=1) # Specifies which images are permitted to be labeled 
    LabelInputs = InputSlot(optional = True, level=1) # Input for providing label data from an external source

    FeatureImages = InputSlot(level=1) # Computed feature images (each channel is a different feature)
    CachedFeatureImages = InputSlot(level=1) # Cached feature data.
    
    AutocontextFeatureIds = InputSlot()
    AutocontextScales = InputSlot()
    AutocontextIterations = InputSlot()

    FreezePredictions = InputSlot(stype='bool')

    PredictionsFromDisk = InputSlot(optional=True, level=1)

    PixelOnlyPredictions = OutputSlot(level=1) # Predictions based only on pixel features
    PixelOnlyPredictionChannels = OutputSlot(level=2)

    PredictionProbabilities = OutputSlot(level=1) # Classification predictions
    PredictionProbabilityChannels = OutputSlot(level=2) # Classification predictions, enumerated by channel
    
    MaxLabelValue = OutputSlot()
    LabelImages = OutputSlot(level=1) # Labels from the user
    NonzeroLabelBlocks = OutputSlot(level=1) # A list if slices that contain non-zero label values
    
    # Holds the chain. Level is set to 1, because it's connected to a OpMulti 
    # Unlike all of our other output multi-slots, the classifiers output is not specific to a single lane.
    # It is shared by all lanes.  The OperatorSubView class respects the magic 'nonlane' meta attribute for this case.
    Classifiers = OutputSlot(level=1, nonlane=True)

    CachedPredictionProbabilities = OutputSlot(level=1) # Classification predictions (via a cache)
    CachedPixelPredictionProbabilities = OutputSlot(level=1)
    
    # GUI-only (not part of the pipeline, but saved to the project)
    LabelNames = OutputSlot()
    LabelColors = OutputSlot()
    
    def __init__( self, *args, **kwargs ):
        """
        Instantiate all internal operators and connect them together.
        """
        super(OpAutocontextClassification, self).__init__(*args, **kwargs)

        # Our internal graph needs to be built (or re-built) when the number of iterations is set.
        self.AutocontextIterations.notifyDirty(self.setupOperators)

        self.LabelNames.setValue( [] ) # Default
        self.LabelColors.setValue( [] ) # Default

    def setupOperators(self, *args, **kwargs):
        self.FreezePredictions.setValue(True) # Default
        
        # Create internal operators
        # Explicitly wrapped:
        self.opInputShapeReader = OperatorWrapper( OpShapeReader, parent=self )
        self.opLabelArray = OperatorWrapper( OpBlockedSparseLabelArray, parent=self  )

        self.predictors = []
        self.prediction_caches = []
        self.prediction_caches_gui = []
        
        #FIXME: we should take it from the input slot
        niter = self.AutocontextIterations.value
        
        for i in range(niter):
            predict = OperatorWrapper( OpPredictRandomForest, parent=self )
            prediction_cache = OperatorWrapper( OpSlicedBlockedArrayCache, parent=self ) 
            prediction_cache_gui = OperatorWrapper( OpSlicedBlockedArrayCache, parent=self )
            
            self.predictors.append(predict)
            self.prediction_caches.append(prediction_cache)
            self.prediction_caches_gui.append(prediction_cache_gui)
        
        #We only display the last prediction layer
         
        self.precomputed_predictions = OperatorWrapper( OpPrecomputedInput, parent=self)
        self.precomputed_predictions_gui = OperatorWrapper( OpPrecomputedInput, parent=self)
        
        #Display pixel-only predictions to compare
        self.precomputed_predictions_pixel = OperatorWrapper( OpPrecomputedInput, parent=self)
        self.precomputed_predictions_pixel_gui = OperatorWrapper( OpPrecomputedInput, parent=self) 

        # NOT wrapped
        self.opMaxLabel = OpMaxValue(parent=self)
        self.trainers = []
        for i in range(niter):
            opTrain = OpTrainRandomForestBlocked( parent=self )
            self.trainers.append(opTrain)

        # Set up label cache shape input
        self.opInputShapeReader.Input.connect( self.InputImages )
        self.opLabelArray.inputs["shape"].connect( self.opInputShapeReader.OutputShape )

        # Set up other label cache inputs
        self.LabelInputs.connect( self.InputImages )
        self.opLabelArray.inputs["Input"].connect( self.LabelInputs )
        self.opLabelArray.inputs["eraser"].setValue(100)
                
        # Initialize the delete input to -1, which means "no label".
        # Now changing this input to a positive value will cause label deletions.
        # (The deleteLabel input is monitored for changes.)
        self.opLabelArray.inputs["deleteLabel"].setValue(-1)
        
        # Find the highest label in all the label images
        self.opMaxLabel.Inputs.connect( self.opLabelArray.outputs['maxLabel'] )

        # Setup autocontext features
        self.autocontextFeatures = []
        self.autocontextFeaturesMulti = []
        self.autocontext_caches = []
        self.featureStackers = []
        
        for i in range(niter-1):
            features = createAutocontextFeatureOperators(self, True)
            self.autocontextFeatures.append(features)
            opMulti = OperatorWrapper( Op50ToMulti, parent=self)
            self.autocontextFeaturesMulti.append(opMulti)
            opStacker = OperatorWrapper( OpMultiArrayStacker, parent=self)
            opStacker.inputs["AxisFlag"].setValue("c")
            opStacker.inputs["AxisIndex"].setValue(3)
            self.featureStackers.append(opStacker)
            autocontext_cache = OperatorWrapper( OpSlicedBlockedArrayCache, parent=self )
            self.autocontext_caches.append(autocontext_cache)
        
        # connect the features to predictors
        for i in range(niter-1):
            for ifeat, feat in enumerate(self.autocontextFeatures[i]):
                feat.inputs['Input'].connect( self.prediction_caches[i].Output)
                print "Multi: Connecting an output", "Input%.2d"%(ifeat)
                self.autocontextFeaturesMulti[i].inputs["Input%.2d"%(ifeat)].connect(feat.outputs["Output"])
            # connect the pixel features to the same multislot
            print "Multi: Connecting an output", "Input%.2d"%(len(self.autocontextFeatures[i]))
            self.autocontextFeaturesMulti[i].inputs["Input%.2d"%(len(self.autocontextFeatures[i]))].connect(self.CachedFeatureImages)
            # stack the autocontext features with pixel features
            self.featureStackers[i].inputs["Images"].connect(self.autocontextFeaturesMulti[i].outputs["Outputs"])
            # cache the stacks
            self.autocontext_caches[i].inputs["Input"].connect(self.featureStackers[i].outputs["Output"])                                                  
            self.autocontext_caches[i].inputs["fixAtCurrent"].setValue(False)

        ##
        # training
        ##
        
        for op in self.trainers:
            op.inputs['Labels'].connect(self.opLabelArray.outputs["Output"])
            op.inputs["nonzeroLabelBlocks"].connect(self.opLabelArray.outputs["nonzeroBlocks"])
            op.inputs['fixClassifier'].setValue(False)
        # Connect the first training operator - just pixel features
        self.trainers[0].inputs['Images'].connect(self.CachedFeatureImages)
        # Connect other training operators - stacked pixel and autocontext features
        for i in range(1, niter):
            self.trainers[i].inputs["Images"].connect(self.featureStackers[i-1].outputs["Output"])
        
        ##
        # prediction
        ##
        
        # The classifier is cached here to allow serializers to force in a pre-calculated classifier...
        self.classifiers = []
        self.classifier_caches = []
        
        for i in range(niter):
            self.classifiers.append(self.trainers[i].outputs['Classifier'])
            cache = OpValueCache(parent=self)
            cache.inputs["Input"].connect(self.trainers[i].outputs['Classifier'])
            self.classifier_caches.append(cache)
        
        for i in range(niter):        
            self.predictors[i].inputs['Classifier'].connect(self.classifier_caches[i].outputs["Output"])
            self.predictors[i].inputs['LabelsCount'].connect(self.opMaxLabel.Output)
            
            self.prediction_caches[i].inputs["fixAtCurrent"].setValue(False)
            self.prediction_caches[i].inputs["Input"].connect(self.predictors[i].PMaps)
            
            
            self.prediction_caches_gui[i].name = "PredictionCache"
            self.prediction_caches_gui[i].inputs["fixAtCurrent"].connect( self.FreezePredictions )
            self.prediction_caches_gui[i].inputs["Input"].connect(self.predictors[i].PMaps)
        
        self.predictors[0].inputs['Image'].connect(self.CachedFeatureImages)
        for i in range(1, niter):
            self.predictors[i].inputs['Image'].connect(self.autocontext_caches[i-1].outputs["Output"])
            
        # The serializer uses these operators to provide prediction data directly from the project file
        # if the predictions haven't become dirty since the project file was opened.
        self.precomputed_predictions.SlowInput.connect( self.prediction_caches[-1].Output )
        self.precomputed_predictions.PrecomputedInput.connect( self.PredictionsFromDisk )
        
        self.precomputed_predictions_pixel.SlowInput.connect( self.prediction_caches[0].Output )
        self.precomputed_predictions_pixel.PrecomputedInput.connect( self.PredictionsFromDisk )

        # !!! here we can change which prediction step we show:
        self.precomputed_predictions_gui.SlowInput.connect( self.prediction_caches_gui[-1].Output )
        self.precomputed_predictions_gui.PrecomputedInput.connect( self.PredictionsFromDisk )
        self.precomputed_predictions_pixel_gui.SlowInput.connect( self.prediction_caches_gui[0].Output)
        self.precomputed_predictions_pixel_gui.PrecomputedInput.connect( self.PredictionsFromDisk )
        

        # Connect our internal outputs to our external outputs
        self.LabelImages.connect(self.opLabelArray.Output)
        self.MaxLabelValue.connect( self.opMaxLabel.Output )
        self.NonzeroLabelBlocks.connect(self.opLabelArray.nonzeroBlocks)
        self.PixelOnlyPredictions.connect(self.predictors[0].PMaps)
        self.PredictionProbabilities.connect(self.predictors[-1].PMaps)
        self.CachedPredictionProbabilities.connect(self.precomputed_predictions.Output)
        self.CachedPixelPredictionProbabilities.connect(self.precomputed_predictions_pixel.Output)
        
        self.multi = Op50ToMulti(parent=self)
        for i in range(niter):
            self.multi.inputs["Input%.2d"%i].connect(self.classifier_caches[i].outputs["Output"])
        
        self.Classifiers.connect( self.multi.outputs["Outputs"] )

        def inputResizeHandler( slot, oldsize, newsize ):
            if ( newsize == 0 ):
                self.LabelImages.resize(0)
                self.NonzeroLabelBlocks.resize(0)
                self.PixelOnlyPredictions.resize(0)
                self.PredictionProbabilities.resize(0)
                self.CachedPredictionProbabilities.resize(0)
                self.CachedPixelPredictionProbabilities.resize(0)
                
        self.InputImages.notifyResized( inputResizeHandler )

        # Check to make sure the non-wrapped operators stayed that way.
        assert self.opMaxLabel.Inputs.operator == self.opMaxLabel
        for i in range(niter):
            assert self.trainers[0].Images.operator == self.trainers[0]
        #assert self.opTrain.Images.operator == self.opTrain
        
        # Also provide each prediction channel as a separate layer (for the GUI)
        self.opPredictionSlicer = OperatorWrapper( OpMultiArraySlicer2, parent=self) 
        self.opPredictionSlicer.Input.connect( self.precomputed_predictions_gui.Output )
        self.opPredictionSlicer.AxisFlag.setValue('c')
        self.PredictionProbabilityChannels.connect( self.opPredictionSlicer.Slices )
        
        self.opPixelPredictionSlicer = OperatorWrapper( OpMultiArraySlicer2, parent=self)
        self.opPixelPredictionSlicer.Input.connect( self.precomputed_predictions_pixel_gui.Output)
        self.opPixelPredictionSlicer.AxisFlag.setValue('c')
        self.PixelOnlyPredictionChannels.connect( self.opPixelPredictionSlicer.Slices )

        def handleNewInputImage( multislot, index, *args ):
            def handleInputReady(slot):
                self.setupCaches( multislot.index(slot) )
            multislot[index].notifyReady(handleInputReady)
                
        self.InputImages.notifyInserted( handleNewInputImage )

        # All input multi-slots should be kept in sync
        # Output multi-slots will auto-sync via the graph
        multiInputs = filter( lambda s: s.level >= 1, self.inputs.values() )
        for s1 in multiInputs:
            for s2 in multiInputs:
                if s1 != s2:
                    def insertSlot( a, b, position, finalsize ):
                        a.insertSlot(position, finalsize)
                    s1.notifyInserted( partial(insertSlot, s2 ) )
                    
                    def removeSlot( a, b, position, finalsize ):
                        a.removeSlot(position, finalsize)
                    s1.notifyRemoved( partial(removeSlot, s2 ) )
        
    def setupCaches(self, imageIndex):
        numImages = len(self.InputImages)
        inputSlot = self.InputImages[imageIndex]
#        # Can't setup if all inputs haven't been set yet.
#        if numImages != len(self.FeatureImages) or \
#           numImages != len(self.CachedFeatureImages):
#            return
#        
#        self.LabelImages.resize(numImages)
        self.LabelInputs.resize(numImages)

        # Special case: We have to set up the shape of our label *input* according to our image input shape
        channelIndex = self.InputImages[imageIndex].meta.axistags.index('c')
        shapeList = list(self.InputImages[imageIndex].meta.shape)
        shapeList[channelIndex] = 1
        self.LabelInputs[imageIndex].meta.shape = tuple(shapeList)
        self.LabelInputs[imageIndex].meta.axistags = inputSlot.meta.axistags

        # Set the blockshapes for each input image separately, depending on which axistags it has.
        axisOrder = [ tag.key for tag in inputSlot.meta.axistags ]
        
        ## Label Array blocks
        blockDims = { 't' : 1, 'x' : 32, 'y' : 32, 'z' : 5, 'c' : 1 }
        blockShape = tuple( blockDims[k] for k in axisOrder )
        self.opLabelArray.blockShape.setValue( blockShape )

        thinCache = 5

        ## Pixel Cache blocks
        blockDimsX = { 't' : (1,1),
                       'z' : (128,256),
                       'y' : (128,256),
                       'x' : (thinCache,thinCache),
                       'c' : (1000,1000) }

        blockDimsY = { 't' : (1,1),
                       'z' : (128,256),
                       'y' : (thinCache,thinCache),
                       'x' : (128,256),
                       'c' : (1000,1000) }

        blockDimsZ = { 't' : (1,1),
                       'z' : (thinCache,thinCache),
                       'y' : (128,256),
                       'x' : (128,256),
                       'c' : (1000,1000) }

        innerBlockShapeX = tuple( blockDimsX[k][0] for k in axisOrder )
        outerBlockShapeX = tuple( blockDimsX[k][1] for k in axisOrder )

        innerBlockShapeY = tuple( blockDimsY[k][0] for k in axisOrder )
        outerBlockShapeY = tuple( blockDimsY[k][1] for k in axisOrder )

        innerBlockShapeZ = tuple( blockDimsZ[k][0] for k in axisOrder )
        outerBlockShapeZ = tuple( blockDimsZ[k][1] for k in axisOrder )

        for cache in self.prediction_caches:
            cache.inputs["innerBlockShape"].setValue( (innerBlockShapeX, innerBlockShapeY, innerBlockShapeZ) )
            cache.inputs["outerBlockShape"].setValue( (outerBlockShapeX, outerBlockShapeY, outerBlockShapeZ) )

        for cache in self.prediction_caches_gui:
            cache.inputs["innerBlockShape"].setValue( (innerBlockShapeX, innerBlockShapeY, innerBlockShapeZ) )
            cache.inputs["outerBlockShape"].setValue( (outerBlockShapeX, outerBlockShapeY, outerBlockShapeZ) )


        #Copy-paste from features
        blockDimsX = { 't' : (1,1),
                        'z' : (128,256),
                        'y' : (128,256),
                        'x' : (thinCache,thinCache),
                        'c' : (1000,1000) } # Overestimate number of feature channels: Cache block dimensions will be clipped to the size of the actual feature image
        
        blockDimsY = { 't' : (1,1),
                        'z' : (128,256),
                        'y' : (thinCache,thinCache),
                        'x' : (128,256),
                        'c' : (1000,1000) }
        
        blockDimsZ = { 't' : (1,1),
                        'z' : (thinCache,thinCache),
                        'y' : (128,256),
                        'x' : (128,256),
                        'c' : (1000,1000) }
        innerBlockShapeX = tuple( blockDimsX[k][0] for k in axisOrder )
        outerBlockShapeX = tuple( blockDimsX[k][1] for k in axisOrder )

        innerBlockShapeY = tuple( blockDimsY[k][0] for k in axisOrder )
        outerBlockShapeY = tuple( blockDimsY[k][1] for k in axisOrder )

        innerBlockShapeZ = tuple( blockDimsZ[k][0] for k in axisOrder )
        outerBlockShapeZ = tuple( blockDimsZ[k][1] for k in axisOrder )

        # Configure the cache   
        for cache in self.autocontext_caches:     
            cache.innerBlockShape.setValue( (innerBlockShapeX, innerBlockShapeY, innerBlockShapeZ) )
            cache.outerBlockShape.setValue( (outerBlockShapeX, outerBlockShapeY, outerBlockShapeZ) )


    
    
    def setInSlot(self, slot, subindex, roi, value):
        # Nothing to do here: All inputs that support __setitem__
        #   are directly connected to internal operators.
        pass

    def propagateDirty(self, inputSlot, subindex, key):
        # Nothing to do here: All outputs are directly connected to 
        #  internal operators that handle their own dirty propagation.
        pass

    def addLane(self, laneIndex):
        numLanes = len(self.InputImages)
        assert numLanes == laneIndex, "Image lanes must be appended."        
        self.InputImages.resize(numLanes+1)
        
    def removeLane(self, laneIndex, finalLength):
        self.InputImages.removeSlot(laneIndex, finalLength)

    def getLane(self, laneIndex):
        return OperatorSubView(self, laneIndex)


def createAutocontextFeatureOperators(oper, wrap):
        #FIXME: just to test, create some array pipers
        ops = []
        if wrap is True:
            ops.append(OperatorWrapper(OpContextVariance, parent=oper))
        else:
            ops.append(OpContextVariance(parent=oper))
        
        #Radii from last year
        ops[0].inputs["Radii"].setValue([[1, 1, 1], [3, 3, 1], [5, 5, 1], [7, 7, 2], [10, 10, 2], \
                  [15, 15, 3], [20, 20, 3], [30, 30, 3], [40, 40, 3]])
        
        #ops[0].inputs["Radii"].setValue([[1, 1, 1], [3, 3, 3], [5, 5, 5], [7, 7, 7], [10, 10, 10], \
        #          [15, 15, 10], [20, 20, 15], [30, 30, 20], [40, 40, 30]])
        
        return ops

class OpShapeReader(Operator):
    """
    This operator outputs the shape of its input image, except the number of channels is set to 1.
    """
    Input = InputSlot()
    OutputShape = OutputSlot(stype='shapetuple')
    
    def __init__(self, *args, **kwargs):
        super(OpShapeReader, self).__init__(*args, **kwargs)
    
    def setupOutputs(self):
        self.OutputShape.meta.shape = (1,)
        self.OutputShape.meta.axistags = 'shapetuple'
        self.OutputShape.meta.dtype = tuple
        
        # Our output is simply the shape of our input, but with only one channel
        channelIndex = self.Input.meta.axistags.index('c')
        shapeList = list(self.Input.meta.shape)
        shapeList[channelIndex] = 1
        self.OutputShape.setValue( tuple(shapeList) )
    
    def execute(self, slot, subindex, roi, result):
        assert False, "Shouldn't get here.  Output is assigned a value in setupOutputs()"

    def propagateDirty(self, inputSlot, subindex, roi):
        # Our output changes when the input changed shape, not when it becomes dirty.
        pass

class OpMaxValue(Operator):
    """
    Accepts a list of non-array values as an input and outputs the max of the list.
    """
    Inputs = InputSlot(level=1) # A list of non-array values
    Output = OutputSlot()
    
    def __init__(self, *args, **kwargs):
        super(OpMaxValue, self).__init__(*args, **kwargs)
        self.Output.meta.shape = (1,)
        self.Output.meta.dtype = object
        self._output = 0
        
    def setupOutputs(self):
        self.updateOutput()
        self.Output.setValue(self._output)

    def execute(self, slot, subindex, roi, result):
        result[0] = self._output
        return result

    def propagateDirty(self, inputSlot, subindex, roi):
        self.updateOutput()
        self.Output.setValue(self._output)

    def updateOutput(self):
        # Return the max value of all our inputs
        maxValue = None
        for i, inputSubSlot in enumerate(self.Inputs):
            # Only use inputs that are actually configured
            if inputSubSlot.ready():
                if maxValue is None:
                    maxValue = inputSubSlot.value
                else:
                    maxValue = max(maxValue, inputSubSlot.value)

        self._output = maxValue


    
