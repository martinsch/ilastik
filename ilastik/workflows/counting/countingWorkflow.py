# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# Copyright 2011-2014, the ilastik developers

from lazyflow.graph import Graph, Operator, OperatorWrapper

from ilastik.workflow import Workflow

from ilastik.applets.projectMetadata import ProjectMetadataApplet
from ilastik.applets.dataSelection import DataSelectionApplet
from ilastik.applets.featureSelection import FeatureSelectionApplet

from ilastik.applets.counting import CountingApplet, CountingDataExportApplet
from ilastik.applets.featureSelection.opFeatureSelection import OpFeatureSelection
from ilastik.applets.counting.opCounting import OpPredictionPipeline

from lazyflow.roi import TinyVector
from lazyflow.graph import Graph, OperatorWrapper
from lazyflow.operators.generic import OpTransposeSlots, OpSelectSubslot

class CountingWorkflow(Workflow):
    workflowName = "Cell Density Counting"
    workflowDescription = "This is obviously self-explanatory."
    defaultAppletIndex = 1 # show DataSelection by default

    def __init__(self, shell, headless, workflow_cmdline_args, project_creation_args, appendBatchOperators=True, *args, **kwargs):
        graph = kwargs['graph'] if 'graph' in kwargs else Graph()
        if 'graph' in kwargs: del kwargs['graph']
        super( CountingWorkflow, self ).__init__( shell, headless, workflow_cmdline_args, project_creation_args, graph=graph, *args, **kwargs )

        ######################
        # Interactive workflow
        ######################

        self.projectMetadataApplet = ProjectMetadataApplet()

        self.dataSelectionApplet = DataSelectionApplet(self,
                                                       "Input Data",
                                                       "Input Data",
                                                       batchDataGui=False,
                                                       force5d=False
                                                      )
        opDataSelection = self.dataSelectionApplet.topLevelOperator
        opDataSelection.DatasetRoles.setValue( ['Raw Data'] )

        self.featureSelectionApplet = FeatureSelectionApplet(self,
                                                             "Feature Selection",
                                                             "FeatureSelections")

        #self.pcApplet = PixelClassificationApplet(self, "PixelClassification")
        self.countingApplet = CountingApplet(workflow=self)
        opCounting = self.countingApplet.topLevelOperator

        self.dataExportApplet = CountingDataExportApplet(self, "Density Export")
        
        opDataExport = self.dataExportApplet.topLevelOperator
        opDataExport.PmapColors.connect(opCounting.PmapColors)
        opDataExport.LabelNames.connect(opCounting.LabelNames)
        opDataExport.UpperBound.connect(opCounting.UpperBound)
        opDataExport.WorkingDirectory.connect(opDataSelection.WorkingDirectory)


        self._applets = []
        self._applets.append(self.projectMetadataApplet)
        self._applets.append(self.dataSelectionApplet)
        self._applets.append(self.featureSelectionApplet)
        self._applets.append(self.countingApplet)
        self._applets.append(self.dataExportApplet)


        if appendBatchOperators:
            # Create applets for batch workflow
            self.batchInputApplet = DataSelectionApplet(self, "Batch Prediction Input Selections", "BatchDataSelection", supportIlastik05Import=False, batchDataGui=True)
            self.batchResultsApplet = CountingDataExportApplet(self, "Batch Prediction Output Locations", isBatch=True)
    
            # Expose in shell        
            self._applets.append(self.batchInputApplet)
            self._applets.append(self.batchResultsApplet)
    
            # Connect batch workflow (NOT lane-based)
            self._initBatchWorkflow()

    @property
    def applets(self):
        return self._applets

    @property
    def imageNameListSlot(self):
        return self.dataSelectionApplet.topLevelOperator.ImageName

    def connectLane(self, laneIndex):
        ## Access applet operators
        opData = self.dataSelectionApplet.topLevelOperator.getLane(laneIndex)
        opTrainingFeatures = self.featureSelectionApplet.topLevelOperator.getLane(laneIndex)
        opCounting = self.countingApplet.topLevelOperator.getLane(laneIndex)
        opDataExport = self.dataExportApplet.topLevelOperator.getLane(laneIndex)


        #### connect input image
        opTrainingFeatures.InputImage.connect(opData.Image)

        opCounting.InputImages.connect(opData.Image)
        opCounting.FeatureImages.connect(opTrainingFeatures.OutputImage)
        opCounting.LabelsAllowedFlags.connect(opData.AllowLabels)
        opCounting.CachedFeatureImages.connect( opTrainingFeatures.CachedOutputImage )
        #opCounting.UserLabels.connect(opClassify.LabelImages)
        #opCounting.ForegroundLabels.connect(opObjExtraction.LabelImage)
        opDataExport.RawData.connect( opData.ImageGroup[0] )
        opDataExport.Input.connect( opCounting.HeadlessPredictionProbabilities )
        opDataExport.RawDatasetInfo.connect( opData.DatasetGroup[0] )
        opDataExport.ConstraintDataset.connect( opData.ImageGroup[0] )

    def _initBatchWorkflow(self):
        """
        Connect the batch-mode top-level operators to the training workflow and to eachother.
        """
        # Access applet operators from the training workflow
        opTrainingDataSelection = self.dataSelectionApplet.topLevelOperator
        opTrainingFeatures = self.featureSelectionApplet.topLevelOperator
        opClassify = self.countingApplet.topLevelOperator
        
        # Access the batch operators
        opBatchInputs = self.batchInputApplet.topLevelOperator
        opBatchResults = self.batchResultsApplet.topLevelOperator
        
        opBatchInputs.DatasetRoles.connect( opTrainingDataSelection.DatasetRoles )
        
        opSelectFirstLane = OperatorWrapper( OpSelectSubslot, parent=self )
        opSelectFirstLane.Inputs.connect( opTrainingDataSelection.ImageGroup )
        opSelectFirstLane.SubslotIndex.setValue(0)
        
        opSelectFirstRole = OpSelectSubslot( parent=self )
        opSelectFirstRole.Inputs.connect( opSelectFirstLane.Output )
        opSelectFirstRole.SubslotIndex.setValue(0)
        
        opBatchResults.ConstraintDataset.connect( opSelectFirstRole.Output )
        
        ## Create additional batch workflow operators
        opBatchFeatures = OperatorWrapper( OpFeatureSelection, operator_kwargs={'filter_implementation':'Original'}, parent=self, promotedSlotNames=['InputImage'] )
        opBatchPredictionPipeline = OperatorWrapper( OpPredictionPipeline, parent=self )
        
        ## Connect Operators ##
        opTranspose = OpTransposeSlots( parent=self )
        opTranspose.OutputLength.setValue(1)
        opTranspose.Inputs.connect( opBatchInputs.DatasetGroup )
        
        # Provide dataset paths from data selection applet to the batch export applet
        opBatchResults.RawDatasetInfo.connect( opTranspose.Outputs[0] )
        opBatchResults.WorkingDirectory.connect( opBatchInputs.WorkingDirectory )
        
        # Connect (clone) the feature operator inputs from 
        #  the interactive workflow's features operator (which gets them from the GUI)
        opBatchFeatures.Scales.connect( opTrainingFeatures.Scales )
        opBatchFeatures.FeatureIds.connect( opTrainingFeatures.FeatureIds )
        opBatchFeatures.SelectionMatrix.connect( opTrainingFeatures.SelectionMatrix )
        
        # Classifier and LabelsCount are provided by the interactive workflow
        opBatchPredictionPipeline.Classifier.connect( opClassify.Classifier )
        opBatchPredictionPipeline.MaxLabel.connect( opClassify.MaxLabelValue )
        opBatchPredictionPipeline.FreezePredictions.setValue( False )
        
        # Provide these for the gui
        opBatchResults.RawData.connect( opBatchInputs.Image )
        opBatchResults.PmapColors.connect( opClassify.PmapColors )
        opBatchResults.LabelNames.connect( opClassify.LabelNames )
        opBatchResults.UpperBound.connect( opClassify.UpperBound )
        
        # Connect Image pathway:
        # Input Image -> Features Op -> Prediction Op -> Export
        opBatchFeatures.InputImage.connect( opBatchInputs.Image )
        opBatchPredictionPipeline.FeatureImages.connect( opBatchFeatures.OutputImage )
        opBatchResults.Input.connect( opBatchPredictionPipeline.HeadlessPredictionProbabilities )

        # We don't actually need the cached path in the batch pipeline.
        # Just connect the uncached features here to satisfy the operator.
        opBatchPredictionPipeline.CachedFeatureImages.connect( opBatchFeatures.OutputImage )

        self.opBatchPredictionPipeline = opBatchPredictionPipeline


    def handleAppletStateUpdateRequested(self):
        """
        Overridden from Workflow base class
        Called when an applet has fired the :py:attr:`Applet.statusUpdateSignal`
        """
        # If no data, nothing else is ready.
        opDataSelection = self.dataSelectionApplet.topLevelOperator
        input_ready = len(opDataSelection.ImageGroup) > 0 and not self.dataSelectionApplet.busy

        opFeatureSelection = self.featureSelectionApplet.topLevelOperator
        featureOutput = opFeatureSelection.OutputImage
        features_ready = input_ready and \
                         len(featureOutput) > 0 and  \
                         featureOutput[0].ready() and \
                         (TinyVector(featureOutput[0].meta.shape) > 0).all()

        opDataExport = self.dataExportApplet.topLevelOperator
        predictions_ready = features_ready and \
                            len(opDataExport.Input) > 0 and \
                            opDataExport.Input[0].ready() and \
                            (TinyVector(opDataExport.Input[0].meta.shape) > 0).all()

        self._shell.setAppletEnabled(self.featureSelectionApplet, input_ready)
        self._shell.setAppletEnabled(self.countingApplet, features_ready)
        self._shell.setAppletEnabled(self.dataExportApplet, predictions_ready)
        
        # Training workflow must be fully configured before batch can be used
        self._shell.setAppletEnabled(self.batchInputApplet, predictions_ready)

        opBatchDataSelection = self.batchInputApplet.topLevelOperator
        batch_input_ready = predictions_ready and \
                            len(opBatchDataSelection.ImageGroup) > 0
        self._shell.setAppletEnabled(self.batchResultsApplet, batch_input_ready)
        
        # Lastly, check for certain "busy" conditions, during which we 
        #  should prevent the shell from closing the project.
        busy = False
        busy |= self.dataSelectionApplet.busy
        busy |= self.featureSelectionApplet.busy
        busy |= self.dataExportApplet.busy
        self._shell.enableProjectChanges( not busy )
