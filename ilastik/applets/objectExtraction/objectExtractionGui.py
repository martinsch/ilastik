from PyQt4.QtGui import QWidget, QColor, QVBoxLayout, QProgressDialog
from PyQt4 import uic
from PyQt4.QtCore import Qt, QString

from lazyflow.rtype import SubRegion

import os

from volumina.api import LazyflowSource, GrayscaleLayer, RGBALayer, ConstantSource, \
                         AlphaModulatedLayer, LayerStackModel, VolumeEditor, VolumeEditorWidget, ColortableLayer
import volumina.colortables as colortables


import logging
logger = logging.getLogger(__name__)
traceLogger = logging.getLogger('TRACE.' + __name__)
from lazyflow.tracer import Tracer


class ObjectExtractionGui( QWidget ):
    """
    """
    
    ###########################################
    ### AppletGuiInterface Concrete Methods ###
    ###########################################
    def centralWidget( self ):
        return self.volumeEditorWidget

    def appletDrawers( self ):
        return [ ("Object Extraction", self._drawer ) ]

    def menus( self ):
        return []

    def viewerControlWidget( self ):
        return self._viewerControlWidget

    def setImageIndex( self, imageIndex ):
        mainOperator = self.mainOperator.innerOperators[imageIndex]
        self.curOp = mainOperator

        ct = colortables.create_default_8bit()
        self.binaryimages = LazyflowSource( mainOperator.BinaryImage )
        layer = GrayscaleLayer( self.binaryimages, range=(0,1), normalize=(0,1) )
        layer.name = "Input Image"
        self.layerstack.append(layer)

        ct = colortables.create_default_16bit()
        self.objectssrc = LazyflowSource( mainOperator.LabelImage )
        ct[0] = QColor(0,0,0,0).rgba() # make 0 transparent
        layer = ColortableLayer( self.objectssrc, ct )
        layer.name = "Label Image"
        layer.opacity = 0.5
        self.layerstack.append(layer)

        self.centerimagesrc = LazyflowSource( mainOperator.ObjectCenterImage )
        layer = RGBALayer( red=ConstantSource(255), alpha=self.centerimagesrc )
        layer.name = "Object Centers"
        self.layerstack.append( layer )

        if mainOperator.Images.meta.shape:
            self.editor.dataShape = mainOperator.LabelImage.meta.shape
        mainOperator.Images.notifyMetaChanged( self._onMetaChanged )            

    def reset( self ):
        print "reset(): not implemented"

    ###########################################
    ###########################################
    
    def __init__(self, mainOperator):
        """
        """
        super(ObjectExtractionGui, self).__init__()
        self.mainOperator = mainOperator
        self.curOp = None
        self.layerstack = LayerStackModel()

        #self.rawsrc = LazyflowSource( self.mainOperator.RawData )
        #layerraw = GrayscaleLayer( self.rawsrc )
        #layerraw.name = "Raw"
        #self.layerstack.append( layerraw )

        self._viewerControlWidget = None
        self._initViewerControlUi()

        self.editor = None
        self._initEditor()

        self._initAppletDrawerUi()

    def _onMetaChanged( self, slot ):
        if slot is self.curOp.Images:
            if slot.meta.shape:
                self.editor.dataShape = slot.meta.shape
 
    def _initEditor(self):
        """
        Initialize the Volume Editor GUI.
        """

        self.editor = VolumeEditor(self.layerstack)

        #self.editor.newImageView2DFocus.connect(self.setIconToViewMenu)
        #self.editor.setInteractionMode( 'navigation' )
        self.volumeEditorWidget = VolumeEditorWidget()
        self.volumeEditorWidget.init(self.editor)

        # The editor's layerstack is in charge of which layer movement buttons are enabled
        model = self.editor.layerStack
        model.canMoveSelectedUp.connect(self._viewerControlWidget.UpButton.setEnabled)
        model.canMoveSelectedDown.connect(self._viewerControlWidget.DownButton.setEnabled)
        model.canDeleteSelected.connect(self._viewerControlWidget.DeleteButton.setEnabled)     

        # Connect our layer movement buttons to the appropriate layerstack actions
        self._viewerControlWidget.layerWidget.init(model)
        self._viewerControlWidget.UpButton.clicked.connect(model.moveSelectedUp)
        self._viewerControlWidget.DownButton.clicked.connect(model.moveSelectedDown)
        self._viewerControlWidget.DeleteButton.clicked.connect(model.deleteSelected)

        self.editor._lastImageViewFocus = 0

            
    def _initAppletDrawerUi(self):
        # Load the ui file (find it in our own directory)
        localDir = os.path.split(__file__)[0]
        self._drawer = uic.loadUi(localDir+"/drawer.ui")

        self._drawer.labelImageButton.pressed.connect(self._onLabelImageButtonPressed)
        self._drawer.extractObjectsButton.pressed.connect(self._onExtractObjectsButtonPressed)
        self._drawer.mergeSegmentationsButton.pressed.connect(self._onMergeSegmentationsButtonPressed)

    def _initViewerControlUi( self ):
        p = os.path.split(__file__)[0]+'/'
        if p == "/": p = "."+p
        self._viewerControlWidget = uic.loadUi(p+"viewerControls.ui")

    def _onLabelImageButtonPressed( self ):
        m = self.curOp.LabelImage.meta
        maxt = m.shape[0]
        progress = QProgressDialog("Labeling Binary Images...", "Stop", 0, maxt * 2)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setCancelButtonText(QString())
        progress.forceShow()

        # LabelImage for background/non-background (channel 0) and division/non-division (channel 2)
        # TODO: run multi-threaded??? see _onExtractObjectsButtonPressed!
        for idx,c in enumerate([0,2]):
            for t in range(maxt):
                progress.setValue(idx * maxt + t * (idx+1))
                if progress.wasCanceled():
                    break
                else:
                    self.curOp.updateLabelImageAt( t, c )
        progress.setValue(maxt * 2)
        roi = SubRegion(self.curOp.LabelImage, start=5*(0,), stop=m.shape)
        # TODO: set LabelImage dirty to update the result for the current view!
        try:         
            self.curOp.LabelImage.setDirty(roi)
        except:
            print "TODO: set LabelImage dirty to update the result for the current view"
        
        print 'Label Segmentation: done.'


    def _onExtractObjectsButtonPressed( self ):
        maxt = self.curOp.LabelImage.meta.shape[0]
        progress = QProgressDialog("Extracting objects...", "Stop", 0, maxt)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setCancelButtonText(QString())

        reqs = []
        self.curOp._opObjectExtractionBg._opRegFeats.fixed = False
        self.curOp._opObjectExtractionDiv._opRegFeats.fixed = False
        for t in range(maxt):
            reqs.append(self.curOp._opObjectExtractionBg.RegionFeatures([t]))
            reqs[-1].submit()
            reqs.append(self.curOp._opObjectExtractionDiv.RegionFeatures([t]))
            reqs[-1].submit()
        for i, req in enumerate(reqs):
            progress.setValue(i)
            if progress.wasCanceled():
                req.cancel()
            else:
                req.wait()
                
        self.curOp._opObjectExtractionBg._opRegFeats.fixed = True 
        self.curOp._opObjectExtractionDiv._opRegFeats.fixed = True
        progress.setValue(maxt)
        
        self.curOp._opObjectExtractionBg.ObjectCenterImage.setDirty( SubRegion(self.curOp._opObjectExtractionBg.ObjectCenterImage))
        self.curOp._opObjectExtractionDiv.ObjectCenterImage.setDirty( SubRegion(self.curOp._opObjectExtractionDiv.ObjectCenterImage))
        
        print 'Object Extraction: done.'


    def _onMergeSegmentationsButtonPressed(self):
        m = self.curOp.LabelImage.meta
        maxt = m.shape[0]
        progress = QProgressDialog("Merging Background and Division Segmentations...", "Stop", 0, maxt)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setCancelButtonText(QString())
        progress.forceShow()

        reqs = []
        self.curOp._opClassExtraction.fixed = False
        for t in range(maxt):
            reqs.append(self.curOp._opClassExtraction.ClassMapping([t]))
            reqs[-1].submit()
        for i, req in enumerate(reqs):
            progress.setValue(i)
            if progress.wasCanceled():
                req.cancel()
            else:
                req.wait()
        
        self.curOp._opClassExtraction.fixed = True
        progress.setValue(maxt)
        
        print 'Merge Segmentation: done.'        