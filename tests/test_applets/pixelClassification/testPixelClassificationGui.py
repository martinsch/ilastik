from tests.helpers import ShellGuiTestCaseBase
from workflows.pixelClassification import PixelClassificationWorkflow

from PyQt4.QtGui import QApplication

class TestPixelClassificationGui(ShellGuiTestCaseBase):
    """
    Run a set of GUI-based tests on the pixel classification workflow.
    
    Note: These tests are named in order so that simple cases are tried before complex ones.
          Additionally, later tests may depend on earlier ones to run properly.
    """
    
    @classmethod
    def workflowClass(cls):
        return PixelClassificationWorkflow

    SAMPLE_DATA = '/magnetic/gigacube.h5'
    PROJECT_FILE = '/magnetic/test_project.ilp'

    @classmethod
    def teardownClass(cls):
        """
        Call our base class to quit the app during teardown.
        (Comment this out if you want the app to stay open for further debugging.)
        """
        super(TestPixelClassificationGui, cls).teardownClass()

    def test_1_NewProject(self):
        """
        Create a blank project and manipulate a couple settings.
        Then save it.
        """
        def impl():
            projFilePath = self.PROJECT_FILE
        
            shell = self.shell
            workflow = self.workflow
            
            # New project
            shell.createAndLoadNewProject(projFilePath)
        
            # Add a file
            from ilastik.applets.dataSelection.opDataSelection import DatasetInfo
            info = DatasetInfo()
            info.filePath = self.SAMPLE_DATA
            opDataSelection = workflow.dataSelectionApplet.topLevelOperator
            opDataSelection.Dataset.resize(1)
            opDataSelection.Dataset[0].setValue(info)
            
            # Set some features
            import numpy
            featureGui = workflow.featureSelectionApplet.gui
            opFeatures = workflow.featureSelectionApplet.topLevelOperator
            opFeatures.Scales.setValue( featureGui.ScalesList )
            opFeatures.FeatureIds.setValue( featureGui.FeatureIds )
            #                    sigma:   0.3    0.7    1.0    1.6    3.5    5.0   10.0
            selections = numpy.array( [[True, False, False, False, False, False, False],
                                       [True, False, False, False, False, False, False],
                                       [True, False, False, False, False, False, False],
                                       [False, False, False, False, False, False, False],
                                       [False, False, False, False, False, False, False],
                                       [False, False, False, False, False, False, False]] )
            opFeatures.SelectionMatrix.setValue(selections)
        
            # Save the project
            shell.onSaveProjectActionTriggered()

        # Run this test from within the shell event loop
        self.exec_in_shell(impl)

    def test_2_AddLabels(self):
        def impl():
            pixClassApplet = self.workflow.pcApplet
            gui = pixClassApplet.gui
            opPix = pixClassApplet.topLevelOperator

            # Select the labeling drawer
            self.shell.setSelectedAppletDrawer(3)

            assert not gui._labelControlUi.checkInteractive.isChecked()
            assert gui._labelControlUi.labelListModel.rowCount() == 0
            
            # Add label classes
            for i in range(3):
                gui._labelControlUi.AddLabelButton.click()
                assert gui._labelControlUi.labelListModel.rowCount() == i+1

            # Select the brush
            gui._labelControlUi.paintToolButton.click()

            # Set the brush size
            gui._labelControlUi.brushSizeComboBox.setCurrentIndex(1)

            # Let the GUI catch up: Process all events
            QApplication.processEvents()

            # Draw some arbitrary labels in each view using mouse events.
            for i in range(3):
                # Post this as an event to ensure sequential execution.
                gui._labelControlUi.labelListModel.select(i)
                
                imgView = gui.editor.imageViews[i]
                self.strokeMouse( imgView, (0,0), (100,100) )

                # Make sure the labels were added to the label array operator
                assert opPix.MaxLabelValue.value == i+1

            self.waitForViews(gui.editor.imageViews)

            # Verify the actual rendering of each view
            for i in range(3):
                imgView = gui.editor.imageViews[i]
                observedColor = self.getPixelColor(imgView, (50,50))
                expectedColor = gui._colorTable16[i+1]
                assert observedColor == expectedColor, "Label was not drawn correctly.  Expected {}, got {}".format( hex(expectedColor), hex(observedColor) )                

            # Save the project
            self.shell.onSaveProjectActionTriggered()

        # Run this test from within the shell event loop
        self.exec_in_shell(impl)

    def test_3_DeleteLabel(self):
        """
        Relies on test 2.
        """
        def impl():
            pixClassApplet = self.workflow.pcApplet
            gui = pixClassApplet.gui
            opPix = pixClassApplet.topLevelOperator

            originalLabelColors = gui._colorTable16[1:4]

            # We assume that there are three labels to start with (see previous test)
            assert opPix.MaxLabelValue.value == 3

            # Make sure that it's okay to delete a row even if the deleted label is selected.
            gui._labelControlUi.labelListModel.select(1)
            gui._labelControlUi.labelListModel.removeRow(1)

            # Let the GUI catch up: Process all events
            QApplication.processEvents()
            
            # Selection should auto-reset back to the first row.
            assert gui._labelControlUi.labelListModel.selectedRow() == 0
            
            # Did the label get removed from the label array?
            assert opPix.MaxLabelValue.value == 2

            self.waitForViews(gui.editor.imageViews)

            # Check the actual rendering of the two views with remaining labels
            for i in [0,2]:
                imgView = gui.editor.imageViews[i]
                observedColor = self.getPixelColor(imgView, (50,50))
                expectedColor = originalLabelColors[i]
                assert observedColor == expectedColor, "Label was not drawn correctly.  Expected {}, got {}".format( hex(expectedColor), hex(observedColor) )                

            # Make sure we actually deleted the middle label (it should no longer be visible)
            for i in [1]:
                imgView = gui.editor.imageViews[i]
                observedColor = self.getPixelColor(imgView, (50,50))
                oldColor = originalLabelColors[i]
                assert observedColor != oldColor, "Label was not deleted."

        # Run this test from within the shell event loop
        self.exec_in_shell(impl)

    def test_4_EraseSome(self):
        def impl():
            pixClassApplet = self.workflow.pcApplet
            gui = pixClassApplet.gui

            # Select the labeling drawer
            self.shell.setSelectedAppletDrawer(3)

            assert not gui._labelControlUi.checkInteractive.isChecked()
            assert gui._labelControlUi.labelListModel.rowCount() == 2
            
            # Use the first view for this test
            imgView = gui.editor.imageViews[0]

            # Sanity check: There should be labels in the view that we can erase
            self.waitForViews([imgView])
            observedColor = self.getPixelColor(imgView, (50,50))
            labelColor = gui._colorTable16[1]
            assert observedColor == labelColor, "Can't run erase test.  Missing the expected label.  Expected {}, got {}".format( hex(labelColor), hex(observedColor) )

            # Hide labels and sample raw data
            labelLayer = gui.layerstack[0]
            assert labelLayer.name == "Labels"
            labelLayer.visible = False            
            self.waitForViews([imgView])
            rawDataColor = self.getPixelColor(imgView, (50,50))
            assert rawDataColor != labelColor
            
            # Show labels
            labelLayer.visible = True
            # Select the eraser and brush size
            gui._labelControlUi.eraserToolButton.click()
            gui._labelControlUi.brushSizeComboBox.setCurrentIndex(3)
            self.waitForViews([imgView])
            
            # Erase and verify
            self.strokeMouse( imgView, (45,45), (55,55) )
            self.waitForViews([imgView])
            erasedColor = self.getPixelColor(imgView, (50,50))
            assert erasedColor == rawDataColor
        
        # Run this test from within the shell event loop
        self.exec_in_shell(impl)

    def test_5_EraseCompleteLabel(self):
        """
        Erase all of the labels of a particular color.
        Verify that nothing screwy happens.
        """
        def impl():
            pixClassApplet = self.workflow.pcApplet
            gui = pixClassApplet.gui
            opPix = pixClassApplet.topLevelOperator

            # Select the labeling drawer
            self.shell.setSelectedAppletDrawer(3)

            assert not gui._labelControlUi.checkInteractive.isChecked()
            assert gui._labelControlUi.labelListModel.rowCount() == 2

            assert opPix.MaxLabelValue.value == 2
            
            # Use the third view for this test (which has the max label value)
            imgView = gui.editor.imageViews[2]

            # Sanity check: There should be labels in the view that we can erase
            self.waitForViews([imgView])
            observedColor = self.getPixelColor(imgView, (50,50))
            labelColor = gui._colorTable16[2]
            assert observedColor == labelColor, "Can't run erase test.  Missing the expected label.  Expected {}, got {}".format( hex(labelColor), hex(observedColor) )

            # Hide labels and sample raw data
            labelLayer = gui.layerstack[0]
            assert labelLayer.name == "Labels"
            labelLayer.visible = False            
            self.waitForViews([imgView])
            rawDataColor = self.getPixelColor(imgView, (50,50))
            assert rawDataColor != labelColor
            
            # Show labels
            labelLayer.visible = True
            # Select the eraser and brush size
            gui._labelControlUi.eraserToolButton.click()
            gui._labelControlUi.brushSizeComboBox.setCurrentIndex(3)
            self.waitForViews([imgView])
            
            # Erase and verify
            self.strokeMouse( imgView, (0,0), (100,100) )
            self.waitForViews([imgView])
            erasedColor = self.getPixelColor(imgView, (50,50))
            assert erasedColor == rawDataColor, "Eraser did not remove labels!"

            # We just erased all the labels of value 2, so the max label value should be reduced.
            assert opPix.MaxLabelValue.value == 1

            # Now stroke the eraser once more.
            # The new stroke should make NO DIFFERENCE to the image.
            rawDataColor = self.getPixelColor(imgView, (25,75))
            self.strokeMouse( imgView, (100,0), (0,100) )
            self.waitForViews([imgView])
            erasedColor = self.getPixelColor(imgView, (25,75))
            assert erasedColor == rawDataColor, "Erasing blank pixels generated non-zero labels."

        # Run this test from within the shell event loop
        self.exec_in_shell(impl)

    def test_6_InteractiveMode(self):
        """
        Click the "interactive mode" checkbox.
        Prerequisites: Relies on test 2.
        """
        def impl():
            pixClassApplet = self.workflow.pcApplet
            gui = pixClassApplet.gui

            # Enable interactive mode            
            assert not gui._labelControlUi.checkInteractive.isChecked()
            gui._labelControlUi.checkInteractive.click()

            self.waitForViews(gui.editor.imageViews)

            # Disable iteractive mode.            
            gui._labelControlUi.checkInteractive.click()

            self.waitForViews(gui.editor.imageViews)
            
        # Run this test from within the shell event loop
        self.exec_in_shell(impl)

if __name__ == "__main__":
    from tests.helpers.shellGuiTestCaseBase import run_shell_nosetest
    run_shell_nosetest(__file__)








