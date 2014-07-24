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
from ilastik.applets.base.appletSerializer import \
    AppletSerializer

class ProjectMetadataSerializer(AppletSerializer):

    def __init__(self, projectMetadata, projectFileGroupName):
        super( ProjectMetadataSerializer, self ).__init__(projectFileGroupName)
        self.projectMetadata = projectMetadata
        self._dirty = False

        def handleChange():
            self._dirty = True
        projectMetadata.changedSignal.connect( handleChange )

    def _serializeToHdf5(self, topGroup, hdf5File, projectFilePath):
        metadataGroup = topGroup

        # Write each of our values to the group
        self.setDataset(metadataGroup, 'ProjectName', self.projectMetadata.projectName)
        self.setDataset(metadataGroup, 'Labeler', self.projectMetadata.labeler)
        self.setDataset(metadataGroup, 'Description', self.projectMetadata.description)
        self._dirty = False

    def _deserializeFromHdf5(self, topGroup, groupVersion, hdf5File, projectFilePath):
        self.projectMetadata.projectName = self.getDataset(topGroup, 'ProjectName')
        self.projectMetadata.labeler = self.getDataset(topGroup, 'Labeler')
        self.projectMetadata.description = self.getDataset(topGroup, 'Description')
        self._dirty = False

    def isDirty(self):
        """ Return true if the current state of this item
            (in memory) does not match the state of the HDF5 group on disk.
            SerializableItems are responsible for tracking their own dirty/notdirty state."""
        return self._dirty

    def unload(self):
        """ Called if either
            (1) the user closed the project or
            (2) the project opening process needs to be aborted for some reason
                (e.g. not all items could be deserialized properly due to a corrupted ilp)
            This way we can avoid invalid state due to a partially loaded project. """
        self.projectMetadata.projectName = ''
        self.projectMetadata.labeler = ''
        self.projectMetadata.description = ''

    def setDataset(self, group, dataName, dataValue):
        if dataName not in group.keys():
            # Create and assign
            group.create_dataset(dataName, data=dataValue)
        else:
            # Assign (this will fail if the dtype doesn't match)
            group[dataName][()] = dataValue

    def getDataset(self, group, dataName):
        try:
            result = group[dataName].value
        except KeyError:
            result = ''
        return result

class Ilastik05ProjectMetadataDeserializer(AppletSerializer):
    def __init__(self, projectMetadata):
        super( Ilastik05ProjectMetadataDeserializer, self ).__init__('')
        self.projectMetadata = projectMetadata

    def serializeToHdf5(self, hdf5File, filePath):
        # This is for deserialization only.
        pass

    def deserializeFromHdf5(self, hdf5File, filePath):
        # Check the overall file version
        ilastikVersion = hdf5File["ilastikVersion"].value

        # This is the v0.5 import deserializer.  Don't work with 0.6 projects (or anything else).
        if ilastikVersion != 0.5:
            return

        try:
            metadataGroup = hdf5File['Project']
        except KeyError:
            self.projectMetadata.projectName = ''
            self.projectMetadata.labeler = ''
            self.projectMetadata.description = ''
            return

        self.projectMetadata.projectName = self.getDataset(metadataGroup, 'Name')
        self.projectMetadata.labeler = self.getDataset(metadataGroup, 'Labeler')
        self.projectMetadata.description = self.getDataset(metadataGroup, 'Description')

    def isDirty(self):
        """ Return true if the current state of this item
            (in memory) does not match the state of the HDF5 group on disk.
            SerializableItems are responsible for tracking their own dirty/notdirty state."""
        return False

    def unload(self):
        """ Called if either
            (1) the user closed the project or
            (2) the project opening process needs to be aborted for some reason
                (e.g. not all items could be deserialized properly due to a corrupted ilp)
            This way we can avoid invalid state due to a partially loaded project. """
        self.projectMetadata.projectName = ''
        self.projectMetadata.labeler = ''
        self.projectMetadata.description = ''

    def getDataset(self, group, dataName):
        try:
            result = group[dataName].value
        except KeyError:
            result = ''
        return result

    def _serializeToHdf5(self, topGroup, hdf5File, projectFilePath):
        assert False

    def _deserializeFromHdf5(self, topGroup, groupVersion, hdf5File, projectFilePath):
        # This deserializer is a special-case.
        # It doesn't make use of the serializer base class, which makes assumptions about the file structure.
        # Instead, if overrides the public serialize/deserialize functions directly
        assert False
