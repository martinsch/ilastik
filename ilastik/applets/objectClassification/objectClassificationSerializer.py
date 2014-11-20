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
import warnings

from ilastik.applets.base.appletSerializer import \
  AppletSerializer, SerialSlot, SerialDictSlot, \
  SerialClassifierSlot, SerialListSlot

class SerialDictSlotWithoutDeserialization(SerialDictSlot):
    
    def __init__(self, *args, **kwargs):
        super(SerialDictSlotWithoutDeserialization, self).__init__(*args, **kwargs)
        #self.mainOperator = mainOperator
    
    def serialize(self, *args):
        #if self.slot.ready() and self.mainOperator._predict_enabled:
        return SerialDictSlot.serialize(self, *args)
    
    def deserialize(self, *args, **kwargs):
        # Do not deserialize this slot
        pass


class SerialClassifierSlotWithoutDeserialization(SerialClassifierSlot):
    
    def __init__(self, *args, **kwargs):
        super(SerialClassifierSlotWithoutDeserialization, self).__init__(*args, **kwargs)
        #self.mainOperator = mainOperator
    
    def serialize(self, *args):
        #if self.slot.ready() and self.mainOperator._predict_enabled:
        return SerialClassifierSlot.serialize(self, *args)
 
    def deserialize(self, *args, **kwargs):
        pass

class ObjectClassificationSerializer(AppletSerializer):
    # FIXME: predictions can only be saved, not loaded, because it
    # would call setValue() on a connected slot

    def __init__(self, topGroupName, operator):
        serialSlots = [
            SerialDictSlot(operator.SelectedFeatures, transform=str),
            SerialListSlot(operator.LabelNames,
                           transform=str),
            SerialListSlot(operator.LabelColors, transform=lambda x: tuple(x.flat)),
            SerialListSlot(operator.PmapColors, transform=lambda x: tuple(x.flat)),
            SerialDictSlot(operator.LabelInputs, transform=int),
            SerialClassifierSlot(operator.Classifier,
                                operator.classifier_cache,
                                name="ClassifierForests"),
            # SerialClassifierSlotWithoutDeserialization(operator.Classifier,
            #                       operator.classifier_cache,
            #                       name="ClassifierForests"),
            SerialDictSlot(operator.CachedProbabilities,
                           operator.InputProbabilities,
                           transform=int),
            SerialDictSlot(operator.CachedUncertainty,
                           operator.InputUncertainty,
                           transform=int),
            SerialDictSlotWithoutDeserialization(operator.Probabilities, transform=str),
            SerialDictSlotWithoutDeserialization(operator.Uncertainty, transform=str)
        ]

        super(ObjectClassificationSerializer, self ).__init__(topGroupName,
                                                              slots=serialSlots,
                                                              operator=operator)
        
    def isDirty(self):
        return True
