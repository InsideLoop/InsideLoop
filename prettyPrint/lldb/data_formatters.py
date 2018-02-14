#===============================================================================
#
# Copyright 2017 The InsideLoop Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#===============================================================================

from __future__ import print_function

import lldb
import lldb.formatters.Logger

def __lldb_init_module (debugger, dict):
    # debugger.HandleCommand("type summary add -F data_formatters.display_string il::String")
    debugger.HandleCommand("type synthetic add -x \"il::String\" --python-class data_formatters.StringProvider")
    debugger.HandleCommand("type synthetic add -x \"il::Array<\" --python-class data_formatters.ArrayProvider")
    # debugger.HandleCommand("type summary add -F data_formatters.display_array -x \"il::Array<\"")
    debugger.HandleCommand("type synthetic add -x \"il::StaticArray<\" --python-class data_formatters.StaticArrayProvider")
    debugger.HandleCommand("type synthetic add -x \"il::Array2D<\" --python-class data_formatters.Array2DProvider")
    debugger.HandleCommand("type synthetic add -x \"il::StaticArray2D<\" --python-class data_formatters.StaticArray2DProvider")
    debugger.HandleCommand("type synthetic add -x \"il::Array2C<\" --python-class data_formatters.Array2CProvider")
    debugger.HandleCommand("type synthetic add -x \"il::Array3D<\" --python-class data_formatters.Array3DProvider")
    debugger.HandleCommand("type synthetic add -x \"il::Array3C<\" --python-class data_formatters.Array3CProvider")



# A SBValue has a name (string), a type and a value
# lldb.SBValue:
#   lldb.SBType GetType()
#   lldb.SBData GetData()
#   ??????????? GetSummary()
#   ??????????? GetNumChildren()    (For a C array such as char data[10])
#   ??????????? GetChildAtIndex(k) is it a SBValue?
#
# lldb.SBType
#   lldb.SBType GetPointeetype()
#
# lldb.SBData
#   'instance' uint8

# type(valobj): lldb.SBValue
# type(valobj.GetChildAtIndex(0)): lldb.SBValue
# type(valobj.GetChildAtIndex(0).GetData()): lldb.SBData

# A read only property that returns an array-like object out of which you can
# read uint8 values
# type(valobj.GetChildAtIndex(0).GetData().uint8): 'instance'

# def display_string(valobj, internal_dict):
#     ans = ''
#     s = valobj.GetChildAtIndex(0).GetData().uint8
#     info = s[23]
#     is_small = info < 128
#     info_type = (info % 128) / 32
#     if info_type == 0:
#         string_type = 'ASCII'
#     elif info_type == 1:
#         string_type = 'UTF8'
#     elif info_type == 2:
#         string_type = 'WTF8'
#     else:
#         string_type = 'Byte'
#     if is_small:
#         k = 0
#         while s[k] != 0:
#             ans = ans + chr(s[k])
#             k = k + 1
#         return '"' + ans + '" [size: ' + str(k) + '] [capacity: 22] [small: ' + str(is_small) + '] [type: ' + string_type + ']'
#     else:
#         the_ans = valobj.GetChildMemberWithName('large_').GetChildMemberWithName('data').GetSummary()
#         the_size = valobj.GetChildMemberWithName('large_').GetChildMemberWithName('size').GetValueAsUnsigned()
#         the_capacity = valobj.GetChildMemberWithName('large_').GetChildMemberWithName('capacity').GetValueAsUnsigned()
#         return the_ans + ' [size: ' + str(the_size) + '] [capacity: ' + str(8 * (the_capacity % (2**61))) + '] [small: ' + str(is_small) + '] [type: ' + string_type + ']'

class StringProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.data = self.valobj.GetChildMemberWithName('large_')
        self.small = valobj.GetChildAtIndex(0).GetData().uint8
        self.string_type_id = (self.small[23] % 128) / 32
        if self.string_type_id == 0:
            self.string_type = 'Ascii'
        elif self.string_type_id == 1:
            self.string_type = 'UTF8'
        elif self.string_type_id == 2:
            self.string_type = 'WTF8'
        elif self.string_type_id == 3:
            self.string_type = 'Byte'
        self.is_small = self.small[23] < 128
        if self.is_small:
            self.string = ''
            k = 0
            while self.small[k] != 0:
                self.string = self.string + chr(self.small[k])
                k = k + 1
            self.size = k
            self.capacity = 22
        else:
            self.string = (valobj.GetChildMemberWithName('large_').GetChildMemberWithName('data').GetSummary())[1:-1]
            self.size = valobj.GetChildMemberWithName('large_').GetChildMemberWithName('size').GetValueAsUnsigned()
            self.capacity = 8 * (valobj.GetChildMemberWithName('large_').GetChildMemberWithName('capacity').GetValueAsUnsigned() % (2**61))

    def num_children(self):
        return 4

    def get_child_at_index(self, index):
        if index < 0:
            return None
        if index >= self.num_children():
            return None
        try:
            my_frame = self.data.frame
            if index == 0:
                x = my_frame.EvaluateExpression('"' + self.string + '"')
                return x.CreateValueFromData('value', x.GetData(), x.GetType())
            if index == 1:
                x = my_frame.EvaluateExpression('static_cast<il::int_t>(' + str(self.size) + ')')
                return x.CreateValueFromData('size', x.GetData(), x.GetType())
            if index == 2:
                x = my_frame.EvaluateExpression('static_cast<il::int_t>(' + str(self.capacity) + ')')
                return x.CreateValueFromData('capacity', x.GetData(), x.GetType())
            if index == 3:
                x = my_frame.EvaluateExpression('"' + self.string_type + '"')
                return x.CreateValueFromData('type', x.GetData(), x.GetType())
        except:
            return None

def display_string(valobj, internal_dict):
    prov = StringProvider(valobj, internal_dict)
    return prov.string

class ArrayProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.data = self.valobj.GetChildMemberWithName('data_')
        self.data_type = self.data.GetType().GetPointeeType()
        self.type_size = self.data_type.GetByteSize()
        self.size = (self.valobj.GetChildMemberWithName('size_').GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.capacity = (self.valobj.GetChildMemberWithName('capacity_').GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size

    def num_children(self):
        return 2 + self.size

    def get_child_at_index(self, index):
        if index < 0:
            return None
        if index >= self.num_children():
            return None
        try:
            my_frame = self.data.frame
            if index == 0:
                x = my_frame.EvaluateExpression('static_cast<il::int_t>(' + str(self.size) + ')')
                return x.CreateValueFromData('size', x.GetData(), x.GetType())
            elif index == 1:
                x = my_frame.EvaluateExpression('static_cast<il::int_t>(' + str(self.capacity) + ')')
                return x.CreateValueFromData('capacity', x.GetData(), x.GetType())
            else:
                offset = (index - 2) * self.type_size
                return self.data.CreateChildAtOffset(
                    '[' + str(index - 2) + ']', offset, self.data_type)
        except:
            return None

# def display_array(valobj, internal_dict):
    # prov = ArrayProvider(valobj, None)
    # return 'Array'# + str(prov.num_children())

class StaticArrayProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.data = self.valobj.GetChildMemberWithName('data_')

    def num_children(self):
        return self.data.GetNumChildren()

    def get_child_at_index(self, index):
        return self.data.GetChildAtIndex(index)

class Array2DProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.data = self.valobj.GetChildMemberWithName('data_')
        self.data_type = self.data.GetType().GetPointeeType()
        self.type_size = self.data_type.GetByteSize()
        self.size_0 = (self.valobj.GetChildMemberWithName('size_').GetChildAtIndex(0).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.capacity_0 = (self.valobj.GetChildMemberWithName('capacity_').GetChildAtIndex(0).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.size_1 = (self.valobj.GetChildMemberWithName('size_').GetChildAtIndex(1).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.capacity_1 = (self.valobj.GetChildMemberWithName('capacity_').GetChildAtIndex(1).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size

    def num_children(self):
        return 4 + self.size_0 * self.size_1

    def get_child_at_index(self, index):
        if index < 0:
            return None
        if index >= self.num_children():
            return None
        try:
            my_frame = self.data.frame
            if index == 0:
                x = my_frame.EvaluateExpression('static_cast<il::int_t>(' + str(self.size_0) + ')')
                return x.CreateValueFromData('size(0)', x.GetData(), x.GetType())
            elif index == 1:
                x = my_frame.EvaluateExpression('static_cast<il::int_t>(' + str(self.size_1) + ')')
                return x.CreateValueFromData('size(1)', x.GetData(), x.GetType())
            elif index == 2:
                x = my_frame.EvaluateExpression('static_cast<il::int_t>(' + str(self.capacity_0) + ')')
                return x.CreateValueFromData('capacity(0)', x.GetData(), x.GetType())
            elif index == 3:
                x = my_frame.EvaluateExpression('static_cast<il::int_t>(' + str(self.capacity_1) + ')')
                return x.CreateValueFromData('capacity(1)', x.GetData(), x.GetType())
            else:
                i = (index - 4) % self.size_0
                j = (index - 4) // self.size_0
                offset = (j * self.capacity_0 + i) * self.type_size
                return self.data.CreateChildAtOffset(
                    '(' + str(i) + ", " + str(j) + ')', offset, self.data_type)
        except:
            return None

class Array2CProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.data = self.valobj.GetChildMemberWithName('data_')
        self.data_type = self.data.GetType().GetPointeeType()
        self.type_size = self.data_type.GetByteSize()
        self.size_0 = (self.valobj.GetChildMemberWithName('size_').GetChildAtIndex(0).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.capacity_0 = (self.valobj.GetChildMemberWithName('capacity_').GetChildAtIndex(0).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.size_1 = (self.valobj.GetChildMemberWithName('size_').GetChildAtIndex(1).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.capacity_1 = (self.valobj.GetChildMemberWithName('capacity_').GetChildAtIndex(1).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size

    def num_children(self):
        return self.size_0 * self.size_1

    def get_child_index(self, name):
        try:
            tmp = name.lstrip('[').rstrip(']')
            values = tmp.split(',')
            return int(values[1]) + int(values[0]) * self.size_1
        except:
            return -1

    def get_child_at_index(self, index):
        if index < 0:
            return None
        if index >= self.num_children():
            return None
        try:
            j = index % self.size_1
            i = index // self.size_1
            offset = (i * self.capacity_1 + j) * self.type_size
            return self.data.CreateChildAtOffset(
                '[' + str(i) + ", " + str(j) + ']', offset, self.data_type)
        except:
            return None

class StaticArray2DProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.data = self.valobj.GetChildMemberWithName('data_')
        self.size_0 = self.valobj.GetChildMemberWithName('size_0_')
        self.size_1 = self.valobj.GetChildMemberWithName('size_1_')

    def num_children(self):
        return self.data.GetNumChildren()

    def get_child_at_index(self, index):
        return self.data.GetChildAtIndex(index)

class Array3DProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.data = self.valobj.GetChildMemberWithName('data_')
        self.data_type = self.data.GetType().GetPointeeType()
        self.type_size = self.data_type.GetByteSize()
        self.size_0 = (self.valobj.GetChildMemberWithName('size_').GetChildAtIndex(0).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.capacity_0 = (self.valobj.GetChildMemberWithName('capacity_').GetChildAtIndex(0).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.size_1 = (self.valobj.GetChildMemberWithName('size_').GetChildAtIndex(1).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.capacity_1 = (self.valobj.GetChildMemberWithName('capacity_').GetChildAtIndex(1).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.size_2 = (self.valobj.GetChildMemberWithName('size_').GetChildAtIndex(2).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.capacity_2 = (self.valobj.GetChildMemberWithName('capacity_').GetChildAtIndex(2).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size

    def num_children(self):
        return self.size_0 * self.size_1 * self.size_2

    def get_child_index(self, name):
        try:
            tmp = name.lstrip('[').rstrip(']')
            values = tmp.split(',')
            return int(values[0]) + self.size_0 * (int(values[1]) + self.size_1 * int(values[2]))
        except:
            return -1

    def get_child_at_index(self, index):
        if index < 0:
            return None
        if index >= self.num_children():
            return None
        try:
            i = index % self.size_0
            j_tmp = index // self.size_0
            j = j_tmp % self.size_1
            k = j_tmp // self.size_1
            offset = (i + self.capacity_0 * (j + self.capacity_1 * k)) * self.type_size
            return self.data.CreateChildAtOffset(
                '[' + str(i) + ", " + str(j) + ", " + str(k) + ']', offset, self.data_type)
        except:
            return None

class Array3CProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.data = self.valobj.GetChildMemberWithName('data_')
        self.data_type = self.data.GetType().GetPointeeType()
        self.type_size = self.data_type.GetByteSize()
        self.size_0 = (self.valobj.GetChildMemberWithName('size_').GetChildAtIndex(0).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.capacity_0 = (self.valobj.GetChildMemberWithName('capacity_').GetChildAtIndex(0).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.size_1 = (self.valobj.GetChildMemberWithName('size_').GetChildAtIndex(1).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.capacity_1 = (self.valobj.GetChildMemberWithName('capacity_').GetChildAtIndex(1).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.size_2 = (self.valobj.GetChildMemberWithName('size_').GetChildAtIndex(2).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.capacity_2 = (self.valobj.GetChildMemberWithName('capacity_').GetChildAtIndex(2).GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size

    def num_children(self):
        return self.size_0 * self.size_1 * self.size_2

    def get_child_index(self, name):
        try:
            tmp = name.lstrip('[').rstrip(']')
            values = tmp.split(',')
            return int(values[2]) + self.size_2 * (int(values[1]) + self.size_1 * int(values[0]))
        except:
            return -1

    def get_child_at_index(self, index):
        if index < 0:
            return None
        if index >= self.num_children():
            return None
        try:
            k = index % self.size_2
            j_tmp = index // self.size_2
            j = j_tmp % self.size_1
            i = j_tmp // self.size_1
            offset = (k + self.capacity_2 * (j + self.capacity_1 * i)) * self.type_size
            return self.data.CreateChildAtOffset(
                '[' + str(i) + ", " + str(j) + ", " + str(k) + ']', offset, self.data_type)
        except:
            return None
