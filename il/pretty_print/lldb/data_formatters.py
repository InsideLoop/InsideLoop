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

import lldb
import lldb.formatters.Logger

def __lldb_init_module (debugger, dict):
    debugger.HandleCommand("type summary add -F data_formatters.display_string il::String")
    debugger.HandleCommand("type synthetic add -x \"il::Array<\" --python-class data_formatters.ArrayProvider")
    debugger.HandleCommand("type synthetic add -x \"il::StaticArray<\" --python-class data_formatters.StaticArrayProvider")
    debugger.HandleCommand("type synthetic add -x \"il::Array2D<\" --python-class data_formatters.Array2DProvider")
    debugger.HandleCommand("type synthetic add -x \"il::Array2C<\" --python-class data_formatters.Array2CProvider")
    debugger.HandleCommand("type synthetic add -x \"il::Array3D<\" --python-class data_formatters.Array3DProvider")
    debugger.HandleCommand("type synthetic add -x \"il::Array3C<\" --python-class data_formatters.Array3CProvider")


def display_string(valobj, internal_dict):
    ans = ''
    s = valobj.GetChildAtIndex(0).GetData().uint8
    for k in range(24):
       character = s[k]
       if character == 0:
           break
       ans = ans + chr(character % 256)
    return '"' + ans + '"'

class ArrayProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.data = self.valobj.GetChildMemberWithName('data_')
        self.data_type = self.data.GetType().GetPointeeType()
        self.type_size = self.data_type.GetByteSize()
        self.size = (self.valobj.GetChildMemberWithName('size_').GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size
        self.capacity = (self.valobj.GetChildMemberWithName('capacity_').GetValueAsUnsigned(0) - self.data.GetValueAsUnsigned(0)) / self.type_size

    def num_children(self):
        return self.size

    def get_child_index(self, name):
        try:
            return int(name.lstrip('[').rstrip(']'))
        except:
            return -1

    def get_child_at_index(self, index):
        if index < 0:
            return None
        if index >= self.num_children():
            return None
        try:
            offset = index * self.type_size
            return self.data.CreateChildAtOffset(
                '[' + str(index) + ']', offset, self.data_type)
        except:
            return None

class StaticArrayProvider:
    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.data = self.valobj.GetChildMemberWithName('data_').GetChildAtIndex(0)
        self.data_type = self.data.GetType()
        self.type_size = self.data_type.GetByteSize()
        self.size = self.valobj.GetChildMemberWithName('size_')

    def num_children(self):
        return self.size

    def get_child_index(self, name):
        try:
            return int(name.lstrip('[').rstrip(']'))
        except:
            return -1

    def get_child_at_index(self, index):
        if index < 0:
            return None
        if index >= self.num_children():
            return None
        try:
            return self.valobj.GetChildMemberWithName('data_').CreateChildAtIndex('[' + str(index) + ']', index)
        except:
            return None

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
        return self.size_0 * self.size_1

    def get_child_index(self, name):
        try:
            tmp = name.lstrip('[').rstrip(']')
            values = tmp.split(',')
            return int(values[0]) + int(values[1]) * self.size_0
        except:
            return -1

    def get_child_at_index(self, index):
        if index < 0:
            return None
        if index >= self.num_children():
            return None
        try:
            i = index % self.size_0
            j = index // self.size_0
            offset = (j * self.capacity_0 + i) * self.type_size
            return self.data.CreateChildAtOffset(
                '[' + str(i) + ", " + str(j) + ']', offset, self.data_type)
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
