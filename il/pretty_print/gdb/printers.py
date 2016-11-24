# ==============================================================================
# 
#                                   InsideLoop
# 
#  This file is distributed under the University of Illinois Open Source
#  License. See LICENSE.txt for details.
#
# ==============================================================================
#
#
# To use it:
#
# * Create a directory and put the file as well as an empty __init__.py in 
#   that directory.
# * Create a ~/.gdbinit file, that contains the following:
#      python
#      import sys
#      sys.path.insert(0, '/path/to/insideloop/printer/directory')
#      from printers import register_eigen_printers
#      register_insideloop_printers (None)
#      end

import gdb
import re

class ArrayPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.data = self.val['data_'].cast(self.innerType.pointer())
		self.size = self.val['size_'] - self.val['data_']
		self.capacity = self.val['capacity_'] - self.val['data_']

	def children(self):
		yield "size", self.size
		yield "capacity", self.capacity
		yield "alignment_r", self.val['align_r_']
		yield "alignment_mod", self.val['align_mod_']
		for k in range(0, self.size):
			dataPtr = self.data + k
			item = dataPtr.dereference()
			yield ("[%s]" % k), item

	def to_string(self):
		return "[size: %s], [capacity: %s]" % (self.size, self.capacity)

class SmallArrayPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.data = self.val['data_'].cast(self.innerType.pointer())
		self.size = self.val['size_'] - self.val['data_']
		self.capacity = self.val['capacity_'] - self.val['data_']

	def children(self):
		yield "size", self.size
		yield "capacity", self.capacity
		# yield "using_small_array", self.data == self.val['small_data_']
		for k in range(0, self.size):
			dataPtr = self.data + k
			item = dataPtr.dereference()
			yield ("[%s]" % k), item

	def to_string(self):
		return "[size: %s], [capacity: %s]" % (self.size, self.capacity)

class ArrayViewPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.data = self.val['data_'].cast(self.innerType.pointer())
		self.size = self.val['size_'] - self.val['data_']

	def children(self):
		yield "size", self.size
		for k in range(0, self.size):
			dataPtr = self.data + k
			item = dataPtr.dereference()
			yield ("[%s]" % k), item

	def to_string(self):
		return "[size: %s], [capacity: %s]" % (self.size, self.capacity)

class Array2DPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.data = self.val['data_'].cast(self.innerType.pointer())
		self.size0 = self.val['size_'][0] - self.data
		self.size1 = self.val['size_'][1] - self.data
		self.capacity0 = self.val['capacity_'][0] - self.data
		self.capacity1 = self.val['capacity_'][1] - self.data

	def children(self):
		yield "size_0", self.size0
		yield "size_1", self.size1
		yield "capacity_0", self.capacity0
		yield "capacity_1", self.capacity1
		for k1 in range(0, self.size1):
			for k0 in range(0, self.size0):
				dataPtr = self.data + self.capacity0 * k1 + k0
				item = dataPtr.dereference()
				yield ("[%s, %s]" % (k0, k1)), item

	def to_string(self):
		return "[size0: %s], [size1: %s], [capacity0: %s], [capacity1: %s]" % (self.size0, self.size1, self.capacity0, self.capacity1)

class Array2CPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.data = self.val['data_'].cast(self.innerType.pointer())
		self.size0 = self.val['size_'][0] - self.data
		self.size1 = self.val['size_'][1] - self.data
		self.capacity0 = self.val['capacity_'][0] - self.data
		self.capacity1 = self.val['capacity_'][1] - self.data

	def children(self):
		yield "size_0", self.size0
		yield "size_1", self.size1
		yield "capacity_0", self.capacity0
		yield "capacity_1", self.capacity1
		for k0 in range(0, self.size0):
			for k1 in range(0, self.size1):
				dataPtr = self.data + self.capacity1 * k0 + k1
				item = dataPtr.dereference()
				yield ("[%s, %s]" % (k0, k1)), item

	def to_string(self):
		return "[size0: %s], [size1: %s], [capacity0: %s], [capacity1: %s]" % (self.size0, self.size1, self.capacity0, self.capacity1)

class TriDiagonalPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.size = self.val['size_']
		self.data = self.val['data_']['data_'].cast(self.innerType.pointer())

	def children(self):
		yield "size", self.size
		dataPtr = self.data
		if self.size > 0:
			yield ("[%s, %s]" % (0, 0)), (dataPtr + self.size).dereference()
			yield ("[%s, %s]" % (0, 1)), (dataPtr + 2 * self.size).dereference()
		for i in range(1, self.size - 1):
			yield ("[%s, %s]" % (i, i - 1)), (dataPtr + i).dereference()
			yield ("[%s, %s]" % (i, i)), (dataPtr + self.size + i).dereference()
			yield ("[%s, %s]" % (i, i + 1)), (dataPtr + 2 * self.size + i).dereference()
		if self.size > 1:
			i = self.size - 1
			yield ("[%s, %s]" % (i, i)), (dataPtr + self.size + i).dereference()
			yield ("[%s, %s]" % (i, i + 1)), (dataPtr + 2 * self.size + i).dereference()

	def to_string(self):
		return "[size0: %s], [size1: %s], [capacity0: %s], [capacity1: %s]" % (self.size0, self.size1, self.capacity0, self.capacity1)

class LowerArray2DPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.data = self.val['data_'].cast(self.innerType.pointer())
		self.size = self.val['size_'] - self.data
		self.capacity = self.val['capacity_'] - self.data

	def children(self):
		yield "size", self.size
		yield "capacity", self.capacity
		for k1 in range(0, self.size):
			for k0 in range(k1, self.size):
				dataPtr = self.data + (k1 * (2 * self.size - (1 + k1))) // 2 + k0
				item = dataPtr.dereference()
				yield ("[%s, %s]" % (k0, k1)), item

	def to_string(self):
		return "[size: %s], [capacity: %s]" % (self.size, self.capacity)

class SparseArray2CPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.size0 = self.val['height_']
		self.size1 = self.val['width_']
		self.non_zeros = self.val['element_']['size_'] - self.val['element_']['data_']

	def children(self):
		yield "nb_rows", self.size0
		yield "nb_cols", self.size1
		yield "nb_nonzeros", self.non_zeros
		elementPtr = self.val['element_']['data_'].cast(self.innerType.pointer())
		int_pointer_type = gdb.lookup_type('int').pointer()
		rowPtr = self.val['row_']['data_'].cast(int_pointer_type)
		columnPtr = self.val['column_']['data_'].cast(int_pointer_type)
		i = 0
		j = 0
		for k in range(0, self.non_zeros):
			item = elementPtr.dereference()
			yield ("[%s, %s]" % (i, columnPtr.dereference())), item
			elementPtr += 1
			columnPtr += 1
			j += 1
			if j >= (rowPtr + 1).dereference():
				rowPtr += 1
				i += 1


	def to_string(self):
		return "[size0: %s], [size1: %s]" % (self.size0, self.size1)

class Array3DPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.data = self.val['data_'].cast(self.innerType.pointer())
		self.size0 = self.val['size_'][0] - self.data
		self.size1 = self.val['size_'][1] - self.data
		self.size2 = self.val['size_'][2] - self.data
		self.capacity0 = self.val['capacity_'][0] - self.data
		self.capacity1 = self.val['capacity_'][1] - self.data
		self.capacity2 = self.val['capacity_'][2] - self.data

	def children(self):
		yield "size_0", self.size0
		yield "size_1", self.size1
		yield "size_2", self.size2
		yield "capacity_0", self.capacity0
		yield "capacity_1", self.capacity1
		yield "capacity_2", self.capacity2
		for k2 in range(0, self.size2):
			for k1 in range(0, self.size1):
				for k0 in range(0, self.size0):
					dataPtr = self.data + self.capacity0 * self.capacity1 * k2 + self.capacity0 * k1 + k0
					item = dataPtr.dereference()
					yield ("[%s, %s, %s]" % (k0, k1, k2)), item

	def to_string(self):
		return "[size0: %s], [size1: %s], [capacity0: %s], [capacity1: %s]" % (self.size0, self.size1, self.capacity0, self.capacity1)

def build_insideloop_dictionary ():
	pretty_printers_dict[re.compile('^il::Array<.*>$')]  = lambda val: ArrayPrinter(val)
	pretty_printers_dict[re.compile('^il::SmallArray<.*>$')]  = lambda val: SmallArrayPrinter(val)
	pretty_printers_dict[re.compile('^il::ArrayView<.*>$')]  = lambda val: ArrayViewPrinter(val)
	pretty_printers_dict[re.compile('^il::ConstArrayView<.*>$')]  = lambda val: ArrayViewPrinter(val)
	pretty_printers_dict[re.compile('^il::Array2D<.*>$')]  = lambda val: Array2DPrinter(val)
	pretty_printers_dict[re.compile('^il::Array2C<.*>$')]  = lambda val: Array2CPrinter(val)
	pretty_printers_dict[re.compile('^il::LowerArray2D<.*>$')]  = lambda val: LowerArray2DPrinter(val)
	pretty_printers_dict[re.compile('^il::TriDiagonal<.*>$')]  = lambda val: TriDiagonalPrinter(val)
	pretty_printers_dict[re.compile('^il::SparseArray2C<.*>$')]  = lambda val: SparseArray2CPrinter(val)
	pretty_printers_dict[re.compile('^il::Array3D<.*>$')]  = lambda val: Array3DPrinter(val)

def register_insideloop_printers(obj):
	"Register insideloop pretty-printers with objfile Obj"

	if obj == None:
		obj = gdb
	obj.pretty_printers.append(lookup_function)

def lookup_function(val):
	"Look-up and return a pretty-printer that can print val."

	type = val.type

	if type.code == gdb.TYPE_CODE_REF:
		type = type.target()

	type = type.unqualified().strip_typedefs()

	typename = type.tag
	if typename == None:
		return None

	for function in pretty_printers_dict:
		if function.search(typename):
			return pretty_printers_dict[function](val)

	return None

pretty_printers_dict = {}

build_insideloop_dictionary ()
