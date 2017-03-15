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
		# if self.val['alignement_'] != 0:
		yield "alignment", self.val['alignment_']
		# yield "alignment_r", self.val['align_r_']
		# yield "alignment_mod", self.val['align_mod_']
		for k in range(0, self.size):
			dataPtr = self.data + k
			item = dataPtr.dereference()
			yield ("[%s]" % k), item

	def to_string(self):
		return "[size: %s], [capacity: %s]" % (self.size, self.capacity)

class StaticArrayPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.data = self.val['data_'].cast(self.innerType.pointer())
		self.size = self.val['size_']

	def children(self):
		yield "size", self.size
		for k in range(0, self.size):
			dataPtr = self.data + k
			item = dataPtr.dereference()
			yield ("[%s]" % k), item

	def to_string(self):
		return "[size: %s]" % (self.size)

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
		yield "alignment", self.val['alignment_']
		for k1 in range(0, self.size1):
			for k0 in range(0, self.size0):
				dataPtr = self.data + self.capacity0 * k1 + k0
				item = dataPtr.dereference()
				yield ("[%s, %s]" % (k0, k1)), item

	def to_string(self):
		return "[size0: %s], [size1: %s], [capacity0: %s], [capacity1: %s]" % (self.size0, self.size1, self.capacity0, self.capacity1)

class StaticArray2DPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.data = self.val['data_'].cast(self.innerType.pointer())
		self.size0 = self.val['size_0_']
		self.size1 = self.val['size_1_']

	def children(self):
		yield "size_0", self.size0
		yield "size_1", self.size1
		for k1 in range(0, self.size1):
			for k0 in range(0, self.size0):
				dataPtr = self.data + self.size0 * k1 + k0
				item = dataPtr.dereference()
				yield ("[%s, %s]" % (k0, k1)), item

	def to_string(self):
		return "[size0: %s], [size1: %s]" % (self.size0, self.size1)

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

class StaticArray2CPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.data = self.val['data_'].cast(self.innerType.pointer())
		self.size0 = self.val['size_0_']
		self.size1 = self.val['size_1_']

	def children(self):
		yield "size_0", self.size0
		yield "size_1", self.size1
		for k0 in range(0, self.size0):
			for k1 in range(0, self.size1):
				dataPtr = self.data + self.size1 * k0 + k1
				item = dataPtr.dereference()
				yield ("[%s, %s]" % (k0, k1)), item

	def to_string(self):
		return "[size0: %s], [size1: %s]" % (self.size0, self.size1)

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

class SparseMatrixCSRPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.size0 = self.val['n0_']
		self.size1 = self.val['n1_']
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

class StaticArray3DPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.data = self.val['data_'].cast(self.innerType.pointer())
		self.size0 = self.val['size_0_']
		self.size1 = self.val['size_1_']
		self.size2 = self.val['size_2_']

	def children(self):
		yield "size_0", self.size0
		yield "size_1", self.size1
		yield "size_2", self.size2
		for k2 in range(0, self.size2):
			for k1 in range(0, self.size1):
				for k0 in range(0, self.size0):
					dataPtr = self.data + self.size0 * self.size1 * k2 + self.size0 * k1 + k0
					item = dataPtr.dereference()
					yield ("[%s, %s, %s]" % (k0, k1, k2)), item

	def to_string(self):
		return "[size0: %s], [size1: %s], [size2: %s]" % (self.size0, self.size1, self.size2)

class Array4DPrinter:
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
		self.size3 = self.val['size_'][3] - self.data
		self.capacity0 = self.val['capacity_'][0] - self.data
		self.capacity1 = self.val['capacity_'][1] - self.data
		self.capacity2 = self.val['capacity_'][2] - self.data
		self.capacity3 = self.val['capacity_'][3] - self.data

	def children(self):
		yield "size_0", self.size0
		yield "size_1", self.size1
		yield "size_2", self.size2
		yield "size_3", self.size3
		yield "capacity_0", self.capacity0
		yield "capacity_1", self.capacity1
		yield "capacity_2", self.capacity2
		yield "capacity_3", self.capacity3
		for k3 in range(0, self.size3):
			for k2 in range(0, self.size2):
				for k1 in range(0, self.size1):
					for k0 in range(0, self.size0):
						dataPtr = self.data + ((k3 * self.capacity2 + k2) * self.capacity1 + k1) * self.capacity0 + k0
						item = dataPtr.dereference()
						yield ("[%s, %s, %s, %s]" % (k0, k1, k2, k3)), item

	def to_string(self):
		return "[size0: %s], [size1: %s], [size2: %s], [size3: %s], [capacity0: %s], [capacity1: %s], [capacity2: %s], [capacity3: %s]" % (self.size0, self.size1, self.size2, self.size3, self.capacity0, self.capacity1, self.capacity2, self.capacity3)

class Array4CPrinter:
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
		self.size3 = self.val['size_'][3] - self.data
		self.capacity0 = self.val['capacity_'][0] - self.data
		self.capacity1 = self.val['capacity_'][1] - self.data
		self.capacity2 = self.val['capacity_'][2] - self.data
		self.capacity3 = self.val['capacity_'][3] - self.data

	def children(self):
		yield "size_0", self.size0
		yield "size_1", self.size1
		yield "size_2", self.size2
		yield "size_3", self.size3
		yield "capacity_0", self.capacity0
		yield "capacity_1", self.capacity1
		yield "capacity_2", self.capacity2
		yield "capacity_3", self.capacity3
		for k0 in range(0, self.size0):
			for k1 in range(0, self.size1):
				for k2 in range(0, self.size2):
					for k3 in range(0, self.size3):
						dataPtr = self.data + ((k0 * self.capacity1 + k1) * self.capacity2 + k2) * self.capacity3 + k3
						item = dataPtr.dereference()
						yield ("[%s, %s, %s, %s]" % (k0, k1, k2, k3)), item

	def to_string(self):
		return "[size0: %s], [size1: %s], [size2: %s], [size3: %s], [capacity0: %s], [capacity1: %s], [capacity2: %s], [capacity3: %s]" % (self.size0, self.size1, self.size2, self.size3, self.capacity0, self.capacity1, self.capacity2, self.capacity3)

class StaticArray4DPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		self.data = self.val['data_'].cast(self.innerType.pointer())
		self.size0 = self.val['size_0_']
		self.size1 = self.val['size_1_']
		self.size2 = self.val['size_2_']
		self.size3 = self.val['size_3_']

	def children(self):
		yield "size_0", self.size0
		yield "size_1", self.size1
		yield "size_2", self.size2
		yield "size_3", self.size3
		for k3 in range(0, self.size3):
			for k2 in range(0, self.size2):
				for k1 in range(0, self.size1):
					for k0 in range(0, self.size0):
						dataPtr = self.data + self.size0 * self.size1 * self.size2 * k3 + self.size0 * self.size1 * k2 + self.size0 * k1 + k0
						item = dataPtr.dereference()
						yield ("[%s, %s, %s, %s]" % (k0, k1, k2, k3)), item

	def to_string(self):
		return "[size0: %s], [size1: %s], [size2: %s], [size3: %s]" % (self.size0, self.size1, self.size2, self.size3)

class StringViewPrinter:
	def __init__(self, val):
		self.val = val
		self.size = self.val['size_'] - self.val['data_']
		self.string = ""
		for k in range(0, self.size):
			self.string += chr(self.val['data_'][k])

	def to_string(self):
		return "\"%s\"" % self.string

class ConstStringViewPrinter:
	def __init__(self, val):
		self.val = val
		self.size = self.val['size_'] - self.val['data_']
		self.string = ""
		for k in range(0, self.size):
			self.string += chr(self.val['data_'][k])

	def to_string(self):
		return "\"%s\"" % self.string

class StringPrinter:
	def __init__(self, val):
		self.val = val
		if self.val['large_']['capacity_'] >= 2**63:
			self.is_small = False
			self.size = self.val['large_']['size']
			self.capacity = self.val['large_']['capacity_'] - 2**63
			self.string = ""
			for k in range(0, self.size):
				self.string += chr(self.val['large_']['data'][k])
		else:
			self.is_small = True
			self.size = 23 - self.val['small_'][23]
			self.capacity = 23
			self.string = ""
			for k in range(0, self.size):
				self.string += chr(self.val['small_'][k])

	def to_string(self):
		# return "[string: \"%s\"] [size: %s] [capacity: %s] [is small: %s]" % (self.string, self.size, self.capacity, self.is_small)
		return "\"%s\"" % self.string

class HashMapPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.keyType = self.type.template_argument(0)
		self.valueType = self.type.template_argument(1)
		self.val = val
		self.size = self.val['nb_element_']
		self.capacity = 2 ** self.val['p_']
		self.val = val
		self.a = gdb.parse_and_eval("(*("+str(self.val.type)+"*)("+str(self.val.address)+")).first()")

	def children(self):
		yield "size", self.size
		yield "capacity", self.capacity
		i = self.a
		for k in range(0, self.size):
			yield ("[key: %s]" % k), gdb.parse_and_eval("(*("+str(self.val.type)+"*)("+str(self.val.address)+")).key("+str(i)+")")
			yield ("[value: %s]" % k), gdb.parse_and_eval("(*("+str(self.val.type)+"*)("+str(self.val.address)+")).const_value("+str(i)+")")
			i = gdb.parse_and_eval("(*("+str(self.val.type)+"*)("+str(self.val.address)+")).next("+str(i)+")")

	def to_string(self):
		return "HashMap"

class InfoPrinter:
	def __init__(self, val):
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.val = val
		if self.val['large_']['capacity'] >= 2**63:
			self.is_small = False
			self.size = self.val['large_']['size']
			self.capacity = self.val['large_']['capacity'] - 2**63
			self.data = self.val['large_']['data']
		else:
			self.is_small = True
			self.size = 0 + self.val['small_'][23]
			self.capacity = 23
			self.data = self.val['small_']

	def children(self):
		k = 0
		while k != self.size:
			delta = self.data[k]
			k += 4
			key = ""
			char = chr(self.data[k])
			while char != '\0':
				key += char
				k += 1
				char = chr(self.data[k])
			k += 1
			type = self.data[k]
			k += 1
			if type == 0:
				value = 0
				factor = 1
				for i in range(0, 8):
					value += factor * self.data[k]
					factor *= 256
					k += 1
				yield key, value
			elif type == 1:
				yield key, None
			elif type == 2:
				begin = k
				char = chr(self.data[k])
				while char != '\0':
					k += 1
					char = chr(self.data[k])
				k += 1
				yield key, self.data + begin
			elif type == 3:
				value = 0
				factor = 1
				for i in range(0, 4):
					value += factor * self.data[k]
					factor *= 256
					k += 1
				yield key, value

	def to_string(self):
		return "Info"

def build_insideloop_dictionary ():
	pretty_printers_dict[re.compile('^il::Array<.*>$')]  = lambda val: ArrayPrinter(val)
	pretty_printers_dict[re.compile('^il::StaticArray<.*>$')]  = lambda val: StaticArrayPrinter(val)
	pretty_printers_dict[re.compile('^il::SmallArray<.*>$')]  = lambda val: SmallArrayPrinter(val)
	pretty_printers_dict[re.compile('^il::ArrayView<.*>$')]  = lambda val: ArrayViewPrinter(val)
	pretty_printers_dict[re.compile('^il::ConstArrayView<.*>$')]  = lambda val: ArrayViewPrinter(val)
	pretty_printers_dict[re.compile('^il::Array2D<.*>$')]  = lambda val: Array2DPrinter(val)
	pretty_printers_dict[re.compile('^il::Array2C<.*>$')]  = lambda val: Array2CPrinter(val)
	pretty_printers_dict[re.compile('^il::StaticArray2D<.*>$')]  = lambda val: StaticArray2DPrinter(val)
	pretty_printers_dict[re.compile('^il::StaticArray2C<.*>$')]  = lambda val: StaticArray2CPrinter(val)
	pretty_printers_dict[re.compile('^il::LowerArray2D<.*>$')]  = lambda val: LowerArray2DPrinter(val)
	pretty_printers_dict[re.compile('^il::TriDiagonal<.*>$')]  = lambda val: TriDiagonalPrinter(val)
	pretty_printers_dict[re.compile('^il::SparseMatrixCSR<.*>$')]  = lambda val: SparseMatrixCSRPrinter(val)
	pretty_printers_dict[re.compile('^il::Array3D<.*>$')]  = lambda val: Array3DPrinter(val)
	pretty_printers_dict[re.compile('^il::StaticArray3D<.*>$')]  = lambda val: StaticArray3DPrinter(val)
	pretty_printers_dict[re.compile('^il::Array4D<.*>$')]  = lambda val: Array4DPrinter(val)
	pretty_printers_dict[re.compile('^il::Array4C<.*>$')]  = lambda val: Array4CPrinter(val)
	pretty_printers_dict[re.compile('^il::StaticArray4D<.*>$')]  = lambda val: StaticArray4DPrinter(val)
	pretty_printers_dict[re.compile('^il::String$')]  = lambda val: StringPrinter(val)
	pretty_printers_dict[re.compile('^il::StringView$')]  = lambda val: StringViewPrinter(val)
	pretty_printers_dict[re.compile('^il::ConstStringView$')]  = lambda val: ConstStringViewPrinter(val)
	# pretty_printers_dict[re.compile('^il::HashMap<.*>$')]  = lambda val: HashMapPrinter(val)
	pretty_printers_dict[re.compile('^il::Info$')]  = lambda val: InfoPrinter(val)

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
