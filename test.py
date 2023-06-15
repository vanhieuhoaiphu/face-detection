import numpy

arr = numpy.expand_dims([8, 8, 28], axis=-1)
print(arr)
arr = arr[..., 3]
print(arr)
