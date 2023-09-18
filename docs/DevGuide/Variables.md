\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Using Variables in SpECTRE {#variables_foundations}

# What is a Variables and Why Use Them?
Variables are a data structure that hold a contiguous memory block with Tensors
pointing to it. Variables temporaries allow you to declare temporary tensors and
scalars so that you can do all allocations needed for a computation at one time.
Since physical memory is shared between CPU cores, processes can't allocate in
parallel since they might try to allocate to the same chunk of memory. As more
CPU cores are used, this becomes a bottleneck, slowing down or stopping other
processes while memory is being allocated. Using a Variables to allocate all
memory needed at once can improve efficiency allowing the computation to
operate smoothly and uninterrupted.

# Defining a Variables of Temporary Tags
To define a Variables, you'll need the TempTensor and Variables headers
```cpp
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Variables.hpp"
```
this will give you access to temporary Scalars and Tensors we'll need to
allocate. You can define a Variables that allocates one Scalar with something
like this:
```cpp
Variables<tmpl::list<::Tags::TempScalar<0>>>
temp_buffer{get<0,0>(spatial_metric).size()};
```
Here, the Variables we've defined `temp_buffer` provides a tmpl::list with a
TempScalar inside as the template argument, this will allocate a single
temporary scalar. The size and DataType of the TempScalar is deduced by what's
inside the {}, you can provide any tensor or std::array with the correct size
needed. Now, to use the allocation you've made, you can do:
```cpp
auto& useful_scalar = get<::Tags::TempScalar<0>>(temp_buffer);
```

# Real Use Example
Now that we've got the basics, using them to allocate multiple Scalars and
Tensors is quite easy. For instance, let's say I need to allocate 2 scalars,
a spatial vector and 2 lower rank 2 tensors for my function.
```cpp
Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
::Tags::TempI<0, 3, Frame::Inertial>, ::Tags::Tempij<0, 3, Frame::Inertial>,
::Tags::Tempij<1, 3, Frame::Inertial>>>
temp_buffer{get<0,0>(spatial_metric).size()};
```
Here, when we allocate for the same type of scalar or tensor (rank 2 lower) the
way we distinguish multiple allocations is through the number within the <>.
Now, to use each individual allocation, we can do something like:
```cpp
auto& cool_scalar1 = get<::Tags::TempScalar<0>>(temp_buffer);
auto& cool_scalar2 = get<::Tags::TempScalar<1>>(temp_buffer);
auto& cool_tensor1 = get<::Tags::Tempij<0, 3, Frame::Inertial>>(temp_buffer);
auto& cool_tensor2 = get<::Tags::Tempij<1, 3, Frame::Inertial>>(temp_buffer);
```

# Tips
In the interest of reducing memory allocations, there a certain scenarios where
you can resuse old allocations that are no longer useful to your computation.

To see this, let's say that you're trying to make two unit vectors. You might
start by saying you'll need two different vectors (rank 1 upper tensors) and 2
different scalars as the magnitude of each vector.
The way we'd allocate for this is by doing:
```cpp
Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
::Tags::TempI<0, 3, Frame::Inertial>, ::Tags::TempI<1, 3, Frame::Inertial>>>
temp_buffer{get<0,0>(spatial_metric).size()};
```
However, doing this allocates more memory than we actually need. Once we finish
calculating the first unit vector, the memory we've allocated for the scalar
magnitude of the first vector will just sit there unused. We can reuse the
allocation for the TempScalar<0> and use it when calculating the second unit
vector without having to allocate another TempScalar. Now, allocating an extra
scalar is not very expensive, but when using tensors, the memory required really
adds up, so this is just another way to help make SpECTRE a bit more efficient.
