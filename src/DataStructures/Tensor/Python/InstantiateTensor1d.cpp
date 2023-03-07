// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/Python/Tensor.tpp"

namespace py_bindings {
template void bind_tensor<1>(pybind11::module& m);
}  // namespace py_bindings
