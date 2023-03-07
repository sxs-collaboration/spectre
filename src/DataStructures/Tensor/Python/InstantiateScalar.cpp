// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/Python/Tensor.tpp"

namespace py_bindings {

void bind_scalar(pybind11::module& m) {
  bind_tensor_impl<Scalar<DataVector>, TensorKind::Scalar>(m, "Scalar");
  bind_tensor_impl<Scalar<double>, TensorKind::Scalar>(m, "Scalar");
}

}  // namespace py_bindings
