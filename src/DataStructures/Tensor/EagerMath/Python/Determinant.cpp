// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/EagerMath/Python/Determinant.hpp"

#include <pybind11/pybind11.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace py = pybind11;

namespace py_bindings {

namespace {
template <typename TensorType>
void bind_determinant_impl(py::module& m) {  // NOLINT
  m.def(
      "determinant",
      [](const TensorType& tensor) { return determinant(tensor); },
      py::arg("tensor"));
}
}  // namespace

void bind_determinant(py::module& m) {
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define TENSOR(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATE_TNSR(_, data)                                      \
  bind_determinant_impl < tnsr::TENSOR(data) < DTYPE(data), DIM(data), \
      FRAME(data) >> (m);
#define INSTANTIATE_JAC(_, data)                                 \
  bind_determinant_impl < TENSOR(data) < DTYPE(data), DIM(data), \
      Frame::ElementLogical, FRAME(data) >> (m);

  GENERATE_INSTANTIATIONS(INSTANTIATE_TNSR, (double, DataVector), (1, 2, 3),
                          (Frame::Inertial), (ij, ii, II))
  GENERATE_INSTANTIATIONS(INSTANTIATE_JAC, (double, DataVector), (1, 2, 3),
                          (Frame::Inertial), (Jacobian, InverseJacobian))

#undef INSTANTIATE_TNSR
#undef INSTANTIATE_JAC
#undef DTYPE
#undef DIM
#undef FRAME
#undef TENSOR
}

}  // namespace py_bindings
