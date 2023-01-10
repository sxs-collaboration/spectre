// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/Python/PartialDerivatives.hpp"

#include <cstddef>
#include <pybind11/pybind11.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace py = pybind11;

namespace py_bindings {
namespace {
template <size_t Dim, typename DerivFrame, typename TensorType>
void bind_partial_derivatives_impl(py::module& m) {  // NOLINT
  m.def(
      "partial_derivative",
      // Wrap in a lambda so the template parameters and the correct overload
      // get inferred
      [](const TensorType& tensor, const Mesh<Dim>& mesh,
         const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                               DerivFrame>& inv_jacobian) {
        return partial_derivative(tensor, mesh, inv_jacobian);
      },
      py::arg("tensor"), py::arg("mesh"), py::arg("inv_jacobian"));
}
}  // namespace

void bind_partial_derivatives(py::module& m) {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define TNSR(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE_SCALAR(_, data) \
  bind_partial_derivatives_impl<DIM(data), FRAME(data), Scalar<DataVector>>(m);
#define INSTANTIATE_TNSR(_, data)                         \
  bind_partial_derivatives_impl < DIM(data), FRAME(data), \
      TNSR(data) < DataVector, DIM(data), FRAME(data) >> (m);

  GENERATE_INSTANTIATIONS(INSTANTIATE_SCALAR, (1, 2, 3), (Frame::Inertial))
  GENERATE_INSTANTIATIONS(INSTANTIATE_TNSR, (1, 2, 3), (Frame::Inertial),
                          (tnsr::I, tnsr::ii))

#undef INSTANTIATE_SCALAR
#undef INSTANTIATE_TNSR
#undef DIM
#undef FRAME
#undef TNSR
}
}  // namespace py_bindings
