// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/EagerMath/Python/Magnitude.hpp"

#include <pybind11/pybind11.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace py = pybind11;

namespace py_bindings {

namespace {
template <typename TensorType>
void bind_magnitude_impl(py::module& m) {  // NOLINT
  m.def(
      "magnitude", [](const TensorType& vector) { return magnitude(vector); },
      py::arg("vector"));
  using MetricIndex =
      change_index_up_lo<tmpl::front<typename TensorType::index_list>>;
  using MetricType = Tensor<typename TensorType::type, Symmetry<1, 1>,
                            index_list<MetricIndex, MetricIndex>>;
  m.def(
      "magnitude",
      [](const TensorType& vector, const MetricType& metric) {
        return magnitude(vector, metric);
      },
      py::arg("vector"), py::arg("metric"));
}
}  // namespace

void bind_magnitude(py::module& m) {  // NOLINT
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define TENSOR(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATE_TNSR(_, data)                                    \
  bind_magnitude_impl < tnsr::TENSOR(data) < DTYPE(data), DIM(data), \
      FRAME(data) >> (m);

  GENERATE_INSTANTIATIONS(INSTANTIATE_TNSR, (double, DataVector), (1, 2, 3),
                          (Frame::Inertial), (i, I))

#undef INSTANTIATE_TNSR
#undef DTYPE
#undef DIM
#undef FRAME
#undef TENSOR
}

}  // namespace py_bindings
