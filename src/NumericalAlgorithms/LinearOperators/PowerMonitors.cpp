// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"

#include <array>
#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace PowerMonitors {

template <size_t Dim>
void power_monitors(const gsl::not_null<std::array<DataVector, Dim>*> result,
                const DataVector& input_data_vector, const Mesh<Dim>& mesh) {
  // Get modal coefficients
  const ModalVector modal_coefficients =
      to_modal_coefficients(input_data_vector, mesh);

  double slice_sum = 0.0;
  size_t n_slice = 0;
  size_t n_stripe = 0;
  for (size_t sliced_dim = 0; sliced_dim < Dim; ++sliced_dim) {
    n_slice = mesh.extents().slice_away(sliced_dim).product();
    n_stripe = mesh.extents(sliced_dim);

    gsl::at(*result, sliced_dim).destructive_resize(n_stripe);

    for (size_t index = 0; index < n_stripe; ++index) {
      slice_sum = 0.0;
      for (SliceIterator si(mesh.extents(), sliced_dim, index); si; ++si) {
        slice_sum += square(modal_coefficients[si.volume_offset()]);
      }
      slice_sum /= n_slice;
      slice_sum = sqrt(slice_sum);

      gsl::at(*result, sliced_dim)[index] = slice_sum;
    }
  }
}

template <size_t Dim>
std::array<DataVector, Dim> power_monitors(
    const DataVector& input_data_vector, const Mesh<Dim>& mesh) {
  std::array<DataVector, Dim> result{};
  power_monitors(make_not_null(&result), input_data_vector, mesh);
  return result;
}

}  // namespace PowerMonitors

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<DataVector, DIM(data)>                                   \
  PowerMonitors::power_monitors(const DataVector& input_data_vector,           \
                                     const Mesh<DIM(data)>& mesh);             \
  template void PowerMonitors::power_monitors(                                 \
    const gsl::not_null<std::array<DataVector, DIM(data)>*> result,            \
    const DataVector& input_data_vector, const Mesh<DIM(data)>& mesh);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
