// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts {

template <typename DataType>
void longitudinal_operator(const gsl::not_null<tnsr::II<DataType, 3>*> result,
                           const tnsr::ii<DataType, 3>& strain,
                           const tnsr::II<DataType, 3>& inv_metric) {
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      // Unroll first iteration of the loop over `k` to avoid filling the result
      // with zero initially. This first assignment is the k=0, l=0 iteration:
      result->get(i, j) =
          (2. * inv_metric.get(i, 0) * inv_metric.get(j, 0) -
           2. / 3. * inv_metric.get(i, j) * get<0, 0>(inv_metric)) *
          get<0, 0>(strain);
      // These are the remaining contributions of the k=0 iteration:
      for (size_t l = 1; l < 3; ++l) {
        result->get(i, j) +=
            (2. * inv_metric.get(i, 0) * inv_metric.get(j, l) -
             2. / 3. * inv_metric.get(i, j) * inv_metric.get(0, l)) *
            strain.get(0, l);
      }
      // This is the loop from which the k=0 iteration is unrolled above:
      for (size_t k = 1; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          result->get(i, j) +=
              (2. * inv_metric.get(i, k) * inv_metric.get(j, l) -
               2. / 3. * inv_metric.get(i, j) * inv_metric.get(k, l)) *
              strain.get(k, l);
        }
      }
    }
  }
}

template <typename DataType>
void longitudinal_operator(
    const gsl::not_null<tnsr::II<DataType, 3>*> result,
    const tnsr::I<DataType, 3>& shift, const tnsr::iJ<DataType, 3>& deriv_shift,
    const tnsr::II<DataType, 3>& inv_metric,
    const tnsr::Ijj<DataType, 3>& christoffel_second_kind) {
  set_number_of_grid_points(result, shift);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      result->get(i, j) = 0.;
      for (size_t k = 0; k < 3; ++k) {
        result->get(i, j) +=
            inv_metric.get(i, k) * deriv_shift.get(k, j) +
            inv_metric.get(j, k) * deriv_shift.get(k, i) -
            2. / 3. * inv_metric.get(i, j) * deriv_shift.get(k, k);
        for (size_t l = 0; l < 3; ++l) {
          result->get(i, j) +=
              inv_metric.get(i, k) * christoffel_second_kind.get(j, k, l) *
                  shift.get(l) +
              inv_metric.get(j, k) * christoffel_second_kind.get(i, k, l) *
                  shift.get(l) -
              2. / 3. * inv_metric.get(i, j) *
                  christoffel_second_kind.get(k, k, l) * shift.get(l);
        }
      }
    }
  }
}

template <typename DataType>
void longitudinal_operator_flat_cartesian(
    const gsl::not_null<tnsr::II<DataType, 3>*> result,
    const tnsr::ii<DataType, 3>& strain) {
  // Compute trace term in 2-2 component of the result
  get<2, 2>(*result) = get<0, 0>(strain);
  for (size_t d = 1; d < 3; ++d) {
    get<2, 2>(*result) += strain.get(d, d);
  }
  get<2, 2>(*result) *= -2. / 3.;
  for (size_t i = 0; i < 3; ++i) {
    // Copy trace term to other diagonal components and complete diagonal
    // components with non-trace contribution
    result->get(i, i) = get<2, 2>(*result) + 2. * strain.get(i, i);
    // Compute off-diagonal contributions
    for (size_t j = 0; j < i; ++j) {
      result->get(i, j) = 2. * strain.get(i, j);
    }
  }
}

template <typename DataType>
void longitudinal_operator_flat_cartesian(
    const gsl::not_null<tnsr::II<DataType, 3>*> result,
    const tnsr::iJ<DataType, 3>& deriv_shift) {
  // Compute trace term in 2-2 component of the result
  get<2, 2>(*result) = get<0, 0>(deriv_shift);
  for (size_t d = 1; d < 3; ++d) {
    get<2, 2>(*result) += deriv_shift.get(d, d);
  }
  get<2, 2>(*result) *= -2. / 3.;
  for (size_t i = 0; i < 3; ++i) {
    // Copy trace term to other diagonal components and complete diagonal
    // components with non-trace contribution
    result->get(i, i) = get<2, 2>(*result) + 2. * deriv_shift.get(i, i);
    // Compute off-diagonal contributions
    for (size_t j = 0; j < i; ++j) {
      result->get(i, j) = deriv_shift.get(i, j) + deriv_shift.get(j, i);
    }
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                     \
  template void longitudinal_operator(                           \
      gsl::not_null<tnsr::II<DTYPE(data), 3>*> result,           \
      const tnsr::ii<DTYPE(data), 3>& strain,                    \
      const tnsr::II<DTYPE(data), 3>& inv_metric);               \
  template void longitudinal_operator(                           \
      gsl::not_null<tnsr::II<DTYPE(data), 3>*> result,           \
      const tnsr::I<DTYPE(data), 3>& shift,                      \
      const tnsr::iJ<DTYPE(data), 3>& deriv_shift,               \
      const tnsr::II<DTYPE(data), 3>& inv_metric,                \
      const tnsr::Ijj<DTYPE(data), 3>& christoffel_second_kind); \
  template void longitudinal_operator_flat_cartesian(            \
      gsl::not_null<tnsr::II<DTYPE(data), 3>*> result,           \
      const tnsr::ii<DTYPE(data), 3>& strain);                   \
  template void longitudinal_operator_flat_cartesian(            \
      gsl::not_null<tnsr::II<DTYPE(data), 3>*> result,           \
      const tnsr::iJ<DTYPE(data), 3>& deriv_shift);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE

}  // namespace Xcts
