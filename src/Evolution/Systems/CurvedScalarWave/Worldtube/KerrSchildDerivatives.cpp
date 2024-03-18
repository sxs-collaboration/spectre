
// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/KerrSchildDerivatives.hpp"

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {
tnsr::iAA<double, 3> spatial_derivative_inverse_ks_metric(
    const tnsr::I<double, 3>& pos) {
  const double r_sq = get(dot_product(pos, pos));
  const double r = sqrt(r_sq);
  const double one_over_r = 1. / r;
  const double one_over_r_2 = 1. / r_sq;
  const double one_over_r_3 = one_over_r_2 * one_over_r;

  tnsr::iAA<double, 3> di_imetric{};
  tnsr::ii<double, 3> delta_ll{0.};
  tnsr::Ij<double, 3> delta_ul{0.};
  tnsr::i<double, 3> pos_lower{};

  for (size_t i = 0; i < 3; ++i) {
    delta_ll.get(i, i) = 1.;
    delta_ul.get(i, i) = 1.;
    pos_lower.get(i) = pos.get(i);
  }

  const auto d_imetric_ij = tenex::evaluate<ti::i, ti::J, ti::K>(
      one_over_r_3 *
      (6. * pos(ti::J) * pos(ti::K) * pos_lower(ti::i) * one_over_r_2 -
       2. * delta_ul(ti::J, ti::i) * pos(ti::K) -
       2. * delta_ul(ti::K, ti::i) * pos(ti::J)));
  const auto d_imetric_i0 = tenex::evaluate<ti::i, ti::J>(
      one_over_r_2 * (-4. * pos_lower(ti::i) * pos(ti::J) * one_over_r_2 +
                      2. * delta_ul(ti::J, ti::i)));
  const auto d_imetric_00 =
      tenex::evaluate<ti::i>(2. * pos_lower(ti::i) * one_over_r_3);
  for (size_t i = 0; i < 3; ++i) {
    di_imetric.get(i, 0, 0) = d_imetric_00.get(i);
    for (size_t j = 0; j < 3; ++j) {
      di_imetric.get(i, j + 1, 0) = d_imetric_i0.get(i, j);
      for (size_t k = 0; k < 3; ++k) {
        di_imetric.get(i, j + 1, k + 1) = d_imetric_ij.get(i, j, k);
      }
    }
  }
  return di_imetric;
}

tnsr::iaa<double, 3> spatial_derivative_ks_metric(
    const tnsr::aa<double, 3>& metric,
    const tnsr::iAA<double, 3>& di_inverse_metric) {
  tnsr::iaa<double, 3> di_metric{};
  tenex::evaluate<ti::i, ti::a, ti::b>(
      make_not_null(&di_metric), -metric(ti::a, ti::c) * metric(ti::b, ti::d) *
                                     di_inverse_metric(ti::i, ti::C, ti::D));
  return di_metric;
}

}  // namespace CurvedScalarWave::Worldtube
