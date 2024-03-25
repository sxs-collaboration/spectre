
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

tnsr::iiAA<double, 3> second_spatial_derivative_inverse_ks_metric(
    const tnsr::I<double, 3>& pos) {
  const double r_sq = get(dot_product(pos, pos));
  const double r = sqrt(r_sq);
  const double one_over_r = 1. / r;
  const double one_over_r_2 = 1. / r_sq;
  const double one_over_r_3 = one_over_r_2 * one_over_r;
  const double one_over_r_4 = one_over_r_2 * one_over_r_2;

  tnsr::iiAA<double, 3> dij_imetric{};
  tnsr::ii<double, 3> delta_ll{0.};
  tnsr::Ij<double, 3> delta_ul{0.};
  tnsr::i<double, 3> pos_lower{};

  for (size_t i = 0; i < 3; ++i) {
    delta_ll.get(i, i) = 1.;
    delta_ul.get(i, i) = 1.;
    pos_lower.get(i) = pos.get(i);
  }

  const auto d2_imetric_ij = tenex::evaluate<ti::i, ti::j, ti::K, ti::L>(
      one_over_r_3 *
      (-2. * (delta_ul(ti::L, ti::i) * delta_ul(ti::K, ti::j) +
              delta_ul(ti::K, ti::i) * delta_ul(ti::L, ti::j)) +
       one_over_r_2 *
           (6. * (delta_ll(ti::i, ti::j) * pos(ti::K) * pos(ti::L) +
                  delta_ul(ti::K, ti::i) * pos_lower(ti::j) * pos(ti::L) +
                  delta_ul(ti::K, ti::j) * pos_lower(ti::i) * pos(ti::L) +
                  delta_ul(ti::L, ti::i) * pos_lower(ti::j) * pos(ti::K) +
                  delta_ul(ti::L, ti::j) * pos_lower(ti::i) * pos(ti::K)) -
            one_over_r_2 * 30. * pos_lower(ti::i) * pos_lower(ti::j) *
                pos(ti::K) * pos(ti::L))));

  const auto d2_imetric_i0 = tenex::evaluate<ti::j, ti::k, ti::I>(
      one_over_r_4 *
      (-4. * (delta_ll(ti::k, ti::j) * pos(ti::I) +
              delta_ul(ti::I, ti::k) * pos_lower(ti::j) +
              delta_ul(ti::I, ti::j) * pos_lower(ti::k)) +
       one_over_r_2 * 16. * pos(ti::I) * pos_lower(ti::j) * pos_lower(ti::k)));
  const auto d2_imetric_00 = tenex::evaluate<ti::i, ti::j>(
      one_over_r_3 * (2. * delta_ll(ti::i, ti::j) -
                      one_over_r_2 * 6. * pos_lower(ti::i) * pos_lower(ti::j)));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      dij_imetric.get(i, j, 0, 0) = d2_imetric_00.get(i, j);
      for (size_t k = 0; k < 3; ++k) {
        dij_imetric.get(i, j, k + 1, 0) = d2_imetric_i0.get(i, j, k);
        for (size_t l = 0; l < 3; ++l) {
          dij_imetric.get(i, j, k + 1, l + 1) = d2_imetric_ij.get(i, j, k, l);
        }
      }
    }
  }
  return dij_imetric;
}

tnsr::iiaa<double, 3> second_spatial_derivative_metric(
    const tnsr::aa<double, 3>& metric, const tnsr::iaa<double, 3>& di_metric,
    const tnsr::iAA<double, 3>& di_inverse_metric,
    const tnsr::iiAA<double, 3>& dij_inverse_metric) {
  tnsr::iiaa<double, 3> dij_metric{};
  tenex::evaluate<ti::j, ti::i, ti::a, ti::b>(
      make_not_null(&dij_metric),
      -metric(ti::a, ti::c) * metric(ti::b, ti::d) *
              dij_inverse_metric(ti::j, ti::i, ti::C, ti::D) -
          2. * metric(ti::a, ti::c) * di_metric(ti::j, ti::b, ti::d) *
              di_inverse_metric(ti::i, ti::C, ti::D));
  return dij_metric;
}

tnsr::iAbb<double, 3> spatial_derivative_christoffel(
    const tnsr::iaa<double, 3>& di_metric,
    const tnsr::iiaa<double, 3>& dij_metric,
    const tnsr::AA<double, 3>& inverse_metric,
    const tnsr::iAA<double, 3>& di_inverse_metric) {
  tnsr::iAbb<double, 3> di_christoffel{};
  tnsr::abb<double, 3> d_metric{};
  tnsr::iabb<double, 3> di_d_metric{};
  for (size_t a = 0; a <= 3; ++a) {
    for (size_t b = 0; b <= 3; ++b) {
      d_metric.get(0, a, b) = 0.;
      for (size_t i = 0; i < 3; ++i) {
        d_metric.get(i + 1, a, b) = di_metric.get(i, a, b);
        di_d_metric.get(i, 0, a, b) = 0.;
        for (size_t j = 0; j < 3; ++j) {
          di_d_metric.get(i, j + 1, a, b) = dij_metric.get(i, j, a, b);
        }
      }
    }
  }
  tenex::evaluate<ti::i, ti::A, ti::b, ti::c>(
      make_not_null(&di_christoffel),
      0.5 * di_inverse_metric(ti::i, ti::A, ti::D) *
              (d_metric(ti::b, ti::c, ti::d) + d_metric(ti::c, ti::b, ti::d) -
               d_metric(ti::d, ti::b, ti::c)) +
          0.5 * inverse_metric(ti::A, ti::D) *
              (di_d_metric(ti::i, ti::b, ti::c, ti::d) +
               di_d_metric(ti::i, ti::c, ti::b, ti::d) -
               di_d_metric(ti::i, ti::d, ti::b, ti::c)));
  return di_christoffel;
}

tnsr::iA<double, 3> spatial_derivative_ks_contracted_christoffel(
    const tnsr::I<double, 3>& pos) {
  const double r_sq = get(dot_product(pos, pos));
  const double r = sqrt(r_sq);
  const double one_over_r = 1. / r;
  const double one_over_r_2 = 1. / r_sq;
  const double one_over_r_3 = cube(one_over_r);
  const double one_over_r_4 = square(one_over_r_2);
  const double one_over_r_5 = one_over_r_4 * one_over_r;

  tnsr::iA<double, 3> di_contracted_christoffel{};
  for (size_t i = 0; i < 3; ++i) {
    di_contracted_christoffel.get(i, 0) = 4. * pos.get(i) * one_over_r_4;
    for (size_t j = 0; j < 3; ++j) {
      di_contracted_christoffel.get(i, j + 1) =
          -6. * pos.get(i) * pos.get(j) * one_over_r_5;
    }
    di_contracted_christoffel.get(i, i + 1) += 2. * one_over_r_3;
  }
  return di_contracted_christoffel;
}

}  // namespace CurvedScalarWave::Worldtube
