// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/ElectricCurrentDensity.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ForceFree {

namespace {

template <bool IncludeDriftCurrent, bool IncludeParallelCurrent>
void tilde_j_impl(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_j,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const double parallel_conductivity, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) {
  static_assert(IncludeDriftCurrent or IncludeParallelCurrent);

  Variables<tmpl::list<::Tags::Tempi<0, 3>, ::Tags::Tempi<1, 3>,
                       ::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>>>
      buffer{get(lapse).size()};

  // compute one-forms of TildeE and TildeB in advance to reduce the number of
  // dot products using spatial metric (which are slower than dot products
  // without using spatial metric)
  auto& tilde_e_one_form = get<::Tags::Tempi<0, 3>>(buffer);
  auto& tilde_b_one_form = get<::Tags::Tempi<1, 3>>(buffer);
  raise_or_lower_index(make_not_null(&tilde_e_one_form), tilde_e,
                       spatial_metric);
  raise_or_lower_index(make_not_null(&tilde_b_one_form), tilde_b,
                       spatial_metric);

  // Compute \tilde{B}^2 = \tilde{B}^j \tilde{B}_j. We need this quantity for
  // both drift (explicit, non-stiff) and parallel (implicit, stiff) components
  // of J^i.
  auto& tilde_b_squared = get<::Tags::TempScalar<0>>(buffer);
  dot_product(make_not_null(&tilde_b_squared), tilde_b, tilde_b_one_form);

  if constexpr (IncludeParallelCurrent) {
    (void)tilde_q;                  // avoid compiler warnings
    (void)sqrt_det_spatial_metric;  // avoid compiler warnings

    auto& tilde_e_squared = get<::Tags::TempScalar<1>>(buffer);
    auto& tilde_e_dot_tilde_b = get<::Tags::TempScalar<2>>(buffer);
    dot_product(make_not_null(&tilde_e_squared), tilde_e, tilde_e_one_form);
    dot_product(make_not_null(&tilde_e_dot_tilde_b), tilde_e, tilde_b_one_form);

    for (size_t i = 0; i < 3; ++i) {
      (*tilde_j).get(i) =
          parallel_conductivity *
          (get(tilde_e_dot_tilde_b) * tilde_b.get(i) +
           max(get(tilde_e_squared) - get(tilde_b_squared), 0.0) *
               tilde_e.get(i));
    }
  } else {
    // tilde_j should be initialized to zero before the summation performed in
    // the next `if constexpr` block
    for (size_t i = 0; i < 3; ++i) {
      (*tilde_j).get(i) = 0;
    }
  }

  if constexpr (IncludeDriftCurrent) {
    (void)parallel_conductivity;  // avoid compiler warnings

    for (LeviCivitaIterator<3> it; it; ++it) {
      const auto& i = it[0];
      const auto& j = it[1];
      const auto& k = it[2];
      (*tilde_j).get(i) +=
          it.sign() * get(tilde_q) * tilde_e_one_form.get(j) *
          tilde_b_one_form.get(k) /
          get(sqrt_det_spatial_metric);  // the extra 1/sqrt{gamma} factor comes
                                         // from the spatial Levi-Civita tensor
    }
  }

  // overall factor
  for (size_t i = 0; i < 3; ++i) {
    (*tilde_j).get(i) *= get(lapse) / get(tilde_b_squared);
  }
}

}  // namespace

void ComputeDriftTildeJ::apply(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> drift_tilde_j,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const double parallel_conductivity, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) {
  tilde_j_impl<true, false>(drift_tilde_j, tilde_q, tilde_e, tilde_b,
                            parallel_conductivity, lapse,
                            sqrt_det_spatial_metric, spatial_metric);
}

void ComputeParallelTildeJ::apply(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        parallel_tilde_j,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const double parallel_conductivity, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) {
  tilde_j_impl<false, true>(parallel_tilde_j, tilde_q, tilde_e, tilde_b,
                            parallel_conductivity, lapse,
                            sqrt_det_spatial_metric, spatial_metric);
}

namespace Tags {
void ComputeTildeJ::function(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_j,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const double parallel_conductivity, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) {
  tilde_j_impl<true, true>(tilde_j, tilde_q, tilde_e, tilde_b,
                           parallel_conductivity, lapse,
                           sqrt_det_spatial_metric, spatial_metric);
}
}  // namespace Tags

}  // namespace ForceFree
