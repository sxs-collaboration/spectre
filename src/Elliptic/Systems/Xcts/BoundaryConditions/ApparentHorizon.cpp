// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/BoundaryConditions/ApparentHorizon.hpp"

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::BoundaryConditions::detail {

template <Xcts::Geometry ConformalGeometry>
ApparentHorizonImpl<ConformalGeometry>::ApparentHorizonImpl(
    std::array<double, 3> center, std::array<double, 3> rotation,
    std::optional<gr::Solutions::KerrSchild> kerr_solution_for_lapse) noexcept
    : center_(center),
      rotation_(rotation),
      // NOLINTNEXTLINE(performance-move-const-arg)
      kerr_solution_for_lapse_(std::move(kerr_solution_for_lapse)) {}

void normal_gradient_term_flat_cartesian(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude) noexcept {
  // Write directly into the output buffer
  DataVector& projected_normal_gradient = get(*n_dot_conformal_factor_gradient);
  projected_normal_gradient = (1. - square(get<0>(face_normal))) *
                                  get<0, 0>(deriv_unnormalized_face_normal) -
                              get<0>(face_normal) * get<1>(face_normal) *
                                  get<0, 1>(deriv_unnormalized_face_normal) -
                              get<0>(face_normal) * get<2>(face_normal) *
                                  get<0, 2>(deriv_unnormalized_face_normal);
  for (size_t i = 1; i < 3; ++i) {
    projected_normal_gradient += deriv_unnormalized_face_normal.get(i, i);
    for (size_t j = 0; j < 3; ++j) {
      projected_normal_gradient -= face_normal.get(i) * face_normal.get(j) *
                                   deriv_unnormalized_face_normal.get(i, j);
    }
  }
  projected_normal_gradient /= get(face_normal_magnitude);
}

void normal_gradient_term_curved(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<DataVector*> temp_buffer,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::I<DataVector, 3>& face_normal_raised,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>&
        conformal_christoffel_second_kind) noexcept {
  // Write directly into the output buffer
  DataVector& projected_normal_gradient = get(*n_dot_conformal_factor_gradient);
  DataVector& projection = *temp_buffer;
  // Possible performance optimization: unroll the first iteration of the loop
  // to avoid zeroing the buffer. It's very verbose to do that though.
  *projected_normal_gradient = 0.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      projection = inv_conformal_metric.get(i, j) -
                   face_normal_raised.get(i) * face_normal_raised.get(j);
      projected_normal_gradient += projection *
                                   deriv_unnormalized_face_normal.get(i, j) /
                                   get(face_normal_magnitude);
      for (size_t k = 0; k < 3; ++k) {
        projected_normal_gradient -=
            projection * face_normal.get(k) *
            conformal_christoffel_second_kind.get(k, i, j);
      }
    }
  }
}

template <Xcts::Geometry ConformalGeometry>
void apparent_horizon_impl(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor,
    const gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const std::array<double, 3>& center, const std::array<double, 3>& rotation,
    const std::optional<gr::Solutions::KerrSchild>& kerr_solution_for_lapse,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x_offcenter,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    [[maybe_unused]] const std::optional<
        std::reference_wrapper<const tnsr::II<DataVector, 3>>>
        inv_conformal_metric,
    [[maybe_unused]] const std::optional<
        std::reference_wrapper<const tnsr::Ijj<DataVector, 3>>>
        conformal_christoffel_second_kind) noexcept {
  // Allocate some temporary memory
  TempBuffer<tmpl::list<::Tags::TempI<0, 3>, ::Tags::TempScalar<1>,
                        ::Tags::TempI<2, 3>>>
      buffer{face_normal.begin()->size()};
  // Center the coordinates
  tnsr::I<DataVector, 3>& x = get<::Tags::TempI<2, 3>>(buffer);
  x = x_offcenter;
  get<0>(x) -= center[0];
  get<1>(x) -= center[1];
  get<2>(x) -= center[2];
  // Note that the face normal points _out_ of the computational domain, i.e.
  // _into_ the excised region. It is opposite the conformal unit normal to the
  // horizon surface: \bar{s}_i = -n_i.
  tnsr::I<DataVector, 3>& face_normal_raised = get<::Tags::TempI<0, 3>>(buffer);
  if constexpr (ConformalGeometry == Xcts::Geometry::FlatCartesian) {
    get<0>(face_normal_raised) = get<0>(face_normal);
    get<1>(face_normal_raised) = get<1>(face_normal);
    get<2>(face_normal_raised) = get<2>(face_normal);
  } else {
    raise_or_lower_index(make_not_null(&face_normal_raised), face_normal,
                         inv_conformal_metric->get());
  }

  // Shift
  {
    DataVector& beta_orthogonal = get(get<::Tags::TempScalar<1>>(buffer));
    beta_orthogonal =
        get(*lapse_times_conformal_factor) / cube(get(*conformal_factor));
    for (size_t i = 0; i < 3; ++i) {
      shift_excess->get(i) = -beta_orthogonal * face_normal_raised.get(i) -
                             shift_background.get(i);
    }
  }
  for (LeviCivitaIterator<3> it; it; ++it) {
    shift_excess->get(it[0]) +=
        it.sign() * gsl::at(rotation, it[1]) * x.get(it[2]);
  }

  // Conformal factor
  if constexpr (ConformalGeometry == Xcts::Geometry::FlatCartesian) {
    normal_gradient_term_flat_cartesian(
        n_dot_conformal_factor_gradient, face_normal,
        deriv_unnormalized_face_normal, face_normal_magnitude);
  } else {
    normal_gradient_term_curved(
        n_dot_conformal_factor_gradient,
        make_not_null(&get(get<::Tags::TempScalar<1>>(buffer))), face_normal,
        face_normal_raised, deriv_unnormalized_face_normal,
        face_normal_magnitude, *inv_conformal_metric,
        *conformal_christoffel_second_kind);
  }
  get(*n_dot_conformal_factor_gradient) *= -0.25 * get(*conformal_factor);
  // At this point we're done with the raised face normal, so we re-purpose the
  // memory buffer
  {
    tnsr::I<DataVector, 3>& n_dot_longitudinal_shift =
        get<::Tags::TempI<0, 3>>(buffer);
    normal_dot_flux(make_not_null(&n_dot_longitudinal_shift), face_normal,
                    longitudinal_shift_background);
    for (size_t i = 0; i < 3; ++i) {
      n_dot_longitudinal_shift.get(i) +=
          n_dot_longitudinal_shift_excess->get(i);
    }
    Scalar<DataVector>& nn_dot_longitudinal_shift =
        get<::Tags::TempScalar<1>>(buffer);
    normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift), face_normal,
                    n_dot_longitudinal_shift);
    get(*n_dot_conformal_factor_gradient) +=
        -get(extrinsic_curvature_trace) * cube(get(*conformal_factor)) / 6. +
        pow<4>(get(*conformal_factor)) / 8. /
            get(*lapse_times_conformal_factor) * get(nn_dot_longitudinal_shift);
  }

  // Lapse
  if (kerr_solution_for_lapse.has_value()) {
    *lapse_times_conformal_factor =
        get<gr::Tags::Lapse<DataVector>>(kerr_solution_for_lapse->variables(
            x, 0., tmpl::list<gr::Tags::Lapse<DataVector>>{}));
  } else {
    get(*n_dot_lapse_times_conformal_factor_gradient) = 0.;
  }
}

template <Xcts::Geometry ConformalGeometry>
void linearized_apparent_horizon_impl(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_correction,
    const std::optional<gr::Solutions::KerrSchild>& kerr_solution_for_lapse,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
    [[maybe_unused]] const std::optional<
        std::reference_wrapper<const tnsr::II<DataVector, 3>>>
        inv_conformal_metric,
    [[maybe_unused]] const std::optional<
        std::reference_wrapper<const tnsr::Ijj<DataVector, 3>>>
        conformal_christoffel_second_kind) noexcept {
  // Allocate some temporary memory
  TempBuffer<tmpl::list<::Tags::TempI<0, 3>, ::Tags::TempScalar<1>>> buffer{
      face_normal.begin()->size()};
  tnsr::I<DataVector, 3>& face_normal_raised = get<::Tags::TempI<0, 3>>(buffer);
  if constexpr (ConformalGeometry == Xcts::Geometry::FlatCartesian) {
    get<0>(face_normal_raised) = get<0>(face_normal);
    get<1>(face_normal_raised) = get<1>(face_normal);
    get<2>(face_normal_raised) = get<2>(face_normal);
  } else {
    raise_or_lower_index(make_not_null(&face_normal_raised), face_normal,
                         inv_conformal_metric->get());
  }

  // Shift
  {
    DataVector& beta_orthogonal_correction =
        get(get<::Tags::TempScalar<1>>(buffer));
    beta_orthogonal_correction = get(*lapse_times_conformal_factor_correction) /
                                     cube(get(conformal_factor)) -
                                 3. * get(lapse_times_conformal_factor) /
                                     pow<4>(get(conformal_factor)) *
                                     get(*conformal_factor_correction);
    for (size_t i = 0; i < 3; ++i) {
      shift_correction->get(i) =
          -beta_orthogonal_correction * face_normal_raised.get(i);
    }
  }

  // Conformal factor
  if constexpr (ConformalGeometry == Xcts::Geometry::FlatCartesian) {
    normal_gradient_term_flat_cartesian(
        n_dot_conformal_factor_gradient_correction, face_normal,
        deriv_unnormalized_face_normal, face_normal_magnitude);
  } else {
    normal_gradient_term_curved(
        n_dot_conformal_factor_gradient_correction,
        make_not_null(&get(get<::Tags::TempScalar<1>>(buffer))), face_normal,
        face_normal_raised, deriv_unnormalized_face_normal,
        face_normal_magnitude, *inv_conformal_metric,
        *conformal_christoffel_second_kind);
  }
  get(*n_dot_conformal_factor_gradient_correction) *=
      -0.25 * get(*conformal_factor_correction);
  // At this point we're done with the raised face normal, so we're re-using the
  // memory buffer
  {
    tnsr::I<DataVector, 3>& n_dot_longitudinal_shift =
        get<::Tags::TempI<0, 3>>(buffer);
    normal_dot_flux(make_not_null(&n_dot_longitudinal_shift), face_normal,
                    longitudinal_shift_background);
    for (size_t i = 0; i < 3; ++i) {
      n_dot_longitudinal_shift.get(i) += n_dot_longitudinal_shift_excess.get(i);
    }
    Scalar<DataVector>& nn_dot_longitudinal_shift =
        get<::Tags::TempScalar<1>>(buffer);
    normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift), face_normal,
                    n_dot_longitudinal_shift);
    get(*n_dot_conformal_factor_gradient_correction) +=
        -0.5 * get(extrinsic_curvature_trace) * square(get(conformal_factor)) *
            get(*conformal_factor_correction) +
        0.5 * pow<3>(get(conformal_factor)) /
            get(lapse_times_conformal_factor) * get(nn_dot_longitudinal_shift) *
            get(*conformal_factor_correction) -
        0.125 * pow<4>(get(conformal_factor)) /
            square(get(lapse_times_conformal_factor)) *
            get(nn_dot_longitudinal_shift) *
            get(*lapse_times_conformal_factor_correction);
  }
  {
    Scalar<DataVector>& nn_dot_longitudinal_shift_correction =
        get<::Tags::TempScalar<1>>(buffer);
    normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift_correction),
                    face_normal, *n_dot_longitudinal_shift_correction);
    get(*n_dot_conformal_factor_gradient_correction) +=
        0.125 * pow<4>(get(conformal_factor)) /
        get(lapse_times_conformal_factor) *
        get(nn_dot_longitudinal_shift_correction);
  }

  // Lapse
  if (kerr_solution_for_lapse.has_value()) {
    get(*lapse_times_conformal_factor_correction) = 0.;
  } else {
    get(*n_dot_lapse_times_conformal_factor_gradient_correction) = 0.;
  }
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizonImpl<ConformalGeometry>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor,
    const gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background)
    const noexcept {
  apparent_horizon_impl<ConformalGeometry>(
      conformal_factor, lapse_times_conformal_factor, shift_excess,
      n_dot_conformal_factor_gradient,
      n_dot_lapse_times_conformal_factor_gradient,
      n_dot_longitudinal_shift_excess, center_, rotation_,
      kerr_solution_for_lapse_, face_normal, deriv_unnormalized_face_normal,
      face_normal_magnitude, x, extrinsic_curvature_trace, shift_background,
      longitudinal_shift_background, std::nullopt, std::nullopt);
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizonImpl<ConformalGeometry>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor,
    const gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind)
    const noexcept {
  apparent_horizon_impl<ConformalGeometry>(
      conformal_factor, lapse_times_conformal_factor, shift_excess,
      n_dot_conformal_factor_gradient,
      n_dot_lapse_times_conformal_factor_gradient,
      n_dot_longitudinal_shift_excess, center_, rotation_,
      kerr_solution_for_lapse_, face_normal, deriv_unnormalized_face_normal,
      face_normal_magnitude, x, extrinsic_curvature_trace, shift_background,
      longitudinal_shift_background, inv_conformal_metric,
      conformal_christoffel_second_kind);
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizonImpl<ConformalGeometry>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess_correction,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess)
    const noexcept {
  linearized_apparent_horizon_impl<ConformalGeometry>(
      conformal_factor_correction, lapse_times_conformal_factor_correction,
      shift_excess_correction, n_dot_conformal_factor_gradient_correction,
      n_dot_lapse_times_conformal_factor_gradient_correction,
      n_dot_longitudinal_shift_excess_correction, kerr_solution_for_lapse_,
      face_normal, deriv_unnormalized_face_normal, face_normal_magnitude,
      extrinsic_curvature_trace, longitudinal_shift_background,
      conformal_factor, lapse_times_conformal_factor,
      n_dot_longitudinal_shift_excess, std::nullopt, std::nullopt);
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizonImpl<ConformalGeometry>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess_correction,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind)
    const noexcept {
  linearized_apparent_horizon_impl<ConformalGeometry>(
      conformal_factor_correction, lapse_times_conformal_factor_correction,
      shift_excess_correction, n_dot_conformal_factor_gradient_correction,
      n_dot_lapse_times_conformal_factor_gradient_correction,
      n_dot_longitudinal_shift_excess_correction, kerr_solution_for_lapse_,
      face_normal, deriv_unnormalized_face_normal, face_normal_magnitude,
      extrinsic_curvature_trace, longitudinal_shift_background,
      conformal_factor, lapse_times_conformal_factor,
      n_dot_longitudinal_shift_excess, inv_conformal_metric,
      conformal_christoffel_second_kind);
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizonImpl<ConformalGeometry>::pup(PUP::er& p) noexcept {
  p | center_;
  p | rotation_;
  p | kerr_solution_for_lapse_;
}

template <Xcts::Geometry ConformalGeometry>
bool operator==(const ApparentHorizonImpl<ConformalGeometry>& lhs,
                const ApparentHorizonImpl<ConformalGeometry>& rhs) noexcept {
  return lhs.center() == rhs.center() and lhs.rotation() == rhs.rotation() and
         lhs.kerr_solution_for_lapse() == rhs.kerr_solution_for_lapse();
}

template <Xcts::Geometry ConformalGeometry>
bool operator!=(const ApparentHorizonImpl<ConformalGeometry>& lhs,
                const ApparentHorizonImpl<ConformalGeometry>& rhs) noexcept {
  return not(lhs == rhs);
}

template class ApparentHorizonImpl<Xcts::Geometry::FlatCartesian>;
template class ApparentHorizonImpl<Xcts::Geometry::Curved>;
template bool operator==(
    const ApparentHorizonImpl<Xcts::Geometry::FlatCartesian>& lhs,
    const ApparentHorizonImpl<Xcts::Geometry::FlatCartesian>& rhs) noexcept;
template bool operator!=(
    const ApparentHorizonImpl<Xcts::Geometry::FlatCartesian>& lhs,
    const ApparentHorizonImpl<Xcts::Geometry::FlatCartesian>& rhs) noexcept;
template bool operator==(
    const ApparentHorizonImpl<Xcts::Geometry::Curved>& lhs,
    const ApparentHorizonImpl<Xcts::Geometry::Curved>& rhs) noexcept;
template bool operator!=(
    const ApparentHorizonImpl<Xcts::Geometry::Curved>& lhs,
    const ApparentHorizonImpl<Xcts::Geometry::Curved>& rhs) noexcept;

}  // namespace Xcts::BoundaryConditions::detail
