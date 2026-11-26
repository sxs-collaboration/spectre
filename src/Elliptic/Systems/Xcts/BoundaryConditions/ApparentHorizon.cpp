// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/BoundaryConditions/ApparentHorizon.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <ostream>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Factory.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace Xcts::BoundaryConditions {

std::ostream& operator<<(
    std::ostream& os, const NegativeExpansionShift negative_expansion_shift) {
  switch (negative_expansion_shift) {
    case NegativeExpansionShift::None:
      return os << "None";
    case NegativeExpansionShift::Orthogonal:
      return os << "Orthogonal";
    case NegativeExpansionShift::AnalyticGuess:
      return os << "AnalyticGuess";
    default:
      ERROR("Missing case for operator<<(NegativeExpansionShift).");
  }
}

template <Xcts::Geometry ConformalGeometry>
ApparentHorizon<ConformalGeometry>::ApparentHorizon(
    std::array<double, 3> center, std::array<double, 3> rotation,
    std::optional<std::unique_ptr<elliptic::analytic_data::InitialGuess>>
        solution_for_lapse,
    std::optional<NegativeExpansionOptions> negative_expansion,
    const Options::Context& /*context*/)
    : center_(center),
      rotation_(rotation),
      // NOLINTNEXTLINE(performance-move-const-arg)
      solution_for_lapse_(std::move(solution_for_lapse)) {
  if (negative_expansion.has_value()) {
    solution_for_negative_expansion_ =
        std::move(negative_expansion->solution);
    negative_expansion_shift_ = negative_expansion->shift;
  }
}

namespace {
// This sets the output buffer to: \bar{m}^{ij} \bar{\nabla}_i n_j
void normal_gradient_term_flat_cartesian(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude) {
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

// This sets the output buffer to: \bar{m}^{ij} \bar{\nabla}_i n_j
void normal_gradient_term_curved(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<DataVector*> temp_buffer,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::I<DataVector, 3>& face_normal_raised,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) {
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

// Compute the expansion Theta = m^ij (D_i s_j - K_ij) and the shift-correction
// epsilon = shift_orthogonal - lapse of an analytic solution. Also returns the
// shift of the solution. With `NegativeExpansionShift::Orthogonal` or
// `AnalyticGuess`, the shift of the solution (given in the inertial frame) is
// transformed to the comoving frame defined by the background shift, i.e. the
// background shift is added to it.
//
// With `AnalyticGuess`, only the part -lapse of epsilon is returned.
// The orthogonal part s_i shift^i is then imposed with the normal of the
// current spatial metric, together with the tangential part, by imposing the
// full shift of the solution (see `apparent_horizon_impl`).
void negative_expansion_quantities(
    const gsl::not_null<Scalar<DataVector>*> expansion,
    const gsl::not_null<Scalar<DataVector>*> beta_orthogonal_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> solution_shift,
    const std::unique_ptr<elliptic::analytic_data::InitialGuess>& solution,
    const NegativeExpansionShift negative_expansion_shift,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::I<DataVector, 3>& x,
    const tnsr::i<DataVector, 3>& conformal_face_normal,
    const Scalar<DataVector>& unnormalized_conformal_face_normal_magnitude,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal) {
  using analytic_tags =
      tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>,
                               tmpl::size_t<3>, Frame::Inertial>,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3>>;
  const auto solution_vars =
      call_with_dynamic_type<tuples::tagged_tuple_from_typelist<analytic_tags>,
                             Xcts::Solutions::all_analytic_solutions>(
          solution.get(), [&x](const auto* const local_solution) {
            return local_solution->variables(x, analytic_tags{});
          });
  const auto& inv_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(solution_vars);
  const auto& deriv_spatial_metric =
      get<::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(solution_vars);
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(solution_vars);
  const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(solution_vars);
  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(solution_vars);
  // The passed-in face normal is normalized with the full conformal metric
  // (typically a superposition of isolated metrics). Therefore, we need to
  // normalized the face normal again with the isolated Kerr metric. Possible
  // optimization if this turns out to have a significant computational cost:
  // We could just re-use the face normal and the projected normal gradient
  // from the full conformal metric, since it's probably sufficiently close to
  // the isolated Kerr metric. However, the difference between the two metric
  // means that the expansion is not exactly zero when the excision surface
  // coincides with the isolated Kerr horizon, which users would probably
  // expect.
  TempBuffer<tmpl::list<::Tags::Tempi<0, 3>, ::Tags::TempScalar<1>,
                        ::Tags::TempI<2, 3>, ::Tags::TempIjj<3, 3>>>
      buffer{x.begin()->size()};
  auto& face_normal = get<::Tags::Tempi<0, 3>>(buffer);
  auto& face_normal_magnitude = get<::Tags::TempScalar<1>>(buffer);
  auto& face_normal_raised = get<::Tags::TempI<2, 3>>(buffer);
  auto& spatial_christoffel_second_kind = get<::Tags::TempIjj<3, 3>>(buffer);
  magnitude(make_not_null(&face_normal_magnitude), conformal_face_normal,
            inv_spatial_metric);
  face_normal = conformal_face_normal;
  for (size_t d = 0; d < 3; ++d) {
    face_normal.get(d) /= get(face_normal_magnitude);
  }
  get(face_normal_magnitude) *=
      get(unnormalized_conformal_face_normal_magnitude);
  raise_or_lower_index(make_not_null(&face_normal_raised), face_normal,
                       inv_spatial_metric);
  gr::christoffel_second_kind(make_not_null(&spatial_christoffel_second_kind),
                              deriv_spatial_metric, inv_spatial_metric);
  // Compute Theta = m^ij (D_i s_j - K_ij)
  // Use second output buffer as temporary memory
  DataVector& projection = get(*beta_orthogonal_correction);
  get(*expansion) = 0.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      projection = inv_spatial_metric.get(i, j) -
                   face_normal_raised.get(i) * face_normal_raised.get(j);
      // This first part is the same as `normal_gradient_term_curved` above
      get(*expansion) += projection * deriv_unnormalized_face_normal.get(i, j) /
                         get(face_normal_magnitude);
      for (size_t k = 0; k < 3; ++k) {
        get(*expansion) -= projection * face_normal.get(k) *
                           spatial_christoffel_second_kind.get(k, i, j);
      }
      // Additional extrinsic-curvature term
      get(*expansion) += projection * extrinsic_curvature.get(i, j);
    }
  }
  get(*expansion) *= -1.;
  // Compute epsilon = s_i (shift^i + shift_background^i) - lapse in the
  // comoving frame, and epsilon = s_i shift^i - lapse otherwise. The background
  // shift transforms the shift of the solution from the inertial frame to the
  // comoving frame in which the XCTS problem is posed (e.g. a linear velocity,
  // or orbital rotation and expansion). Note that the face normal points into
  // the excised region, n_i = -s_i.
  for (size_t i = 0; i < 3; ++i) {
    solution_shift->get(i) = shift.get(i);
    if (negative_expansion_shift != NegativeExpansionShift::None) {
      solution_shift->get(i) += shift_background.get(i);
    }
  }
  if (negative_expansion_shift == NegativeExpansionShift::AnalyticGuess) {
    get(*beta_orthogonal_correction) = -get(lapse);
    return;
  }
  normal_dot_flux(beta_orthogonal_correction, face_normal, *solution_shift);
  get(*beta_orthogonal_correction) *= -1.;
  get(*beta_orthogonal_correction) -= get(lapse);
}

template <Xcts::Geometry ConformalGeometry>
void apparent_horizon_impl(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const std::array<double, 3>& center, const std::array<double, 3>& rotation,
    const std::optional<std::unique_ptr<elliptic::analytic_data::InitialGuess>>&
        solution_for_lapse,
    const std::optional<std::unique_ptr<elliptic::analytic_data::InitialGuess>>&
        solution_for_negative_expansion,
    const NegativeExpansionShift negative_expansion_shift,
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
        conformal_christoffel_second_kind) {
  // Allocate some temporary memory
  TempBuffer<tmpl::list<::Tags::TempI<0, 3>, ::Tags::TempScalar<1>,
                        ::Tags::TempI<2, 3>, ::Tags::TempScalar<3>,
                        ::Tags::TempI<4, 3>>>
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

  // Compute quantities for negative-expansion boundary conditions. We collect
  // all calls into the analytic solution in one place so it doesn't have to
  // compute intermediate quantities multiple times. Possible optimization:
  // Store the result in the DataBox and use it in repeated applications of the
  // boundary conditions.
  Scalar<DataVector>& expansion_of_solution =
      get<::Tags::TempScalar<3>>(buffer);
  Scalar<DataVector>& beta_orthogonal = get<::Tags::TempScalar<1>>(buffer);
  tnsr::I<DataVector, 3>& solution_shift = get<::Tags::TempI<4, 3>>(buffer);
  if (solution_for_negative_expansion.has_value()) {
    negative_expansion_quantities(
        make_not_null(&expansion_of_solution), make_not_null(&beta_orthogonal),
        make_not_null(&solution_shift), *solution_for_negative_expansion,
        negative_expansion_shift, shift_background, x, face_normal,
        face_normal_magnitude, deriv_unnormalized_face_normal);
    ASSERT(max(get(expansion_of_solution)) <
               100.0 * std::numeric_limits<double>::epsilon(),
           "Expansion of solution is not negative everywhere.");
  }

  // Shift
  {
    if (solution_for_negative_expansion.has_value()) {
      get(beta_orthogonal) /= square(get(*conformal_factor_minus_one) + 1.);
    } else {
      get(beta_orthogonal) = 0.;
    }
    get(beta_orthogonal) +=
        (get(*lapse_times_conformal_factor_minus_one) + 1.) /
        cube(get(*conformal_factor_minus_one) + 1.);
    for (size_t i = 0; i < 3; ++i) {
      shift_excess->get(i) = -get(beta_orthogonal) * face_normal_raised.get(i) -
                             shift_background.get(i);
    }
  }
  for (LeviCivitaIterator<3> it; it; ++it) {
    shift_excess->get(it[0]) +=
        it.sign() * gsl::at(rotation, it[1]) * x.get(it[2]);
  }
  // Full comoving shift of the negative-expansion solution. Its orthogonal part
  // s_i beta_co^i (with the unit normal s_i = psi^2 \bar{s}_i of the current
  // spatial metric) completes epsilon = s_i beta_co^i - lapse, of which
  // `negative_expansion_quantities` returned only the part -lapse. Its
  // tangential part tau^i = beta_co^i - \bar{s}^i \bar{s}_j beta_co^j is
  // projected with the same normal, so both parts recombine to beta_co^i.
  if (solution_for_negative_expansion.has_value() and
      negative_expansion_shift == NegativeExpansionShift::AnalyticGuess) {
    for (size_t i = 0; i < 3; ++i) {
      shift_excess->get(i) += solution_shift.get(i);
    }
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
  // At this point we're done with `face_normal_raised`, so we can re-purpose
  // the memory buffer.
  get(*n_dot_conformal_factor_gradient) *=
      -0.25 * (get(*conformal_factor_minus_one) + 1.);
  if (solution_for_negative_expansion.has_value()) {
    get(*n_dot_conformal_factor_gradient) -=
        0.25 * cube(get(*conformal_factor_minus_one) + 1.) *
        get(expansion_of_solution);
  }
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
        -get(extrinsic_curvature_trace) *
            cube(get(*conformal_factor_minus_one) + 1.) / 6. +
        pow<4>(get(*conformal_factor_minus_one) + 1.) / 8. /
            (get(*lapse_times_conformal_factor_minus_one) + 1.) *
            get(nn_dot_longitudinal_shift);
  }

  // Lapse
  if (solution_for_lapse.has_value()) {
    using Binary =
        Xcts::AnalyticData::Binary<elliptic::analytic_data::AnalyticSolution,
                                   Xcts::Solutions::all_analytic_solutions>;
    // Evaluate isolated solutions at coordinates centered on the excision, but
    // evaluate the superposed binary data at the original coordinates
    const bool evaluate_centered =
        dynamic_cast<const Binary*>(solution_for_lapse.value().get()) ==
        nullptr;
    *lapse_times_conformal_factor_minus_one = call_with_dynamic_type<
        Scalar<DataVector>,
        tmpl::push_back<Xcts::Solutions::all_analytic_solutions, Binary>>(
        solution_for_lapse.value().get(),
        [&evaluate_centered, &x,
         &x_offcenter](const auto* const local_solution) {
          return get<Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>>(
              local_solution->variables(
                  evaluate_centered ? x : x_offcenter,
                  tmpl::list<Xcts::Tags::LapseTimesConformalFactorMinusOne<
                      DataVector>>{}));
        });
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
    const std::array<double, 3>& center,
    const std::optional<std::unique_ptr<elliptic::analytic_data::InitialGuess>>&
        solution_for_lapse,
    const std::optional<std::unique_ptr<elliptic::analytic_data::InitialGuess>>&
        solution_for_negative_expansion,
    const NegativeExpansionShift negative_expansion_shift,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x_offcenter,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
    [[maybe_unused]] const std::optional<
        std::reference_wrapper<const tnsr::II<DataVector, 3>>>
        inv_conformal_metric,
    [[maybe_unused]] const std::optional<
        std::reference_wrapper<const tnsr::Ijj<DataVector, 3>>>
        conformal_christoffel_second_kind) {
  // Allocate some temporary memory
  TempBuffer<tmpl::list<::Tags::TempI<0, 3>, ::Tags::TempScalar<1>,
                        ::Tags::TempScalar<2>, ::Tags::TempI<3, 3>>>
      buffer{face_normal.begin()->size()};

  // Negative-expansion quantities
  Scalar<DataVector>& expansion_of_solution =
      get<::Tags::TempScalar<2>>(buffer);
  Scalar<DataVector>& beta_orthogonal_correction =
      get<::Tags::TempScalar<1>>(buffer);
  if (solution_for_negative_expansion.has_value()) {
    // Center the coordinates
    tnsr::I<DataVector, 3>& x = get<::Tags::TempI<0, 3>>(buffer);
    x = x_offcenter;
    get<0>(x) -= center[0];
    get<1>(x) -= center[1];
    get<2>(x) -= center[2];
    // The shift of the solution (and hence the term added with
    // `AnalyticGuess`) doesn't depend on the fields, so it doesn't
    // contribute to the linearization. The part of epsilon returned by
    // `negative_expansion_quantities` is divided by psi^2 in the nonlinear
    // condition, which gives the term -2 epsilon delta psi / psi^3 below.
    tnsr::I<DataVector, 3>& unused_solution_shift =
        get<::Tags::TempI<3, 3>>(buffer);
    negative_expansion_quantities(
        make_not_null(&expansion_of_solution),
        make_not_null(&beta_orthogonal_correction),
        make_not_null(&unused_solution_shift), *solution_for_negative_expansion,
        negative_expansion_shift, shift_background, x, face_normal,
        face_normal_magnitude, deriv_unnormalized_face_normal);
  }

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
    if (solution_for_negative_expansion.has_value()) {
      get(beta_orthogonal_correction) *=
          -2. * get(*conformal_factor_correction) /
          cube(get(conformal_factor_minus_one) + 1.);
    } else {
      get(beta_orthogonal_correction) = 0.;
    }
    get(beta_orthogonal_correction) +=
        get(*lapse_times_conformal_factor_correction) /
            cube(get(conformal_factor_minus_one) + 1.) -
        3. * (get(lapse_times_conformal_factor_minus_one) + 1.) /
            pow<4>(get(conformal_factor_minus_one) + 1.) *
            get(*conformal_factor_correction);
    for (size_t i = 0; i < 3; ++i) {
      shift_correction->get(i) =
          -get(beta_orthogonal_correction) * face_normal_raised.get(i);
    }
  }
  // At this point we're done with `beta_orthogonal_correction`, so we can
  // re-purpose the memory buffer.

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
  // At this point we're done with `face_normal_raised`, so we can re-purpose
  // the memory buffer.
  get(*n_dot_conformal_factor_gradient_correction) *=
      -0.25 * get(*conformal_factor_correction);
  if (solution_for_negative_expansion.has_value()) {
    get(*n_dot_conformal_factor_gradient_correction) -=
        0.75 * square(get(conformal_factor_minus_one) + 1.) *
        get(expansion_of_solution) * get(*conformal_factor_correction);
  }
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
        -0.5 * get(extrinsic_curvature_trace) *
            square(get(conformal_factor_minus_one) + 1.) *
            get(*conformal_factor_correction) +
        0.5 * pow<3>(get(conformal_factor_minus_one) + 1.) /
            (get(lapse_times_conformal_factor_minus_one) + 1.) *
            get(nn_dot_longitudinal_shift) * get(*conformal_factor_correction) -
        0.125 * pow<4>(get(conformal_factor_minus_one) + 1.) /
            square(get(lapse_times_conformal_factor_minus_one) + 1.) *
            get(nn_dot_longitudinal_shift) *
            get(*lapse_times_conformal_factor_correction);
  }
  {
    Scalar<DataVector>& nn_dot_longitudinal_shift_correction =
        get<::Tags::TempScalar<1>>(buffer);
    normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift_correction),
                    face_normal, *n_dot_longitudinal_shift_correction);
    get(*n_dot_conformal_factor_gradient_correction) +=
        0.125 * pow<4>(get(conformal_factor_minus_one) + 1.) /
        (get(lapse_times_conformal_factor_minus_one) + 1.) *
        get(nn_dot_longitudinal_shift_correction);
  }

  // Lapse
  if (solution_for_lapse.has_value()) {
    get(*lapse_times_conformal_factor_correction) = 0.;
  } else {
    get(*n_dot_lapse_times_conformal_factor_gradient_correction) = 0.;
  }
}
}  // namespace

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizon<ConformalGeometry>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse_times_conformal_factor*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess*/,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background) const {
  apparent_horizon_impl<ConformalGeometry>(
      conformal_factor_minus_one, lapse_times_conformal_factor_minus_one,
      shift_excess, n_dot_conformal_factor_gradient,
      n_dot_lapse_times_conformal_factor_gradient,
      n_dot_longitudinal_shift_excess, center_, rotation_, solution_for_lapse_,
      solution_for_negative_expansion_, negative_expansion_shift_, face_normal,
      deriv_unnormalized_face_normal, face_normal_magnitude, x,
      extrinsic_curvature_trace, shift_background,
      longitudinal_shift_background, std::nullopt, std::nullopt);
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizon<ConformalGeometry>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse_times_conformal_factor*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess*/,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) const {
  apparent_horizon_impl<ConformalGeometry>(
      conformal_factor_minus_one, lapse_times_conformal_factor_minus_one,
      shift_excess, n_dot_conformal_factor_gradient,
      n_dot_lapse_times_conformal_factor_gradient,
      n_dot_longitudinal_shift_excess, center_, rotation_, solution_for_lapse_,
      solution_for_negative_expansion_, negative_expansion_shift_, face_normal,
      deriv_unnormalized_face_normal, face_normal_magnitude, x,
      extrinsic_curvature_trace, shift_background,
      longitudinal_shift_background, inv_conformal_metric,
      conformal_christoffel_second_kind);
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizon<ConformalGeometry>::apply_linearized(
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
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/,
    const tnsr::i<DataVector,
                  3>& /*deriv_lapse_times_conformal_factor_correction*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess_correction*/,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess) const {
  linearized_apparent_horizon_impl<ConformalGeometry>(
      conformal_factor_correction, lapse_times_conformal_factor_correction,
      shift_excess_correction, n_dot_conformal_factor_gradient_correction,
      n_dot_lapse_times_conformal_factor_gradient_correction,
      n_dot_longitudinal_shift_excess_correction, center_, solution_for_lapse_,
      solution_for_negative_expansion_, negative_expansion_shift_, face_normal,
      deriv_unnormalized_face_normal, face_normal_magnitude, x,
      extrinsic_curvature_trace, shift_background,
      longitudinal_shift_background, conformal_factor_minus_one,
      lapse_times_conformal_factor_minus_one, n_dot_longitudinal_shift_excess,
      std::nullopt, std::nullopt);
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizon<ConformalGeometry>::apply_linearized(
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
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/,
    const tnsr::i<DataVector,
                  3>& /*deriv_lapse_times_conformal_factor_correction*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess_correction*/,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) const {
  linearized_apparent_horizon_impl<ConformalGeometry>(
      conformal_factor_correction, lapse_times_conformal_factor_correction,
      shift_excess_correction, n_dot_conformal_factor_gradient_correction,
      n_dot_lapse_times_conformal_factor_gradient_correction,
      n_dot_longitudinal_shift_excess_correction, center_, solution_for_lapse_,
      solution_for_negative_expansion_, negative_expansion_shift_, face_normal,
      deriv_unnormalized_face_normal, face_normal_magnitude, x,
      extrinsic_curvature_trace, shift_background,
      longitudinal_shift_background, conformal_factor_minus_one,
      lapse_times_conformal_factor_minus_one, n_dot_longitudinal_shift_excess,
      inv_conformal_metric, conformal_christoffel_second_kind);
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizon<ConformalGeometry>::pup(PUP::er& p) {
  p | center_;
  p | rotation_;
  p | solution_for_lapse_;
  p | solution_for_negative_expansion_;
  p | negative_expansion_shift_;
}

template <Xcts::Geometry ConformalGeometry>
PUP::able::PUP_ID ApparentHorizon<ConformalGeometry>::my_PUP_ID = 0;  // NOLINT

template class ApparentHorizon<Xcts::Geometry::FlatCartesian>;
template class ApparentHorizon<Xcts::Geometry::Curved>;

}  // namespace Xcts::BoundaryConditions

template <>
Xcts::BoundaryConditions::NegativeExpansionShift
Options::create_from_yaml<Xcts::BoundaryConditions::NegativeExpansionShift>::
    create<void>(const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if ("None" == type_read) {
    return Xcts::BoundaryConditions::NegativeExpansionShift::None;
  } else if ("Orthogonal" == type_read) {
    return Xcts::BoundaryConditions::NegativeExpansionShift::Orthogonal;
  } else if ("AnalyticGuess" == type_read) {
    return Xcts::BoundaryConditions::NegativeExpansionShift::AnalyticGuess;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to Xcts::BoundaryConditions::NegativeExpansionShift. "
                     "Must be one of 'None', 'Orthogonal' or 'AnalyticGuess'.");
}
