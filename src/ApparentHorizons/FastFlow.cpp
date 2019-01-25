// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/FastFlow.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <pup.h>
#include <string>
#include <utility>

#include "ApparentHorizons/SpherepackIterator.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/StrahlkorperGr.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Error.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
// IWYU pragma: no_forward_declare Tensor

// IWYU pragma: no_include <complex>

FastFlow::FastFlow(FastFlow::Flow::type flow, FastFlow::Alpha::type alpha,
                   FastFlow::Beta::type beta, FastFlow::AbsTol::type abs_tol,
                   FastFlow::TruncationTol::type trunc_tol,
                   FastFlow::DivergenceTol::type divergence_tol,
                   FastFlow::DivergenceIter::type divergence_iter,
                   FastFlow::MaxIts::type max_its) noexcept
    : alpha_(alpha),
      beta_(beta),
      abs_tol_(abs_tol),
      trunc_tol_(trunc_tol),
      divergence_tol_(divergence_tol),
      divergence_iter_(divergence_iter),
      max_its_(max_its),
      flow_(flow),
      current_iter_(0),
      previous_residual_mesh_norm_(0.0),
      min_residual_mesh_norm_(std::numeric_limits<double>::max()),
      iter_at_min_residual_mesh_norm_(0) {}

template <typename Frame>
size_t FastFlow::current_l_mesh(const Strahlkorper<Frame>& strahlkorper) const
    noexcept {
  const size_t l_max = strahlkorper.ylm_spherepack().l_max();
  // This is the formula used in SpEC (if l_max>=4). We may want to make this
  // formula an option in the future, if we want to experiment with it.
  return static_cast<size_t>(std::floor(1.5 * l_max));
}

namespace {
template <typename Frame>
DataVector fast_flow_weight(
    const Scalar<DataVector>& one_form_magnitude,
    const db::item_type<StrahlkorperTags::Rhat<Frame>>& r_hat,
    const db::item_type<StrahlkorperTags::Radius<Frame>>& radius,
    const tnsr::II<DataVector, 3, Frame>& inverse_surface_metric) noexcept {
  // Form Euclidean surface metric
  auto flat_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame>>(radius, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    flat_metric.get(i, i) = 1.0;
    for (size_t j = i; j < 3; ++j) {
      flat_metric.get(i, j) -= r_hat.get(i) * r_hat.get(j);
    }
  }

  auto denominator = make_with_value<DataVector>(radius, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      denominator += inverse_surface_metric.get(i, j) * flat_metric.get(i, j);
    }
  }

  return 2.0 * get(one_form_magnitude) * square(radius) / denominator;
}
}  // namespace

template <typename Frame>
std::pair<FastFlow::Status, FastFlow::IterInfo>
FastFlow::iterate_horizon_finder(
    const gsl::not_null<Strahlkorper<Frame>*> current_strahlkorper,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::Ijj<DataVector, 3, Frame>& christoffel_2nd_kind) noexcept {
  const size_t l_surface = current_strahlkorper->l_max();
  const size_t l_mesh = current_l_mesh(*current_strahlkorper);

  // Evaluate the Strahlkorper on a higher resolution mesh
  const Strahlkorper<Frame> strahlkorper(l_mesh, l_mesh, *current_strahlkorper);

  // Make a DataBox with this strahlkorper.
  // Do we want to define ComputeItems for expansion, normalized
  // unit norms, etc in this DataBox? So far we do not.
  auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame>>,
      db::AddComputeTags<StrahlkorperTags::compute_items_tags<Frame>>>(
      strahlkorper);

  // Get minimum radius.
  const auto& radius = db::get<StrahlkorperTags::Radius<Frame>>(box);
  const auto r_minmax = std::minmax_element(radius.begin(), radius.end());
  const auto r_min = *r_minmax.first;
  const auto r_max = *r_minmax.second;

  if (r_min < 0.0) {
    return std::make_pair(Status::NegativeRadius,
                          IterInfo{current_iter_, r_min, r_max});
  }

  // Evaluate the current residual on the surface
  const auto one_form_magnitude =
      magnitude(db::get<StrahlkorperTags::NormalOneForm<Frame>>(box),
                upper_spatial_metric);
  const DataVector one_over_one_form_magnitude = 1.0 / get(one_form_magnitude);
  const auto unit_normal_one_form = StrahlkorperGr::unit_normal_one_form(
      db::get<StrahlkorperTags::NormalOneForm<Frame>>(box),
      one_over_one_form_magnitude);
  const auto inverse_surface_metric = StrahlkorperGr::inverse_surface_metric(
      raise_or_lower_index(unit_normal_one_form, upper_spatial_metric),
      upper_spatial_metric);

  const auto surface_residual = StrahlkorperGr::expansion(
      StrahlkorperGr::grad_unit_normal_one_form(
          db::get<StrahlkorperTags::Rhat<Frame>>(box),
          db::get<StrahlkorperTags::Radius<Frame>>(box), unit_normal_one_form,
          db::get<StrahlkorperTags::D2xRadius<Frame>>(box),
          one_over_one_form_magnitude, christoffel_2nd_kind),
      inverse_surface_metric, extrinsic_curvature);

  auto weighted_residual = get(surface_residual);
  switch (flow_) {
    case FlowType::Jacobi:
      // Do nothing
      break;
    case FlowType::Curvature: {
      weighted_residual *= get(one_form_magnitude);
    } break;
    case FlowType::Fast: {
      weighted_residual *= fast_flow_weight<Frame>(
          one_form_magnitude, db::get<StrahlkorperTags::Rhat<Frame>>(box),
          db::get<StrahlkorperTags::Radius<Frame>>(box),
          inverse_surface_metric);
    } break;
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Cannot find the specified value of FlowType, need to add a case?");
      // LCOV_EXCL_STOP
  }

  // Norm of the residual on the surface of size l_mesh.
  // Note: In SpEC, this norm is computed as a pointwise L2 norm.  But
  // here we compute the L2 integral norm.  The integral should be
  // more accurate, but if it turns out that this integral is
  // expensive, we can switch back to the pointwise L2 norm.
  const double residual_mesh_norm = sqrt(strahlkorper.ylm_spherepack().average(
      strahlkorper.ylm_spherepack().phys_to_spec(square(weighted_residual))));

  if (residual_mesh_norm < min_residual_mesh_norm_) {
    min_residual_mesh_norm_ = residual_mesh_norm;
    iter_at_min_residual_mesh_norm_ = current_iter_;
  }

  // Transform the weighted residual
  const auto weighted_residual_coefs =
      strahlkorper.ylm_spherepack().phys_to_spec(weighted_residual);

  // Restrict to the basis of the surface
  const auto residual_on_surface =
      strahlkorper.ylm_spherepack().prolong_or_restrict(
          weighted_residual_coefs, current_strahlkorper->ylm_spherepack());

  // Evaluate the norm of the residual on the surface of size l_surface.
  // See comment on pointwise norm vs integral norm above.
  const auto residual_ylm_norm =
      sqrt(current_strahlkorper->ylm_spherepack().average(
          current_strahlkorper->ylm_spherepack().phys_to_spec(
              square(current_strahlkorper->ylm_spherepack().spec_to_phys(
                  residual_on_surface)))));

  // Fill iter_info
  const auto minmax_residual =
      std::minmax_element(weighted_residual.begin(), weighted_residual.end());
  IterInfo iter_info{current_iter_,
                     r_min,
                     r_max,
                     *minmax_residual.first,
                     *minmax_residual.second,
                     residual_ylm_norm,
                     residual_mesh_norm};

  // Exit if converged.
  // What should happen is that as iterations proceed,
  // residual_mesh_norm approaches a constant (the truncation error), and
  // residual_ylm_norm decreases to roundoff.
  //
  // The first convergence condition is just
  // residual_ylm_norm<abs_tol_.  The second convergence condition has
  // two parts: First, we require that residual_ylm_norm <
  // trunc_tol_*residual_mesh_norm.  But this may happen because
  // residual_mesh_norm grows quickly, rather than because
  // residual_ylm_norm shrinks.  So we require also that
  // residual_mesh_norm-previous_residual_mesh_norm_ is small; that
  // is, that residual_mesh_norm is converging to something.  But we
  // can't require that
  // residual_mesh_norm-previous_residual_mesh_norm_ is small on the
  // first step, since previous_residual_mesh_norm_ is not defined, so
  // we skip this part of the check on the first iteration.
  if (residual_ylm_norm < abs_tol_) {
    // clang-tidy: std::move of trivially-copyable type
    return std::make_pair(Status::AbsTol, std::move(iter_info));  // NOLINT
  } else if (residual_ylm_norm < trunc_tol_ * residual_mesh_norm) {
    // This may be convergence by TruncationTol, but first make sure
    // that either residual_mesh_norm is converging, or that it is the
    // first step (i.e. previous_residual_mesh_norm_ == 0) so that we
    // don't know yet if residual_mesh_norm is converging.
    if (previous_residual_mesh_norm_ == 0 or
        equal_within_roundoff(residual_mesh_norm, previous_residual_mesh_norm_,
                              divergence_tol_ - 1.0, 0.0)) {
      // clang-tidy: std::move of trivially-copyable type
      return std::make_pair(Status::TruncationTol,
                            std::move(iter_info));  // NOLINT
    }
  }

  // Treat the case in which residual_mesh_norm is increasing
  if (residual_mesh_norm > divergence_tol_ * min_residual_mesh_norm_ and
      iter_at_min_residual_mesh_norm_ <= current_iter_ - divergence_iter_) {
    // clang-tidy: std::move of trivially-copyable type
    return std::make_pair(Status::DivergenceError,
                          std::move(iter_info));  // NOLINT
  }

  if (current_iter_ == max_its_) {
    // clang-tidy: std::move of trivially-copyable type
    return std::make_pair(Status::MaxIts, std::move(iter_info));  // NOLINT
  }

  // We have succeeded in an iteration.  So return the next guess.
  ++current_iter_;

  // Construct new coefs.  Parameters flow_A and flow_B are from
  // Gundlach, PRD 57, 863 (1998), eq. 44.
  const double flow_A = alpha_ / (l_surface * (l_surface + 1)) + beta_;
  const double flow_B = beta_ / alpha_;
  auto coefs = current_strahlkorper->coefficients();
  for (auto cit = SpherepackIterator(current_strahlkorper->l_max(),
                                     current_strahlkorper->l_max());
       cit; ++cit) {
    coefs[cit()] -= flow_A / (1.0 + flow_B * cit.l() * (cit.l() + 1)) *
                    residual_on_surface[cit()];
  }
  *current_strahlkorper = Strahlkorper<Frame>(coefs, *current_strahlkorper);

  // Set up for next iter
  previous_residual_mesh_norm_ = residual_mesh_norm;

  // clang-tidy: std::move of trivially-copyable type
  return std::make_pair(Status::SuccessfulIteration,
                        std::move(iter_info));  // NOLINT
}

std::ostream& operator<<(std::ostream& os,
                         const FastFlow::FlowType& flow_type) noexcept {
  switch (flow_type) {
    case FastFlow::FlowType::Jacobi:
      return os << "Jacobi";
    case FastFlow::FlowType::Curvature:
      return os << "Curvature";
    case FastFlow::FlowType::Fast:
      return os << "Fast";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Unknown FastFlow::FlowType");
      // LCOV_EXCL_STOP
  }
}

void FastFlow::pup(PUP::er& p) noexcept {
  p | alpha_;
  p | beta_;
  p | abs_tol_;
  p | trunc_tol_;
  p | divergence_tol_;
  p | divergence_iter_;
  p | max_its_;
  p | flow_;
  p | current_iter_;
  p | previous_residual_mesh_norm_;
  p | min_residual_mesh_norm_;
  p | iter_at_min_residual_mesh_norm_;
}

std::ostream& operator<<(std::ostream& os,
                         const FastFlow::Status& status) noexcept {
  switch (status) {
    case FastFlow::Status::SuccessfulIteration:
      return os << "Still iterating";
    case FastFlow::Status::AbsTol:
      return os << "Converged: Absolute tolerance";
    case FastFlow::Status::TruncationTol:
      return os << "Converged: Truncation tolerance";
    case FastFlow::Status::MaxIts:
      return os << "Failed: Too many iterations";
    case FastFlow::Status::NegativeRadius:
      return os << "Failed: Negative radius";
    case FastFlow::Status::DivergenceError:
      return os << "Failed: Diverging";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Need to add another case, don't understand value of 'status'");
      // LCOV_EXCL_STOP
  }
}

bool operator==(const FastFlow& lhs, const FastFlow& rhs) noexcept {
  return lhs.alpha_ == rhs.alpha_ and lhs.beta_ == rhs.beta_ and
         lhs.abs_tol_ == rhs.abs_tol_ and lhs.trunc_tol_ == rhs.trunc_tol_ and
         lhs.divergence_tol_ == rhs.divergence_tol_ and
         lhs.divergence_iter_ == rhs.divergence_iter_ and
         lhs.max_its_ == rhs.max_its_ and lhs.flow_ == rhs.flow_ and
         lhs.current_iter_ == rhs.current_iter_ and
         lhs.previous_residual_mesh_norm_ ==
             rhs.previous_residual_mesh_norm_ and
         lhs.min_residual_mesh_norm_ == rhs.min_residual_mesh_norm_ and
         lhs.iter_at_min_residual_mesh_norm_ ==
             rhs.iter_at_min_residual_mesh_norm_;
}

template <>
FastFlow::FlowType create_from_yaml<FastFlow::FlowType>::create<void>(
    const Option& options) {
  const std::string flow_type_read = options.parse_as<std::string>();
  if ("Jacobi" == flow_type_read) {
    return FastFlow::FlowType::Jacobi;
  } else if ("Curvature" == flow_type_read) {
    return FastFlow::FlowType::Curvature;
  } else if ("Fast" == flow_type_read) {
    return FastFlow::FlowType::Fast;
  }
  PARSE_ERROR(options.context(), "Failed to convert \""
                                     << flow_type_read
                                     << "\" to FastFlow::FlowType. Must be "
                                        "one of Jacobi, Curvature, Fast.");
}

template size_t FastFlow::current_l_mesh(
    const Strahlkorper<Frame::Inertial>& strahlkorper) const noexcept;

template std::pair<FastFlow::Status, FastFlow::IterInfo>
FastFlow::iterate_horizon_finder<Frame::Inertial>(
    const gsl::not_null<Strahlkorper<Frame::Inertial>*> current_strahlkorper,
    const tnsr::II<DataVector, 3, Frame::Inertial>& upper_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    const tnsr::Ijj<DataVector, 3, Frame::Inertial>&
        christoffel_2nd_kind) noexcept;
