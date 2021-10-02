// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Interval.hpp"

#include <cmath>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {

Interval::Interval(const double A, const double B, const double a,
                   const double b, const Distribution distribution,
                   const std::optional<double> singularity_pos)
    : A_(A),
      B_(B),
      a_(a),
      b_(b),
      distribution_(distribution),
      singularity_pos_(singularity_pos) {
  ASSERT(
      A != B and a != b,
      "The left and right boundaries for both source and target interval must "
      "differ, but are; ["
          << A << ", " << B << "] -> [" << a << ", " << b << "]");
  if (distribution == domain::CoordinateMaps::Distribution::Logarithmic or
      distribution == domain::CoordinateMaps::Distribution::Inverse) {
    ASSERT(singularity_pos.has_value(),
           "Both the logarithmic and inverse distribution require "
           "`singularity_pos` to be specified explicitly.");
    ASSERT(std::min(a, b) > singularity_pos.value() or
               std::max(a, b) < singularity_pos.value(),
           "The singularity for the logarithmic and inverse Interval falls "
           "inside the domain with ["
               << std::min(a, b) << ", " << std::max(a, b) << "], but is"
               << singularity_pos.value());
  }
  is_identity_ =
      distribution == domain::CoordinateMaps::Distribution::Linear and
      A == a and B == b;
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 1> Interval::operator()(
    const std::array<T, 1>& source_coords) const {
  switch (distribution_) {
    case Distribution::Linear: {
      return {{((b_ - a_) * source_coords[0] + a_ * B_ - b_ * A_) / (B_ - A_)}};
    }
    case Distribution::Equiangular: {
      return {
          {0.5 * (a_ + b_ +
                  (b_ - a_) * tan(M_PI_4 * (2.0 * source_coords[0] - B_ - A_) /
                                  (B_ - A_)))}};
    }
    case Distribution::Logarithmic: {
      const double singularity_pos = singularity_pos_.value();
      const double logarithmic_zero =
          0.5 * (log((b_ - singularity_pos) * (a_ - singularity_pos)));
      const double logarithmic_rate =
          0.5 * (log((b_ - singularity_pos) / (a_ - singularity_pos)));
      const double singularity_sign =
          std::min(a_, b_) > singularity_pos ? 1. : -1.;
      return {{singularity_sign *
                   exp(logarithmic_zero +
                       logarithmic_rate * (2.0 * source_coords[0] - B_ - A_) /
                           (B_ - A_)) +
               singularity_pos}};
    }
    case Distribution::Inverse: {
      const double singularity_pos = singularity_pos_.value();
      return {
          {2.0 * (a_ - singularity_pos) * (b_ - singularity_pos) /
               (a_ + b_ - 2.0 * singularity_pos -
                (b_ - a_) / (B_ - A_) * (2.0 * source_coords[0] - B_ - A_)) +
           singularity_pos}};
    }
    default:
      ERROR("Unknown domain::CoordinateMaps::Distribution type for Interval");
  }
}

std::optional<std::array<double, 1>> Interval::inverse(
    const std::array<double, 1>& target_coords) const {
  switch (distribution_) {
    case Distribution::Linear: {
      return {
          {{((B_ - A_) * target_coords[0] - a_ * B_ + b_ * A_) / (b_ - a_)}}};
    }
    case Distribution::Equiangular: {
      return {
          {{0.5 * (A_ + B_ +
                   (B_ - A_) / M_PI_4 *
                       atan((2.0 * target_coords[0] - a_ - b_) / (b_ - a_)))}}};
    }
    case Distribution::Logarithmic: {
      const double singularity_pos = singularity_pos_.value();
      const double logarithmic_zero =
          0.5 * (log((b_ - singularity_pos) * (a_ - singularity_pos)));
      const double logarithmic_rate =
          0.5 * (log((b_ - singularity_pos) / (a_ - singularity_pos)));
      const double singularity_sign =
          std::min(a_, b_) > singularity_pos ? 1. : -1.;
      return {{{0.5 * ((B_ - A_) *
                           (log(singularity_sign *
                                (target_coords[0] - singularity_pos)) -
                            logarithmic_zero) /
                           logarithmic_rate +
                       B_ + A_)}}};
    }
    case Distribution::Inverse: {
      const double singularity_pos = singularity_pos_.value();
      return {
          {{0.5 * (A_ + B_ +
                   (B_ - A_) / (b_ - a_) *
                       ((a_ - singularity_pos) + (b_ - singularity_pos) -
                        2. * (a_ - singularity_pos) * (b_ - singularity_pos) /
                            (target_coords[0] - singularity_pos)))}}};
    }
    default:
      ERROR("Unknown domain::CoordinateMaps::Distribution type for Interval");
  }
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> Interval::jacobian(
    const std::array<T, 1>& source_coords) const {
  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);
  switch (distribution_) {
    case Distribution::Linear: {
      get<0, 0>(jacobian_matrix) = (b_ - a_) / (B_ - A_);
      return jacobian_matrix;
    }
    case Distribution::Equiangular: {
      const tt::remove_cvref_wrap_t<T> tan_variable =
          tan((2.0 * source_coords[0] - B_ - A_) * M_PI_4 / (B_ - A_));
      get<0, 0>(jacobian_matrix) =
          M_PI_4 * (b_ - a_) / (B_ - A_) * (1.0 + square(tan_variable));
      return jacobian_matrix;
    }
    case Distribution::Logarithmic: {
      const double singularity_pos = singularity_pos_.value();
      const double logarithmic_zero =
          0.5 * (log((b_ - singularity_pos) * (a_ - singularity_pos)));
      const double logarithmic_rate =
          0.5 * (log((b_ - singularity_pos) / (a_ - singularity_pos)));
      const double singularity_sign =
          std::min(a_, b_) > singularity_pos ? 1. : -1.;
      get<0, 0>(jacobian_matrix) =
          2.0 / (B_ - A_) * logarithmic_rate * singularity_sign *
          exp(logarithmic_zero + logarithmic_rate *
                                     (2.0 * source_coords[0] - B_ - A_) /
                                     (B_ - A_));
      return jacobian_matrix;
    }
    case Distribution::Inverse: {
      const double singularity_pos = singularity_pos_.value();
      get<0, 0>(jacobian_matrix) =
          (a_ - singularity_pos) * (b_ - singularity_pos) * (b_ - a_) *
          (B_ - A_) /
          square((b_ - a_) * source_coords[0] + a_ * A_ - b_ * B_ +
                 singularity_pos * (B_ - A_));
      return jacobian_matrix;
    }
    default:
      ERROR("Unknown domain::CoordinateMaps::Distribution type for Interval");
  }
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> Interval::inv_jacobian(
    const std::array<T, 1>& source_coords) const {
  auto inv_jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);
  switch (distribution_) {
    case Distribution::Linear: {
      get<0, 0>(inv_jacobian_matrix) = (B_ - A_) / (b_ - a_);
      return inv_jacobian_matrix;
    }
    case Distribution::Equiangular: {
      const tt::remove_cvref_wrap_t<T> tan_variable =
          tan(M_PI_4 * (2.0 * source_coords[0] - B_ - A_) / (B_ - A_));
      get<0, 0>(inv_jacobian_matrix) =
          (B_ - A_) / (M_PI_4 * (b_ - a_) * (1.0 + square(tan_variable)));
      return inv_jacobian_matrix;
    }
    case Distribution::Logarithmic: {
      const double singularity_pos = singularity_pos_.value();
      const double logarithmic_zero =
          0.5 * (log((b_ - singularity_pos) * (a_ - singularity_pos)));
      const double logarithmic_rate =
          0.5 * (log((b_ - singularity_pos) / (a_ - singularity_pos)));
      const double singularity_sign =
          std::min(a_, b_) > singularity_pos ? 1. : -1.;
      get<0, 0>(inv_jacobian_matrix) =
          0.5 * (B_ - A_) / logarithmic_rate * singularity_sign *
          exp(-logarithmic_zero - logarithmic_rate *
                                      (2.0 * source_coords[0] - B_ - A_) /
                                      (B_ - A_));
      return inv_jacobian_matrix;
    }
    case Distribution::Inverse: {
      const double singularity_pos = singularity_pos_.value();
      get<0, 0>(inv_jacobian_matrix) =
          square((b_ - a_) * source_coords[0] + a_ * A_ - b_ * B_ +
                 singularity_pos * (B_ - A_)) /
          ((a_ - singularity_pos) * (b_ - singularity_pos) * (b_ - a_) *
           (B_ - A_));
      return inv_jacobian_matrix;
    }
    default:
      ERROR("Unknown domain::CoordinateMaps::Distribution type for Interval");
  }
}

void Interval::pup(PUP::er& p) {
  p | A_;
  p | B_;
  p | a_;
  p | b_;
  p | distribution_;
  p | singularity_pos_;
  p | is_identity_;
}

bool operator==(const CoordinateMaps::Interval& lhs,
                const CoordinateMaps::Interval& rhs) {
  return lhs.A_ == rhs.A_ and lhs.B_ == rhs.B_ and lhs.a_ == rhs.a_ and
         lhs.b_ == rhs.b_ and lhs.distribution_ == rhs.distribution_ and
         lhs.singularity_pos_ == rhs.singularity_pos_;
}

// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 1>                 \
  Interval::operator()(const std::array<DTYPE(data), 1>& source_coords) const; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame>   \
  Interval::jacobian(const std::array<DTYPE(data), 1>& source_coords) const;   \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame>   \
  Interval::inv_jacobian(const std::array<DTYPE(data), 1>& source_coords)      \
      const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef DTYPE
#undef INSTANTIATE
}  // namespace domain::CoordinateMaps
