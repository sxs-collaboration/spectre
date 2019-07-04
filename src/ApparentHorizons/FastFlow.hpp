// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <ostream>
#include <utility>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
template <typename>
class Strahlkorper;
/// \endcond

/// \ingroup SurfacesGroup
/// \brief Fast flow method for finding apparent horizons.
///
/// \details Based on \cite Gundlach1997us.
//  The method is iterative.
class FastFlow {
 public:
  enum class FlowType { Jacobi, Curvature, Fast };

  // Error conditions are negative, successes are nonnegative
  enum class Status {
    // SuccessfulIteration means we should iterate again.
    SuccessfulIteration = 0,
    // The following indicate convergence, can stop iterating.
    AbsTol = 1,
    TruncationTol = 2,
    // The following indicate errors.
    MaxIts = -1,
    NegativeRadius = -2,
    DivergenceError = -3
  };

  /// Holds information about an iteration of the algorithm.
  struct IterInfo {
    size_t iteration{std::numeric_limits<size_t>::max()};
    double r_min{std::numeric_limits<double>::signaling_NaN()},
        r_max{std::numeric_limits<double>::signaling_NaN()},
        min_residual{std::numeric_limits<double>::signaling_NaN()},
        max_residual{std::numeric_limits<double>::signaling_NaN()},
        residual_ylm{std::numeric_limits<double>::signaling_NaN()},
        residual_mesh{std::numeric_limits<double>::signaling_NaN()};
  };

  struct Flow {
    using type = FlowType;
    static constexpr OptionString help = {
        "Flow method: Jacobi, Curvature, or Fast"};
    static type default_value() noexcept { return FlowType::Fast; }
  };

  struct Alpha {
    using type = double;
    static constexpr OptionString help = {
        "Alpha parameter in PRD 57, 863 (1998)"};
    static type default_value() noexcept { return 1.0; }
  };

  struct Beta {
    using type = double;
    static constexpr OptionString help = {
        "Beta parameter in PRD 57, 863 (1998)"};
    static type default_value() noexcept { return 0.5; }
  };

  struct AbsTol {
    using type = double;
    static constexpr OptionString help = {
        "Convergence found if R_{Y_lm} < AbsTol"};
    static type default_value() noexcept { return 1.e-12; }
  };

  struct TruncationTol {
    using type = double;
    static constexpr OptionString help = {
        "Convergence found if R_{Y_lm} < TruncationTol*R_{mesh}"};
    static type default_value() noexcept { return 1.e-2; }
  };

  struct DivergenceTol {
    using type = double;
    static constexpr OptionString help = {
        "Fraction that residual can increase before dying"};
    static type default_value() noexcept { return 1.2; }
    static type lower_bound() noexcept { return 1.0; }
  };

  struct DivergenceIter {
    using type = size_t;
    static constexpr OptionString help = {
        "Num iterations residual can increase before dying"};
    static type default_value() noexcept { return 5; }
  };

  struct MaxIts {
    using type = size_t;
    static constexpr OptionString help = {"Maximum number of iterations."};
    static type default_value() noexcept { return 100; }
  };

  using options = tmpl::list<Flow, Alpha, Beta, AbsTol, TruncationTol,
                             DivergenceTol, DivergenceIter, MaxIts>;

  static constexpr OptionString help{
      "Find a Strahlkorper using a 'fast flow' method.\n"
      "Based on Gundlach, PRD 57, 863 (1998).\n"
      "Expands the surface in terms of spherical harmonics Y_lm up to a given\n"
      "l_surface, and varies the coefficients S_lm where 0<=l<=l_surface to\n"
      "minimize the residual of the apparent horizon equation.  Also keeps\n"
      "another representation of the surface that is expanded up to\n"
      "l_mesh > l_surface.  Let R_{Y_lm} be the residual computed using the\n"
      "surface represented up to l_surface; this residual can in principle be\n"
      "lowered to machine roundoff by enough iterations. Let R_{mesh} be the\n"
      "residual computed using the surface represented up to l_mesh; this\n"
      "residual represents the truncation error, since l_mesh>l_surface and\n"
      "since coefficients S_lm with l>l_surface are not modified in the\n"
      "iteration.\n\n"
      "Convergence is achieved if R_{Y_lm}< TruncationTol*R_{mesh}, or if\n"
      "R_{Y_lm}<AbsTol, where TruncationTol and AbsTol are input parameters.\n"
      "If instead |R_{mesh}|_i > DivergenceTol * min_{j}(|R_{mesh}|_j) where\n"
      "i is the iteration index and j runs from 0 to i-DivergenceIter, then\n"
      "FastFlow exits with Status::DivergenceError.  Here DivergenceIter and\n"
      "DivergenceTol are input parameters."};

  FastFlow(Flow::type flow, Alpha::type alpha, Beta::type beta,
           AbsTol::type abs_tol, TruncationTol::type trunc_tol,
           DivergenceTol::type divergence_tol,
           DivergenceIter::type divergence_iter, MaxIts::type max_its) noexcept;

  FastFlow() noexcept
      : FastFlow(FlowType::Fast, 1.0, 0.5, 1.e-12, 1.e-2, 1.2, 5, 100) {}

  FastFlow(const FastFlow& /*rhs*/) = default;
  FastFlow& operator=(const FastFlow& /*rhs*/) = default;
  FastFlow(FastFlow&& /*rhs*/) noexcept = default;
  FastFlow& operator=(FastFlow&& /*rhs*/) noexcept = default;
  ~FastFlow() = default;

  // clang-tidy: no runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

  /// Evaluate residuals and compute the next iteration.  If
  /// Status==SuccessfulIteration, then `current_strahlkorper` is
  /// modified and `current_iteration()` is incremented.  Otherwise, we
  /// end with success or failure, and neither `current_strahlkorper`
  /// nor `current_iteration()` is changed.
  template <typename Frame>
  std::pair<Status, IterInfo> iterate_horizon_finder(
      gsl::not_null<Strahlkorper<Frame>*> current_strahlkorper,
      const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
      const tnsr::Ijj<DataVector, 3, Frame>& christoffel_2nd_kind) noexcept;

  size_t current_iteration() const noexcept { return current_iter_; }

  /// Given a Strahlkorper defined up to some maximum Y_lm l called
  /// l_surface, returns a larger value of l, l_mesh, that is used for
  /// evaluating convergence.
  template <typename Frame>
  size_t current_l_mesh(const Strahlkorper<Frame>& strahlkorper) const noexcept;

  /// Resets the finder.
  SPECTRE_ALWAYS_INLINE void reset_for_next_find() noexcept {
    current_iter_ = 0;
    previous_residual_mesh_norm_ = 0.0;
    min_residual_mesh_norm_ = std::numeric_limits<double>::max();
    iter_at_min_residual_mesh_norm_ = 0;
  }

 private:
  friend bool operator==(const FastFlow& /*lhs*/,
                         const FastFlow& /*rhs*/) noexcept;
  double alpha_, beta_, abs_tol_, trunc_tol_, divergence_tol_;
  size_t divergence_iter_, max_its_;
  FlowType flow_;
  size_t current_iter_;
  double previous_residual_mesh_norm_, min_residual_mesh_norm_;
  size_t iter_at_min_residual_mesh_norm_;
};

SPECTRE_ALWAYS_INLINE bool converged(const FastFlow::Status& status) noexcept {
  return static_cast<int>(status) > 0;
}

std::ostream& operator<<(std::ostream& os,
                         const FastFlow::FlowType& flow_type) noexcept;

std::ostream& operator<<(std::ostream& os,
                         const FastFlow::Status& status) noexcept;

SPECTRE_ALWAYS_INLINE bool operator!=(const FastFlow& lhs,
                                      const FastFlow& rhs) noexcept {
  return not(lhs == rhs);
}

template <>
struct create_from_yaml<FastFlow::FlowType> {
  template <typename Metavariables>
  static FastFlow::FlowType create(const Option& options) {
    return create<void>(options);
  }
};
template <>
FastFlow::FlowType create_from_yaml<FastFlow::FlowType>::create<void>(
    const Option& options);
