// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace amr {
/// \brief The limits on refinement level and resolution for AMR
///
/// \details
/// - For a default constructed Limits, the refinement level is
///   bounded between 0 and ElementId<Dim>::max_refinement_level, and the
///   resolution is bounded between 1 and
///   Spectral::maximum_number_of_points<Spectral::Basis::Legendre>
///   which are limits based on the implementation details of ElementId and
///   Mesh. There is no error for attempting to go beyond the limits.
/// - If you specify the limits on the refinement levels and resolutions, they
///   must respect the above limits.
/// - Depending upon which Spectral::Basis is chosen, the actual minimum
///   resolution may be higher (usually 2), but this is automatically enforced
///   by EnforcePolicies.
class Limits {
 public:
  /// Inclusive bounds on the refinement level
  struct RefinementLevel {
    using type = Options::Auto<std::array<size_t, 2>>;
    static constexpr Options::String help = {
        "Inclusive bounds on the refinement level for AMR."};
  };

  /// Inclusive bounds on the number of grid points per dimension
  struct NumGridPoints {
    using type = Options::Auto<std::array<size_t, 2>>;
    static constexpr Options::String help = {
        "Inclusive bounds on the number of grid points per dimension for AMR."};
  };

  /// \brief Whether the code should error if EnforcePolicies has to prevent
  /// refinement from going beyond the given limits.
  ///
  /// \details The Limits class is just a holder for this value, the actual
  /// error happens in `amr::Actions::EvaluateRefinementCriteria` or
  /// `amr::Events::RefineMesh`
  struct ErrorBeyondLimits {
    using type = bool;
    static constexpr Options::String help = {
        "If adaptive mesh refinement tries to go beyond the RefinementLevel or "
        "NumGridPoints limit, error"};
  };

  using options = tmpl::list<RefinementLevel, NumGridPoints, ErrorBeyondLimits>;

  static constexpr Options::String help = {
      "Limits on refinement level and resolution for adaptive mesh "
      "refinement."};

  Limits();

  Limits(const std::optional<std::array<size_t, 2>>& refinement_level_bounds,
         const std::optional<std::array<size_t, 2>>& resolution_bounds,
         bool error_beyond_limits, const Options::Context& context = {});

  Limits(size_t minimum_refinement_level, size_t maximum_refinement_level,
         size_t minimum_resolution, size_t maximum_resolution);

  size_t minimum_refinement_level() const { return minimum_refinement_level_; }
  size_t maximum_refinement_level() const { return maximum_refinement_level_; }
  size_t minimum_resolution() const { return minimum_resolution_; }
  size_t maximum_resolution() const { return maximum_resolution_; }
  bool error_beyond_limits() const { return error_beyond_limits_; }

  void pup(PUP::er& p);

 private:
  size_t minimum_refinement_level_{0};
  size_t maximum_refinement_level_{16};
  size_t minimum_resolution_{1};
  size_t maximum_resolution_{20};
  bool error_beyond_limits_{false};
};

bool operator==(const Limits& lhs, const Limits& rhs);

bool operator!=(const Limits& lhs, const Limits& rhs);
}  // namespace amr
