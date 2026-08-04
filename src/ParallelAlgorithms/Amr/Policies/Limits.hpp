// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <utility>

#include "NumericalAlgorithms/Spectral/Hash.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Spectral {
enum class Basis : uint8_t;
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

namespace amr {
/// \brief The limits on refinement level and resolution for AMR
///
/// \details
/// - For a default constructed Limits, the refinement level is
///   bounded between 0 and ElementId<Dim>::max_refinement_level, and the
///   resolution is bounded between Spectral::limits::min and
///   Spectral::limits::max which are limits based on the implementation details
///   of ElementId and Mesh. ErrorBeyondLimits is set to false.
/// - Specifying `Auto` for any option uses the above limits.
/// - If you specify a lower bound that violates the above limits, the limit
///   will be raised to that needed by the code.
/// - If you specify an upper bound that violates the above limits, the code
///   will error on parsing the option.
class Limits {
 public:
  /// Inclusive bounds on the refinement level
  struct RefinementLevel {
    using type = Options::Auto<std::array<size_t, 2>>;
    static constexpr Options::String help = {
        "Inclusive bounds on the refinement level for AMR."};
  };

  /// Inclusive bounds on the number of polynomial modes for a I1 or B1 topology
  struct NumPolynomialModes {
    using type = Options::Auto<std::array<size_t, 2>>;
    static constexpr Options::String help = {
        "Inclusive bounds on the number of polynomial modes for a I1 or B1 "
        "topology."};
  };

  /// Inclusive bounds on the \f$m\f$ of Fourier modes for a S1 or B2 topology
  struct FourierM {
    using type = Options::Auto<std::array<size_t, 2>>;
    static constexpr Options::String help = {
        "Inclusive bounds on m for the Fourier series in topology S1 or B2."};
  };

  /// Inclusive bounds on the \f$\ell\f$ of spherical harmonic modes
  struct SphericalHarmonicL {
    using type = Options::Auto<std::array<size_t, 2>>;
    static constexpr Options::String help = {
        "Inclusive bounds on l for spherical harmonics in topology S2 or B3."};
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

  using options = tmpl::list<RefinementLevel, NumPolynomialModes, FourierM,
                             SphericalHarmonicL, ErrorBeyondLimits>;

  static constexpr Options::String help = {
      "Limits on refinement level and resolution for adaptive mesh "
      "refinement."};

  Limits();

  Limits(const std::optional<std::array<size_t, 2>>& refinement_level_bounds,
         const std::optional<std::array<size_t, 2>>& polynomial_mode_bounds,
         const std::optional<std::array<size_t, 2>>& fourier_mode_bounds,
         const std::optional<std::array<size_t, 2>>& spherical_harmonic_bounds,
         bool error_beyond_limits, const Options::Context& context = {});

  size_t minimum_refinement_level() const { return minimum_refinement_level_; }
  size_t maximum_refinement_level() const { return maximum_refinement_level_; }
  size_t minimum_resolution(Spectral::Basis basis,
                            Spectral::Quadrature quadrature) const;
  size_t maximum_resolution(Spectral::Basis basis,
                            Spectral::Quadrature quadrature) const;
  bool error_beyond_limits() const { return error_beyond_limits_; }

  std::ostream& print(std::ostream& os) const;
  void pup(PUP::er& p);

 private:
  friend bool operator==(const Limits& lhs, const Limits& rhs);
  size_t minimum_refinement_level_{0};
  size_t maximum_refinement_level_{16};
  std::unordered_map<std::pair<Spectral::Basis, Spectral::Quadrature>, size_t>
      minimum_resolution_{};
  std::unordered_map<std::pair<Spectral::Basis, Spectral::Quadrature>, size_t>
      maximum_resolution_{};
  bool error_beyond_limits_{false};
};

std::ostream& operator<<(std::ostream& os, const Limits& limits);

bool operator!=(const Limits& lhs, const Limits& rhs);
}  // namespace amr
