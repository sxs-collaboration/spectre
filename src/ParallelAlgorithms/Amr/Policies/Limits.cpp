// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Policies/Limits.hpp"

#include <pup.h>
#include <string>

#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Limits.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/StdHelpers.hpp"

namespace amr {
namespace {
std::unordered_map<std::pair<Spectral::Basis, Spectral::Quadrature>, size_t>
make_map(const size_t polynomial_n, const size_t fourier_m,
         const size_t spherical_harmonic_l) {
  std::unordered_map<std::pair<Spectral::Basis, Spectral::Quadrature>, size_t>
      result{};
  result.emplace(std::make_pair(Spectral::Basis::Chebyshev,
                                Spectral::Quadrature::GaussLobatto),
                 polynomial_n == 0 ? 2 : polynomial_n + 1);
  result.emplace(
      std::make_pair(Spectral::Basis::Chebyshev, Spectral::Quadrature::Gauss),
      polynomial_n + 1);
  result.emplace(std::make_pair(Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto),
                 polynomial_n == 0 ? 2 : polynomial_n + 1);
  result.emplace(
      std::make_pair(Spectral::Basis::Legendre, Spectral::Quadrature::Gauss),
      polynomial_n + 1);
  result.emplace(std::make_pair(Spectral::Basis::SphericalHarmonic,
                                Spectral::Quadrature::Gauss),
                 spherical_harmonic_l + 1);
  result.emplace(std::make_pair(Spectral::Basis::SphericalHarmonic,
                                Spectral::Quadrature::Equiangular),
                 2 * spherical_harmonic_l + 1);
  result.emplace(std::make_pair(Spectral::Basis::Fourier,
                                Spectral::Quadrature::Equiangular),
                 2 * fourier_m + 1);
  result.emplace(std::make_pair(Spectral::Basis::ZernikeB1,
                                Spectral::Quadrature::GaussRadauUpper),
                 polynomial_n + 1);
  result.emplace(std::make_pair(Spectral::Basis::ZernikeB2,
                                Spectral::Quadrature::GaussRadauUpper),
                 fourier_m / 2 + 1);
  result.emplace(std::make_pair(Spectral::Basis::ZernikeB2,
                                Spectral::Quadrature::Equiangular),
                 2 * fourier_m + 1);
  result.emplace(std::make_pair(Spectral::Basis::ZernikeB3,
                                Spectral::Quadrature::GaussRadauUpper),
                 spherical_harmonic_l / 2 + 1);
  result.emplace(
      std::make_pair(Spectral::Basis::ZernikeB3, Spectral::Quadrature::Gauss),
      spherical_harmonic_l + 1);
  result.emplace(std::make_pair(Spectral::Basis::ZernikeB3,
                                Spectral::Quadrature::Equiangular),
                 2 * spherical_harmonic_l + 1);
  result.emplace(std::make_pair(Spectral::Basis::Cartoon,
                                Spectral::Quadrature::SphericalSymmetry),
                 1);
  result.emplace(std::make_pair(Spectral::Basis::Cartoon,
                                Spectral::Quadrature::AxialSymmetry),
                 1);
  return result;
}
}  // namespace

Limits::Limits()
    : maximum_refinement_level_(ElementId<1>::max_refinement_level),
      minimum_resolution_(
          make_map(0, 0, Spectral::limits::min_spherical_harmonic_mode)),
      maximum_resolution_(
          make_map(Spectral::limits::max_i1_polynomial_mode,
                   Spectral::limits::max_fourier_mode,
                   Spectral::limits::max_spherical_harmonic_mode)) {}

Limits::Limits(
    const std::optional<std::array<size_t, 2>>& refinement_level_bounds,
    const std::optional<std::array<size_t, 2>>& polynomial_mode_bounds,
    const std::optional<std::array<size_t, 2>>& fourier_mode_bounds,
    const std::optional<std::array<size_t, 2>>& spherical_harmonic_bounds,
    const bool error_beyond_limits, const Options::Context& context)
    : minimum_refinement_level_(refinement_level_bounds.has_value()
                                    ? refinement_level_bounds.value()[0]
                                    : 0),
      maximum_refinement_level_(refinement_level_bounds.has_value()
                                    ? refinement_level_bounds.value()[1]
                                    : ElementId<1>::max_refinement_level),
      minimum_resolution_(make_map(
          polynomial_mode_bounds.has_value() ? polynomial_mode_bounds.value()[0]
                                             : 0,
          fourier_mode_bounds.has_value() ? fourier_mode_bounds.value()[0] : 0,
          spherical_harmonic_bounds.has_value()
              ? spherical_harmonic_bounds.value()[0]
              : Spectral::limits::min_spherical_harmonic_mode)),
      maximum_resolution_(make_map(
          polynomial_mode_bounds.has_value()
              ? polynomial_mode_bounds.value()[1]
              : Spectral::limits::max_i1_polynomial_mode,
          fourier_mode_bounds.has_value() ? fourier_mode_bounds.value()[1]
                                          : Spectral::limits::max_fourier_mode,
          spherical_harmonic_bounds.has_value()
              ? spherical_harmonic_bounds.value()[1]
              : Spectral::limits::max_spherical_harmonic_mode)),
      error_beyond_limits_(error_beyond_limits) {
  if (minimum_refinement_level_ > maximum_refinement_level_) {
    PARSE_ERROR(context, "RefinementLevel lower bound '" +
                             std::to_string(minimum_refinement_level_) +
                             "' cannot be larger than upper bound '" +
                             std::to_string(maximum_refinement_level_) + "'.");
  }
  if (maximum_refinement_level_ > ElementId<1>::max_refinement_level) {
    PARSE_ERROR(context,
                "RefinementLevel upper bound '" +
                    std::to_string(maximum_refinement_level_) +
                    "' cannot be larger than refinement limit '" +
                    std::to_string(ElementId<1>::max_refinement_level) + "'.");
  }
  if (polynomial_mode_bounds.has_value()) {
    const auto [min, max] = polynomial_mode_bounds.value();
    if (min > max) {
      PARSE_ERROR(context, "NumPolynomialModes lower bound '" +
                               std::to_string(min) +
                               "' cannot be larger than upper bound '" +
                               std::to_string(max));
    }
    if (max > Spectral::limits::max_i1_polynomial_mode) {
      PARSE_ERROR(context,
                  "NumPolynomialModes upper bound '" + std::to_string(max) +
                      "' cannot be larger than "
                      "Spectral::limits::max_i1_polynomial_mode '" +
                      std::to_string(Spectral::limits::max_i1_polynomial_mode));
    }
  }
  if (fourier_mode_bounds.has_value()) {
    const auto [min, max] = fourier_mode_bounds.value();
    if (min > max) {
      PARSE_ERROR(context, "FourierM lower bound '" + std::to_string(min) +
                               "' cannot be larger than upper bound '" +
                               std::to_string(max));
    }
    if (max > Spectral::limits::max_fourier_mode) {
      PARSE_ERROR(context,
                  "FourierM upper bound '" + std::to_string(max) +
                      "' cannot be larger than "
                      "Spectral::limits::max_fourier_mode '" +
                      std::to_string(Spectral::limits::max_fourier_mode));
    }
  }
  if (spherical_harmonic_bounds.has_value()) {
    const auto [min, max] = spherical_harmonic_bounds.value();
    if (min > max) {
      PARSE_ERROR(context, "SphericalHarmonicL lower bound '" +
                               std::to_string(min) +
                               "' cannot be larger than upper bound '" +
                               std::to_string(max));
    }
    if (max > Spectral::limits::max_spherical_harmonic_mode) {
      PARSE_ERROR(
          context,
          "SphericalHarmonicL upper bound '" + std::to_string(max) +
              "' cannot be larger than "
              "Spectral::limits::max_spherical_harmonic_mode '" +
              std::to_string(Spectral::limits::max_spherical_harmonic_mode));
    }
  }
}

size_t Limits::minimum_resolution(const Spectral::Basis basis,
                                  const Spectral::Quadrature quadrature) const {
  return minimum_resolution_.at(std::pair{basis, quadrature});
}

size_t Limits::maximum_resolution(const Spectral::Basis basis,
                                  const Spectral::Quadrature quadrature) const {
  return maximum_resolution_.at(std::pair{basis, quadrature});
}

void Limits::pup(PUP::er& p) {
  p | minimum_refinement_level_;
  p | maximum_refinement_level_;
  p | minimum_resolution_;
  p | maximum_resolution_;
  p | error_beyond_limits_;
}

bool operator==(const Limits& lhs, const Limits& rhs) {
  return lhs.minimum_refinement_level_ == rhs.minimum_refinement_level_ and
         lhs.maximum_refinement_level_ == rhs.maximum_refinement_level_ and
         lhs.minimum_resolution_ == rhs.minimum_resolution_ and
         lhs.maximum_resolution_ == rhs.maximum_resolution_ and
         lhs.error_beyond_limits_ == rhs.error_beyond_limits_;
}

bool operator!=(const Limits& lhs, const Limits& rhs) {
  return not(lhs == rhs);
}

std::ostream& Limits::print(std::ostream& os) const {
  using ::operator<<;
  os << "ErrorBeyondLimits: " << std::boolalpha << error_beyond_limits_ << "\n";
  os << "RefinementLevel: [" << minimum_refinement_level_ << ", "
     << maximum_refinement_level_ << "]\n";
  os << "Minimum resolution:\n" << minimum_resolution_ << "\n";
  os << "Maximum resolution:\n" << maximum_resolution_ << "\n";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Limits& limits) {
  return limits.print(os);
}
}  // namespace amr
