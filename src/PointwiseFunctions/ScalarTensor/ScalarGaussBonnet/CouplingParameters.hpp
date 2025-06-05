// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ScalarTensor {

/*!
 * \brief Linear, quadratic and quartic coupling parameters to curvature.
 */
struct CouplingParameterOptions {
  static constexpr Options::String help = {
      "Options for coupling parameters to curvature."};

  struct Linear {
    using type = double;
    static constexpr Options::String help = "Linear coupling parameter.";
  };

  struct Quadratic {
    using type = double;
    static constexpr Options::String help = "Quadratic coupling parameter.";
  };

  struct Quartic {
    using type = double;
    static constexpr Options::String help = "Quartic coupling parameter.";
  };

  using options = tmpl::list<Linear, Quadratic, Quartic>;

  CouplingParameterOptions() = default;
  CouplingParameterOptions(double linear_in, double quadratic_in,
                           double quartic_in);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  double linear{std::numeric_limits<double>::signaling_NaN()};
  double quadratic{std::numeric_limits<double>::signaling_NaN()};
  double quartic{std::numeric_limits<double>::signaling_NaN()};
};

bool operator==(const CouplingParameterOptions& lhs,
                const CouplingParameterOptions& rhs);
bool operator!=(const CouplingParameterOptions& lhs,
                const CouplingParameterOptions& rhs);

}  // namespace ScalarTensor
