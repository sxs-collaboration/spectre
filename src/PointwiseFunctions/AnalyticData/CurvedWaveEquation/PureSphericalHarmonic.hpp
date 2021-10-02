// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <pup.h>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace CurvedScalarWave::AnalyticData {

/*!
 * \brief Analytic initial data for a pure spherical harmonic in three
 * dimensions.
 *
 * \details The initial data is taken from \cite Scheel2003vs , Eqs. 4.1--4.3,
 * and sets the evolved variables of the scalar wave as follows:
 *
 * \f{align}
 * \Psi &= 0 \\
 * \Phi_i &= 0 \\
 * \Pi &= \Pi_0(r, \theta, \phi) =  e^{- (r - r_0)^2 / w^2} Y_{lm}(\theta,
 * \phi), \f}
 *
 * where \f$r_0\f$ is the radius of the profile and \f$w\f$ is its width. This
 * describes a pure spherical harmonic mode \f$Y_{lm}(\theta, \phi)\f$ truncated
 * by a circular Gaussian window function.
 *
 * When evolved, the scalar field \f$\Phi\f$ will briefly build up around the
 * radius \f$r_0\f$ and then disperse. This can be used to study the ringdown
 * behavior and late-time tails in different background spacetimes.
 */

class PureSphericalHarmonic : public MarkAsAnalyticData {
 public:
  struct Radius {
    using type = double;
    static constexpr Options::String help = {
        "The radius of the spherical harmonic profile"};
    static type lower_bound() { return 0.0; }
  };

  struct Width {
    using type = double;
    static constexpr Options::String help = {
        "The width of the spherical harmonic profile."};
    static type lower_bound() { return 0.0; }
  };

  struct Mode {
    using type = std::pair<size_t, int>;
    static constexpr Options::String help = {
        "The l-mode and m-mode of the spherical harmonic Ylm"};
  };

  using options = tmpl::list<Radius, Width, Mode>;

  static constexpr Options::String help = {
      "Initial data for a pure spherical harmonic mode truncated by a circular "
      "Gaussian window funtion. The expression is taken from Scheel(2003), "
      "equations 4.1-4.3."};

  PureSphericalHarmonic() = default;

  PureSphericalHarmonic(double radius, double width,
                        std::pair<size_t, int> mode,
                        const Options::Context& context = {});

  /// Retrieve the evolution variables at spatial coordinates `x`
  tuples::TaggedTuple<CurvedScalarWave::Pi, CurvedScalarWave::Phi<3>,
                      CurvedScalarWave::Psi>
  variables(const tnsr::I<DataVector, 3>& x, double /*t*/,
            tmpl::list<CurvedScalarWave::Pi, CurvedScalarWave::Phi<3>,
                       CurvedScalarWave::Psi> /*meta*/) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/);

 private:
  double radius_{std::numeric_limits<double>::signaling_NaN()};
  double width_sq_{std::numeric_limits<double>::signaling_NaN()};
  std::pair<size_t, int> mode_{std::numeric_limits<size_t>::signaling_NaN(),
                               std::numeric_limits<int>::signaling_NaN()};

  friend bool operator==(const PureSphericalHarmonic& lhs,
                         const PureSphericalHarmonic& rhs);

  friend bool operator!=(const PureSphericalHarmonic& lhs,
                         const PureSphericalHarmonic& rhs);
};

}  // namespace CurvedScalarWave::AnalyticData
