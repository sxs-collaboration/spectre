// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <pup.h>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ScalarSelfForce::AnalyticData {

/*!
 * \brief Scalar self force for a scalar charge on a circular equatorial orbit.
 *
 * This class implements Eq. (2.9) in \cite Osburn:2022bby . It does so by
 * defining the background fields $\alpha$, $\beta$, and $\gamma_i$ in the
 * general form of the equations
 * \begin{equation}
 * -\partial_i F^i + \beta \Psi_m + \gamma_i F^i = S_m
 * \text{.}
 * \end{equation}
 * with the flux
 * \begin{equation}
 * F^i = \{\partial_{r_\star}, \alpha \partial_{\cos\theta}\} \Psi_m
 * \text{.}
 * \end{equation}
 * Note that we use $\cos\theta$ as angular coordinate but \cite Osburn:2022bby
 * uses $\theta$. We also multiply Eq. (2.9) by the factor $\Sigma^2 / (r^2 +
 * a^2)^2$ so we can easily write it in first-order flux form. The resulting
 * factors in the equation are:
 *
 * \begin{align}
 * &\alpha = \frac{\Delta}{(r^2 + a^2)^2} \sin^2\theta \\
 * &\beta = \left(-m^2\Omega^2 \Sigma^2 + 4a m^2 \Omega M r + \Delta \left[
 *   \frac{m^2}{\sin^2\theta} + \frac{2M}{r}(1-\frac{a^2}{Mr}) + \frac{2iam}{r}
 *   \right]\right) \frac{1}{(r^2 + a^2)^2} \\
 * &\gamma_{r_\star} = -\frac{2iam}{r^2+a^2} + \frac{2a^2}{r}
 *   \frac{\alpha}{\sin^2\theta} \\
 * &\gamma_{\cos\theta} = 0
 * \end{align}
 *
 * This class also provides the effective source $S_m^\mathrm{eff} = \Delta_m
 * \Psi_m^P$ and the singular field $\Psi_m^P$ in the regularized region (see
 * `ScalarSelfForce::FirstOrderSystem` and Sec. III in \cite Osburn:2022bby ).
 * The effective source is computed using the scalar EffectiveSource code by
 * Wardell et. al. (https://github.com/barrywardell/EffectiveSource and
 * \cite Wardell:2011gb ). It is then transformed to correspond to the m-mode
 * decomposition used in \cite Osburn:2022bby Eq. (2.8) as:
 * \begin{align}
 *   &\Psi_m^P = \frac{r}{2 \pi} e^{-i m \Delta\phi} \Phi_m^\mathrm{Wardell} \\
 *   &S_m^\mathrm{eff} = \frac{r}{2 \pi} e^{-i m \Delta\phi}
 *     \frac{\Delta\,(r^2 + a^2\cos^2\theta)}{(r^2 + a^2)^2}
 *     S_m^\mathrm{Wardell}
 *   \text{,}
 * \end{align}
 * where $\Delta\phi = \frac{a}{r_+ - r_-} \ln(\frac{r-r_+}{r-r_-})$ (Eq. (2.7)
 * in \cite Osburn:2022bby ).
 */
class CircularOrbit : public elliptic::analytic_data::Background,
                      public elliptic::analytic_data::InitialGuess {
 public:
  struct BlackHoleMass {
    static constexpr Options::String help =
        "Kerr mass parameter 'M' of the black hole";
    using type = double;
  };
  struct BlackHoleSpin {
    static constexpr Options::String help =
        "Kerr dimensionless spin parameter 'chi' of the black hole";
    using type = double;
  };
  struct OrbitalRadius {
    static constexpr Options::String help =
        "Radius 'r_0' of the circular orbit";
    using type = double;
  };
  struct MModeNumber {
    static constexpr Options::String help =
        "Mode number 'm' of the scalar field";
    using type = int;
  };
  using options =
      tmpl::list<BlackHoleMass, BlackHoleSpin, OrbitalRadius, MModeNumber>;
  static constexpr Options::String help =
      "Quasicircular orbit of a scalar point charge in Kerr spacetime";

  CircularOrbit() = default;
  CircularOrbit(const CircularOrbit&) = default;
  CircularOrbit& operator=(const CircularOrbit&) = default;
  CircularOrbit(CircularOrbit&&) = default;
  CircularOrbit& operator=(CircularOrbit&&) = default;
  ~CircularOrbit() override = default;

  CircularOrbit(double black_hole_mass, double black_hole_spin,
                double orbital_radius, int m_mode_number);

  explicit CircularOrbit(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CircularOrbit);

  tnsr::I<double, 2> puncture_position() const;
  double black_hole_mass() const { return black_hole_mass_; }
  double black_hole_spin() const { return black_hole_spin_; }
  double orbital_radius() const { return orbital_radius_; }
  int m_mode_number() const { return m_mode_number_; }

  using background_tags = tmpl::list<Tags::Alpha, Tags::Beta, Tags::Gamma>;
  using source_tags = tmpl::list<
      ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
      Tags::BoyerLindquistRadius>;

  // Background
  tuples::tagged_tuple_from_typelist<background_tags> variables(
      const tnsr::I<DataVector, 2>& x, background_tags /*meta*/) const;

  // Initial guess
  static tuples::TaggedTuple<Tags::MMode> variables(
      const tnsr::I<DataVector, 2>& x, tmpl::list<Tags::MMode> /*meta*/);

  // Fixed sources
  tuples::tagged_tuple_from_typelist<source_tags> variables(
      const tnsr::I<DataVector, 2>& x, source_tags /*meta*/) const;

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 2>& x, const Mesh<2>& /*mesh*/,
      const InverseJacobian<DataVector, 2, Frame::ElementLogical,
                            Frame::Inertial>& /*inv_jacobian*/,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables(x, tmpl::list<RequestedTags...>{});
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const CircularOrbit& lhs, const CircularOrbit& rhs);

  double black_hole_mass_{std::numeric_limits<double>::signaling_NaN()};
  double black_hole_spin_{std::numeric_limits<double>::signaling_NaN()};
  double orbital_radius_{std::numeric_limits<double>::signaling_NaN()};
  int m_mode_number_{};
};

bool operator!=(const CircularOrbit& lhs, const CircularOrbit& rhs);

}  // namespace ScalarSelfForce::AnalyticData
