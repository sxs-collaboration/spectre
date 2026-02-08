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
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace GrSelfForce::AnalyticData {

/*!
 * \brief Gravitational self-force of a point mass on a circular equatorial
 * orbit in Kerr.
 *
 * This class defines the gravitational self-force equations for a circular
 * orbit by setting the coefficients $\alpha$, $\beta$, and $\gamma$ (see
 * `GrSelfForce::FirstOrderSystem`). It also sets the effective source
 * $S_m^\mathrm{eff}$ and the singular field $\Psi_m^P$ in the regularized
 * region. The coefficients are computed using Mathematica-generated functions
 * (see CircularOrbitCoeffs.hpp) and the effective source is computed using the
 * GravitationalEffectiveSource code by Wardell et. al.
 * (https://github.com/barrywardell/GravitationalEffectiveSource) and then
 * transformed to our form of the equations with more Mathematica-generated
 * functions (see CircularOrbitConvertEffsource.hpp). The derivation of these
 * equations will be presented in a future publication. A very strong test of
 * the validity of these equations is evaluating them on the singular field and
 * the corresponding effective source provided by the external
 * GravitationalEffectiveSource code (see Test_CircularOrbit.cpp).
 */
class CircularOrbit
    : public SPECTRE_CHARM_DERIVED(CircularOrbit,
                                   elliptic::analytic_data::Background),
      public SPECTRE_CHARM_DERIVED(CircularOrbit,
                                   elliptic::analytic_data::InitialGuess) {
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
        "Mode number 'm' of the m-mode decomposition";
    using type = int;
  };
  using options =
      tmpl::list<BlackHoleMass, BlackHoleSpin, OrbitalRadius, MModeNumber>;
  static constexpr Options::String help =
      "Quasicircular orbit of a point mass in Kerr spacetime";

  CircularOrbit() = default;
  CircularOrbit(const CircularOrbit&) = default;
  CircularOrbit& operator=(const CircularOrbit&) = default;
  CircularOrbit(CircularOrbit&&) = default;
  CircularOrbit& operator=(CircularOrbit&&) = default;
  ~CircularOrbit() override = default;

  CircularOrbit(double black_hole_mass, double black_hole_spin,
                double orbital_radius, int m_mode_number);

  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CircularOrbit);

  tnsr::I<double, 2> puncture_position() const;
  double black_hole_mass() const { return black_hole_mass_; }
  double black_hole_spin() const { return black_hole_spin_; }
  double orbital_radius() const { return orbital_radius_; }
  int m_mode_number() const { return m_mode_number_; }

  using background_tags =
      tmpl::list<Tags::Alpha, Tags::Beta, Tags::GammaRstar, Tags::GammaTheta>;
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

}  // namespace GrSelfForce::AnalyticData
