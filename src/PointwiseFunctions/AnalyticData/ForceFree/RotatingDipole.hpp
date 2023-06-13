// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ForceFree::AnalyticData {

/*!
 * \brief The magnetosphere of an isolated rotating star with dipolar initial
 * magnetic field in the flat spacetime. This is a toy model of a pulsar
 * magnetosphere.
 *
 * \note Coordinate radius of the star is rescaled to 1.0 in code units.
 *
 * The vector potential of the initial magnetic field has the form
 * \cite Most2022
 *
 * \begin{equation}
 *  A_\phi = \frac{A_0 \varpi_0 (x^2+y^2)}{(r^2 + \delta^2)^{3/2}}
 * \end{equation}
 *
 * where $A_0$ is the vector potential amplitude, $\varpi_0$ is a constant with
 * the unit of length, $r^2 = x^2 + y^2 + z^2$, and $\delta$ is a small number
 * for regularization of the dipole magnetic field at the origin ($r=0$).
 *
 * In the Cartesian coordinates, components of densitized magnetic fields are
 * given as
 *
 * \begin{align}
 *  \tilde{B}^x & = A_0 \varpi_0 \frac{3xz}{(r^2 + \delta^2)^{5/2}} , \\
 *  \tilde{B}^y & = A_0 \varpi_0 \frac{3yz}{(r^2 + \delta^2)^{5/2}} , \\
 *  \tilde{B}^z & = A_0 \varpi_0 \frac{3z^2 - r^2 + 2\delta^2}{(r^2 +
 *                    \delta^2)^{5/2}} .
 * \end{align}
 *
 * Rotation of the star is switched on at $t=0$ with the angular velocity
 * specified in the input file. The grid points inside the star ($x^2 + y^2 +
 * z^2 < 1.0$) are identified as the interior and masked by `interior_mask()`
 * member function. In the masked region we impose the MHD condition
 * $\mathbf{E} + \mathbf{v} \times \mathbf{B} = 0$, where the velocity field is
 * given by $\mathbf{v} \equiv \Omega \hat{z} \times \mathbf{r}$. By this means,
 * initial dipolar magnetic field is effectively "anchored" inside the star
 * while electromagnetic fields are evolved self-consistently outside the star.
 *
 * \note We impose the MHD condition stated above and $q=0$ inside the masked
 * interior region during the evolution phase. While this initial data class
 * sets both electric field and charge density to zero at $t=0$, those variables
 * are immediately overwritten with proper values ($\mathbf{E} =
 * -\mathbf{v}\times\mathbf{B}$, $q=0$) once the simulation begins.
 *
 * When the system reaches a stationary state, magnetic field lines far from the
 * star are opening up while the field lines close to the star are corotating.
 * The light cylinder and the Y-point, which marks the boundary between these
 * two regions with different magnetic field topology, are expected to be formed
 * at $r_\text{LC} = c/\Omega$.
 *
 * The option `TiltAngle` controls the angle $\alpha$ between the rotation axis
 * ($z$) and the magnetic axis of initial magnetic field on the $x-z$ plane. An
 * aligned rotator ($\alpha = 0$) is a common test problem for FFE codes,
 * whereas an oblique rotator ($\alpha \neq 0$) is a more realistic model of
 * pulsars \cite Spitkovsky2006.
 *
 */
class RotatingDipole : public evolution::initial_data::InitialData,
                       public MarkAsAnalyticData {
 public:
  struct VectorPotentialAmplitude {
    using type = double;
    static constexpr Options::String help = {
        "The vector potential amplitude A_0"};
  };

  struct Varpi0 {
    using type = double;
    static constexpr Options::String help = {"The length constant varpi_0"};
    static type lower_bound() { return 0.0; }
  };

  struct Delta {
    using type = double;
    static constexpr Options::String help = {
        "A small value used to regularize magnetic fields at r=0."};
    static type lower_bound() { return 0.0; }
  };

  struct AngularVelocity {
    using type = double;
    static constexpr Options::String help = {
        "Rotation angular velocity of the star."};
    static type upper_bound() { return 1.0; }
    static type lower_bound() { return -1.0; }
  };

  struct TiltAngle {
    using type = double;
    static constexpr Options::String help = {
        "Angle between the rotation axis (z) and magnetic axis at t = 0."};
    static type upper_bound() { return M_PI; }
    static type lower_bound() { return 0.0; }
  };

  using options = tmpl::list<VectorPotentialAmplitude, Varpi0, Delta,
                             AngularVelocity, TiltAngle>;
  static constexpr Options::String help{
      "Magnetosphere of an isolated rotating star with dipole magnetic field."};

  RotatingDipole() = default;
  RotatingDipole(const RotatingDipole&) = default;
  RotatingDipole& operator=(const RotatingDipole&) = default;
  RotatingDipole(RotatingDipole&&) = default;
  RotatingDipole& operator=(RotatingDipole&&) = default;
  ~RotatingDipole() override = default;

  RotatingDipole(double vector_potential_amplitude, double varpi0, double delta,
                 double angular_velocity, double tilt_angle,
                 const Options::Context& context = {});

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit RotatingDipole(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(RotatingDipole);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  /// @{
  /// Retrieve the EM variables.
  static auto variables(const tnsr::I<DataVector, 3>& coords,
                        tmpl::list<Tags::TildeE> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildeE>;

  auto variables(const tnsr::I<DataVector, 3>& coords,
                 tmpl::list<Tags::TildeB> /*meta*/) const
      -> tuples::TaggedTuple<Tags::TildeB>;

  static auto variables(const tnsr::I<DataVector, 3>& coords,
                        tmpl::list<Tags::TildePsi> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildePsi>;

  static auto variables(const tnsr::I<DataVector, 3>& coords,
                        tmpl::list<Tags::TildePhi> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildePhi>;

  static auto variables(const tnsr::I<DataVector, 3>& coords,
                        tmpl::list<Tags::TildeQ> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildeQ>;
  /// @}

  /// Retrieve a collection of EM variables at position x
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataVector, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataVector, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const {
    constexpr double dummy_time = 0.0;
    return background_spacetime_.variables(x, dummy_time, tmpl::list<Tag>{});
  }

  // Returns the value of NS interior mask
  static std::optional<Scalar<DataVector>> interior_mask(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x);

  // Returns the value of angular velocity.
  double angular_velocity() const { return angular_velocity_; };

 private:
  double vector_potential_amplitude_ =
      std::numeric_limits<double>::signaling_NaN();
  double varpi0_ = std::numeric_limits<double>::signaling_NaN();
  double delta_ = std::numeric_limits<double>::signaling_NaN();
  double angular_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double tilt_angle_ = std::numeric_limits<double>::signaling_NaN();
  gr::Solutions::Minkowski<3> background_spacetime_{};

  friend bool operator==(const RotatingDipole& lhs, const RotatingDipole& rhs);
};

bool operator!=(const RotatingDipole& lhs, const RotatingDipole& rhs);

}  // namespace ForceFree::AnalyticData
