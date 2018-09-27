// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Utilities/MakeArray.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace NewtonianEuler {
namespace Solutions {

/*!
 * \brief Newtonian isentropic vortex in Cartesian coordinates
 *
 * The analytic solution to the 2-D Newtonian Euler system
 * representing the slow advection of an incompressible, isentropic
 * vortex \ref vortex_ref "[1]". The initial condition is the superposition of a
 * mean uniform flow with a gaussian-profile vortex. When embedded in
 * 3-D space, the isentropic vortex is still a solution to the corresponding 3-D
 * system if the velocity along the third axis is a constant. In Cartesian
 * coordinates \f$(x, y, z)\f$, and using dimensionless units, the primitive
 * quantities at a given time \f$t\f$ are then
 *
 * \f{align*}
 * \rho &= \left[1 - \dfrac{(\gamma - 1)\beta^2}{8\gamma\pi^2}\exp\left(
 * 1 - r^2\right)\right]^{1/(\gamma - 1)}, \\
 * v_x &= U - \dfrac{\beta\tilde y}{2\pi}\exp\left(\dfrac{1 - r^2}{2}\right),\\
 * v_y &= V + \dfrac{\beta\tilde x}{2\pi}\exp\left(\dfrac{1 - r^2}{2}\right),\\
 * v_z &= W,\\
 * \epsilon &= \frac{\rho^{\gamma - 1}}{\gamma - 1},
 * \f}
 *
 * with
 *
 * \f{align*}
 * r^2 &= {\tilde x}^2 + {\tilde y}^2,\\
 * \tilde x &= x - X_0 - U t,\\
 * \tilde y &= y - Y_0 - V t,
 * \f}
 *
 * where \f$(X_0, Y_0)\f$ is the position of the vortex on the \f$(x, y)\f$
 * plane at \f$t = 0\f$, \f$(U, V, W)\f$ are the components of the mean flow
 * velocity, \f$\beta\f$ is the vortex strength, and \f$\gamma\f$ is the
 * adiabatic index. The pressure \f$p\f$ is then obtained from the dimensionless
 * polytropic relation
 *
 * \f{align*}
 * p = \rho^\gamma.
 * \f}
 *
 * On the other hand, if the velocity along the \f$z-\f$axis is not a constant
 * but a function of the \f$z\f$ coordinate, the resulting modified isentropic
 * vortex is still a solution to the Newtonian Euler system, but with source
 * terms that are proportional to \f$dv_z/dz\f$. (See
 * NewtonianEuler::Sources::IsentropicVortexSource.) For testing purposes,
 * we choose to write the velocity as a uniform field plus a periodic
 * perturbation,
 *
 * \f{align*}
 * v_z(z) = W + \epsilon \sin{z},
 * \f}
 *
 * where \f$\epsilon\f$ is the amplitude of the perturbation. The resulting
 * source for the Newtonian Euler system will then be proportional to
 * \f$\epsilon \cos{z}\f$.
 *
 * \anchor vortex_ref [1] H.C Yee, N.D Sandham, M.J Djomehri, Low-dissipative
 * high-order shock-capturing methods using characteristic-based filters, J.
 * Comput. Phys. [150 (1999) 199](http://dx.doi.org/10.1006/jcph.1998.6177)
 */
class IsentropicVortex {
 public:
  /// The adiabatic index of the fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr OptionString help = {"The adiabatic index of the fluid."};
    // Note: bounds only valid for an ideal gas.
    static type lower_bound() { return 1.0; }
    static type upper_bound() { return 2.0; }
  };

  /// The position of the center of the vortex at \f$t = 0\f$
  struct Center {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {
        "The coordinates of the center of the vortex at t = 0."};
    static type default_value() { return {{0.0, 0.0, 0.0}}; }
  };

  /// The mean flow velocity.
  struct MeanVelocity {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {"The mean flow velocity."};
  };

  /// The amplitude of the perturbation generating a source term.
  struct PerturbAmplitude {
    using type = double;
    static constexpr OptionString help = {
        "The amplitude of the perturbation producing sources."};
  };

  /// The strength of the vortex.
  struct Strength {
    using type = double;
    static constexpr OptionString help = {"The strength of the vortex."};
    static type lower_bound() { return 0.0; }
  };

  using options = tmpl::list<AdiabaticIndex, Center, MeanVelocity,
                             PerturbAmplitude, Strength>;
  static constexpr OptionString help = {"Newtonian Isentropic Vortex."};

  IsentropicVortex() = default;
  IsentropicVortex(const IsentropicVortex& /*rhs*/) = delete;
  IsentropicVortex& operator=(const IsentropicVortex& /*rhs*/) = delete;
  IsentropicVortex(IsentropicVortex&& /*rhs*/) noexcept = default;
  IsentropicVortex& operator=(IsentropicVortex&& /*rhs*/) noexcept = default;
  ~IsentropicVortex() = default;

  IsentropicVortex(double adiabatic_index, Center::type center,
                   MeanVelocity::type mean_velocity,
                   double perturbation_amplitude, double strength);

  explicit IsentropicVortex(CkMigrateMessage* /*unused*/) noexcept {}

  template <typename DataType>
  using primitive_t =
      tmpl::list<Tags::MassDensity<DataType>, Tags::Velocity<DataType, 3>,
                 Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  using conservative_t = tmpl::list<Tags::MassDensity<DataType>,
                                    Tags::MomentumDensity<DataType, 3>,
                                    Tags::EnergyDensity<DataType>>;

  template <typename DataType>
  Scalar<DataType> perturbation(const DataType& coord_z) const noexcept;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<primitive_t<DataType>> primitive_variables(
      const tnsr::I<DataType, 3>& x, double t) const noexcept;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<conservative_t<DataType>>
  conservative_variables(const tnsr::I<DataType, 3>& x, double t) const
      noexcept;

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

  constexpr double adiabatic_index() const noexcept { return adiabatic_index_; }
  constexpr const Center::type& center() const noexcept { return center_; }
  constexpr const MeanVelocity::type& mean_velocity() const noexcept {
    return mean_velocity_;
  }
  constexpr double perturbation_amplitude() const noexcept {
    return perturbation_amplitude_;
  }
  constexpr double strength() const noexcept { return strength_; }

 private:
  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  Center::type center_ = {{0.0, 0.0, 0.0}};
  MeanVelocity::type mean_velocity_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  double perturbation_amplitude_ = std::numeric_limits<double>::signaling_NaN();
  double strength_ = std::numeric_limits<double>::signaling_NaN();
};

inline constexpr bool operator==(const IsentropicVortex& lhs,
                                 const IsentropicVortex& rhs) noexcept {
  return lhs.adiabatic_index() == rhs.adiabatic_index() and
         lhs.center() == rhs.center() and
         lhs.mean_velocity() == rhs.mean_velocity() and
         lhs.perturbation_amplitude() == rhs.perturbation_amplitude() and
         lhs.strength() == rhs.strength();
}

inline constexpr bool operator!=(const IsentropicVortex& lhs,
                                 const IsentropicVortex& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Solutions
}  // namespace NewtonianEuler
