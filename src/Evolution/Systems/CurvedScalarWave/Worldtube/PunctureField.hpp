// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Context.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace CurvedScalarWave::Worldtube {
/*!
 * \brief Dispatcher to compute the puncture/singular field for a scalar charge
 * on a generic orbit in Schwarzschild or Kerr spacetime. The Kerr puncture
 * reduces to Schwarzsschild for zero spin but is faster to evaluate.
 */
class PunctureField {
 public:
  enum class Type { Schwarzschild, Kerr };

  /*!
   * \brief Use the Schwarzschild puncture field model.
   */
  struct Schwarzschild {
    using type = Schwarzschild;
    static constexpr Options::String help = {
        "Use the Schwarzschild puncture field model."};

    /*!
     * \brief Puncture field expansion order. Currently orders 0 and 1 are
     * implemented.
     */
    struct ExpansionOrder {
      using type = size_t;
      static constexpr Options::String help{
          "Puncture field expansion order. Currently orders 0 and 1 are "
          "implemented."};
      static size_t upper_bound() { return 1; }
    };

    /*!
     * \brief The mass of the central black hole.
     */
    struct BlackHoleMass {
      using type = double;
      static constexpr Options::String help{
          "The mass of the central black hole."};
      static double lower_bound() { return 0.; }
    };

    using options = tmpl::list<ExpansionOrder, BlackHoleMass>;

    Schwarzschild() = default;
    Schwarzschild(size_t expansion_order_in, double black_hole_mass_in,
                  const Options::Context& context = {});

    size_t expansion_order{};
    double black_hole_mass{};
  };

  /*!
   * \brief Use the Kerr puncture field model. This option is currently parsed
   * but not yet implemented at runtime.
   */
  struct Kerr {
    using type = Kerr;
    static constexpr Options::String help = {
        "Use the Kerr puncture field model. This option is currently parsed "
        "but not yet implemented at runtime."};

    /*!
     * \brief Puncture field expansion order. Currently orders 0 and 1 are
     * implemented.
     */
    struct ExpansionOrder {
      using type = size_t;
      static constexpr Options::String help{
          "Puncture field expansion order. Currently orders 0 and 1 are "
          "implemented."};
      static size_t upper_bound() { return 1; }
    };

    /*!
     * \brief The mass of the central black hole.
     */
    struct BlackHoleMass {
      using type = double;
      static constexpr Options::String help{
          "The mass of the central black hole."};
      static double lower_bound() { return 0.; }
    };

    /*!
     * \brief The dimensionless z-component of the black-hole spin.
     */
    struct SpinAlongZAxis {
      using type = double;
      static constexpr Options::String help{
          "The dimensionless z-component of the black-hole spin."};
    };

    using options = tmpl::list<ExpansionOrder, BlackHoleMass, SpinAlongZAxis>;

    Kerr() = default;
    Kerr(size_t expansion_order_in, double black_hole_mass_in,
         double spin_along_z_axis_in, const Options::Context& context = {});

    size_t expansion_order{};
    double black_hole_mass{};
    double spin_along_z_axis{};
  };

  using options = tmpl::list<
      Options::Alternatives<tmpl::list<Schwarzschild>, tmpl::list<Kerr>>>;

  static constexpr Options::String help = {
      "Configuration and dispatcher for puncture-field expressions. Choose "
      "either Schwarzschild or Kerr."};

  PunctureField() = default;
  explicit PunctureField(const Schwarzschild& schwarzschild,
                         const Options::Context& context = {});
  explicit PunctureField(const Kerr& kerr,
                         const Options::Context& context = {});

  void pup(PUP::er& p);

  Type type() const;
  size_t expansion_order() const;
  double black_hole_mass() const;
  double spin_along_z_axis() const;

  /*!
   * \brief Compute and write the puncture field and its derivatives for the
   * configured puncture model and expansion order.
   */
  void apply_puncture(
      gsl::not_null<Variables<tmpl::list<
          CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
          ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                        Frame::Inertial>>>*>
          result,
      const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
      const tnsr::I<double, 3>& particle_position,
      const tnsr::I<double, 3>& particle_velocity,
      const tnsr::I<double, 3>& particle_acceleration) const;

  /*!
   * \brief Compute and write the corrections to the
   * puncture field for the configured puncture model and expansion order. These
   * terms arise at non-geodesic accelerations such as the self-force.
   */
  void apply_acceleration_terms(
      gsl::not_null<Variables<tmpl::list<
          CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
          ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                        Frame::Inertial>>>*>
          result,
      const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
      const tnsr::I<double, 3>& particle_position,
      const tnsr::I<double, 3>& particle_velocity,
      const tnsr::I<double, 3>& particle_acceleration, double ft, double fx,
      double fy, double dt_ft, double dt_fx, double dt_fy, double Du_ft,
      double Du_fx, double Du_fy, double dt_Du_ft, double dt_Du_fx,
      double dt_Du_fy) const;

 private:
  Type type_{Type::Schwarzschild};
  size_t expansion_order_{0};
  double black_hole_mass_{1.};
  double spin_along_z_axis_{0.};
};
/*!
 * \brief Computes the puncture/singular field \f$\Psi^\mathcal{P}\f$ of a
 * scalar charge on a generic orbit in Schwarzschild spacetime.
 * described in \cite Detweiler2003.
 *
 * \details The field is computed using a Detweiler-Whiting singular
 * Green's function and perturbatively expanded in the geodesic distance from
 * the particle. It solves the inhomogeneous wave equation
 *
 * \f{align*}{
 * \Box \Psi^\mathcal{P} = -4 \pi q \int \sqrt{-g} \delta^4(x^i, z(\tau)) d \tau
 * \f}
 *
 * where \f$q\f$ is the scalar charge and \f$z(\tau)\f$ is the worldline of the
 * particle. The expression is expanded up to a certain order in geodesic
 * distance and transformed to Kerr-Schild coordinates.
 *
 * The function given here assumes that the particle has scalar charge \f$q=1\f$
 * and is on a fixed geodesic orbit. It returns the
 * singular field at the requested coordinates as well as its time and spatial
 * derivative. For non-geodesic orbits, corresponding acceleration terms have to
 * be added to the puncture field.
 *
 * \note The expressions were computed with Mathematica and optimized by
 * applying common subexpression elimination with sympy. The memory allocations
 * of temporaries were optimized manually.
 */
void puncture_field_0(
    gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
    const tnsr::I<double, 3>& particle_position,
    const tnsr::I<double, 3>& particle_velocity,
    const tnsr::I<double, 3>& particle_acceleration, double bh_mass);

/*!
 * \brief Computes the puncture/singular field \f$\Psi^\mathcal{P}\f$ of a
 * scalar charge on a generic orbit in Schwarzschild spacetime.
 * described in \cite Detweiler2003.
 *
 * \details For non-geodesic orbits, there are additional contributions, see
 * `acceleration_terms_0`.
 */
void puncture_field_1(
    gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
    const tnsr::I<double, 3>& particle_position,
    const tnsr::I<double, 3>& particle_velocity,
    const tnsr::I<double, 3>& particle_acceleration, double bh_mass);

/*!
 * \brief Computes the acceleration terms of a puncture/singular field
 * \f$\Psi^\mathcal{P}\f$ of a scalar charge on a generic orbit in Schwarzschild
 * spacetime up to zeroth order in coordinate distance.
 * \details The appropriate expression can be found in Eq. (37) of
 * \cite Wittek:2024gxn. The values ft, fx, fy are the time, x and y component
 * of the self force per unit mass evaluated at the position of the particle;
 * dt_ft, dt_fx, dt_fy are the respective total time derivatives. The code in
 * this function was auto-generated by generating the full expressions with
 * Mathematica and employing common subexpression elimination with sympy. The
 * mathematica file and generating script can be found at
 * https://github.com/nikwit/puncture-field.
 */
void acceleration_terms_0(
    gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
    const tnsr::I<double, 3>& particle_position,
    const tnsr::I<double, 3>& particle_velocity,
    const tnsr::I<double, 3>& particle_acceleration, double ft, double fx,
    double fy, double dt_ft, double dt_fx, double dt_fy, double bh_mass);

/*!
 * \brief Computes the acceleration terms of a puncture/singular field
 * \f$\Psi^\mathcal{P}\f$ of a scalar charge on a generic orbit in Schwarzschild
 * spacetime up to first order in coordinate distance (i.e. zeroth and first
 * order).
 * \details The appropriate expression can be found in Eq. (37) of
 * \cite Wittek:2024gxn. The values ft, fx, fy are the time, x and y component
 * of the self force per unit mass evaluated at the position of the particle;
 * dt_ft, dt_fx, dt_fy are the respective total time derivatives. The code in
 * this function was auto-generated by generating the full expressions with
 * Mathematica and employing common subexpression elimination with sympy. The
 * mathematica file and generating script can be found at
 * https://github.com/nikwit/puncture-field.
 *
 */
void acceleration_terms_1(
    gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
    const tnsr::I<double, 3>& particle_position,
    const tnsr::I<double, 3>& particle_velocity,
    const tnsr::I<double, 3>& particle_acceleration, double ft, double fx,
    double fy, double dt_ft, double dt_fx, double dt_fy, double Du_ft,
    double Du_fx, double Du_fy, double dt_Du_ft, double dt_Du_fx,
    double dt_Du_fy, double bh_mass);
}  // namespace CurvedScalarWave::Worldtube
