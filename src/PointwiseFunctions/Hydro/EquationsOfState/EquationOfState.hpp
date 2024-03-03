// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/list/for_each.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/enum.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace EquationsOfState {
template <typename ColdEquilEos>
class Barotropic3D;
template <bool IsRelativistic>
class DarkEnergyFluid;
template <typename EquilEos>
class Equilibrium3D;
template <typename ColdEquationOfState>
class HybridEos;
template <bool IsRelativistic>
class IdealFluid;
template <bool IsRelativistic>
class PolytropicFluid;
template <bool IsRelativistic>
class PiecewisePolytropicFluid;
class Spectral;
template <typename LowDensityEoS>
class Enthalpy;
template <bool IsRelativistic>
class Tabulated3D;
}  // namespace EquationsOfState
/// \endcond

/// Contains all equations of state, including base class
namespace EquationsOfState {

namespace detail {
template <bool IsRelativistic, size_t ThermodynamicDim>
struct DerivedClasses {};

template <>
struct DerivedClasses<true, 1> {
  using type = tmpl::list<
      Enthalpy<Enthalpy<Enthalpy<Spectral>>>, Enthalpy<Enthalpy<Spectral>>,
      Enthalpy<Spectral>, Enthalpy<PolytropicFluid<true>>,
      PiecewisePolytropicFluid<true>, PolytropicFluid<true>, Spectral>;
};

template <>
struct DerivedClasses<false, 1> {
  using type =
      tmpl::list<PiecewisePolytropicFluid<false>, PolytropicFluid<false>>;
};

template <>
struct DerivedClasses<true, 2> {
  using type = tmpl::list<DarkEnergyFluid<true>, IdealFluid<true>,
                          HybridEos<PolytropicFluid<true>>, HybridEos<Spectral>,
                          HybridEos<Enthalpy<Spectral>>>;
};

template <>
struct DerivedClasses<false, 2> {
  using type = tmpl::list<IdealFluid<false>, HybridEos<PolytropicFluid<false>>>;
};

template <>
struct DerivedClasses<true, 3> {
  using type =
      tmpl::list<Tabulated3D<true>, Barotropic3D<PolytropicFluid<true>>,
                 Barotropic3D<Spectral>, Barotropic3D<Enthalpy<Spectral>>,
                 Equilibrium3D<HybridEos<PolytropicFluid<true>>>,
                 Equilibrium3D<HybridEos<Spectral>>,
                 Equilibrium3D<HybridEos<Enthalpy<Spectral>>>,
                 Equilibrium3D<DarkEnergyFluid<true>>,
                 Equilibrium3D<IdealFluid<true>>,
                 Barotropic3D<PiecewisePolytropicFluid<true>>,
                 Barotropic3D<Enthalpy<Enthalpy<Spectral>>>,
                 Barotropic3D<Enthalpy<Enthalpy<Enthalpy<Spectral>>>>>;
};

template <>
struct DerivedClasses<false, 3> {
  using type = tmpl::list<Tabulated3D<false>, Equilibrium3D<IdealFluid<false>>,
                          Barotropic3D<PiecewisePolytropicFluid<false>>,
                          Equilibrium3D<HybridEos<PolytropicFluid<false>>>,
                          Barotropic3D<PolytropicFluid<false>>>;
};

}  // namespace detail

/*!
 * \ingroup EquationsOfStateGroup
 * \brief Base class for equations of state depending on whether or not the
 * system is relativistic, and the number of independent thermodynamic variables
 * (`ThermodynamicDim`) needed to determine the pressure.
 *
 * The template parameter `IsRelativistic` is `true` for relativistic equations
 * of state and `false` for non-relativistic equations of state.
 */
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;

template <typename T>
struct get_eos_base_impl {
  using type = EquationsOfState::EquationOfState<T::is_relativistic,
                                                 T::thermodynamic_dim>;
};

template <bool IsRelativistic, size_t ThermodynamicDim>
struct get_eos_base_impl<
    EquationsOfState::EquationOfState<IsRelativistic, ThermodynamicDim>> {
  using type =
      EquationsOfState::EquationOfState<IsRelativistic, ThermodynamicDim>;
};

template <typename T>
using get_eos_base = typename get_eos_base_impl<T>::type;

/*!
 * \ingroup EquationsOfStateGroup
 * \brief Base class for equations of state which need one thermodynamic
 * variable in order to determine the pressure.
 *
 * The template parameter `IsRelativistic` is `true` for relativistic equations
 * of state and `false` for non-relativistic equations of state.
 */
template <bool IsRelativistic>
class EquationOfState<IsRelativistic, 1> : public PUP::able {
 public:
  static constexpr bool is_relativistic = IsRelativistic;
  static constexpr size_t thermodynamic_dim = 1;
  using creatable_classes =
      typename detail::DerivedClasses<IsRelativistic, 1>::type;

  EquationOfState() = default;
  EquationOfState(const EquationOfState&) = default;
  EquationOfState& operator=(const EquationOfState&) = default;
  EquationOfState(EquationOfState&&) = default;
  EquationOfState& operator=(EquationOfState&&) = default;
  ~EquationOfState() override = default;

  explicit EquationOfState(CkMigrateMessage* msg) : PUP::able(msg) {}

  WRAPPED_PUPable_abstract(EquationOfState);  // NOLINT

  virtual std::unique_ptr<EquationOfState<IsRelativistic, 1>> get_clone()
      const = 0;

  virtual bool is_equal(
      const EquationOfState<IsRelativistic, 1>& rhs) const = 0;
  virtual std::unique_ptr<EquationOfState<IsRelativistic, 3>>
  promote_to_3d_eos() const = 0;
  /// \brief Returns `true` if the EOS is barotropic
  bool is_barotropic() const { return true; }
  /// @{
  /*!
   * Computes the electron fraction in beta-equilibrium \f$Y_e^{\rm eq}\f$ from
   * the rest mass density \f$\rho\f$.
   */
  virtual Scalar<double> equilibrium_electron_fraction_from_density_temperature(
      const Scalar<double>& rest_mass_density,
      const Scalar<double>& /*temperature*/) const {
    return make_with_value<Scalar<double>>(rest_mass_density, 0.1);
  }

  virtual Scalar<DataVector>
  equilibrium_electron_fraction_from_density_temperature(
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& /*temperature*/) const {
    return make_with_value<Scalar<DataVector>>(rest_mass_density, 0.1);
  }
  /// @}

  /// @{
  /*!
   * Computes the pressure \f$p\f$ from the rest mass density \f$\rho\f$.
   */
  virtual Scalar<double> pressure_from_density(
      const Scalar<double>& /*rest_mass_density*/) const = 0;
  virtual Scalar<DataVector> pressure_from_density(
      const Scalar<DataVector>& /*rest_mass_density*/) const = 0;
  /// @}

  /// @{
  /*!
   * Computes the rest mass density \f$\rho\f$ from the specific enthalpy
   * \f$h\f$.
   */
  virtual Scalar<double> rest_mass_density_from_enthalpy(
      const Scalar<double>& /*specific_enthalpy*/) const = 0;
  virtual Scalar<DataVector> rest_mass_density_from_enthalpy(
      const Scalar<DataVector>& /*specific_enthalpy*/) const = 0;
  /// @}

  /// @{
  /*!
   * Computes the specific internal energy \f$\epsilon\f$ from the rest mass
   * density \f$\rho\f$.
   */
  virtual Scalar<double> specific_internal_energy_from_density(
      const Scalar<double>& /*rest_mass_density*/) const = 0;
  virtual Scalar<DataVector> specific_internal_energy_from_density(
      const Scalar<DataVector>& /*rest_mass_density*/) const = 0;
  /// @}

  /// @{
  /*!
   * Computes the temperature \f$T\f$ from the rest mass
   * density \f$\rho\f$.
   */
  virtual Scalar<double> temperature_from_density(
      const Scalar<double>& /*rest_mass_density*/) const {
    return Scalar<double>{0.0};
  }
  virtual Scalar<DataVector> temperature_from_density(
      const Scalar<DataVector>& rest_mass_density) const {
    return make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
  }
  /// @}

  /// @{
  /*!
   * Computes the temperature \f$\T\f$ from the specific internal energy
   * \f$\epsilon\f$.
   */
  virtual Scalar<double> temperature_from_specific_internal_energy(
      const Scalar<double>& /*specific_internal_energy*/) const {
    return Scalar<double>{0.0};
  }
  virtual Scalar<DataVector> temperature_from_specific_internal_energy(
      const Scalar<DataVector>& specific_internal_energy) const {
    return make_with_value<Scalar<DataVector>>(specific_internal_energy, 0.0);
  }
  /// @}

  /// @{
  /*!
   * Computes \f$\chi=\partial p / \partial \rho\f$ from \f$\rho\f$, where
   * \f$p\f$ is the pressure and \f$\rho\f$ is the rest mass density.
   */
  virtual Scalar<double> chi_from_density(
      const Scalar<double>& /*rest_mass_density*/) const = 0;
  virtual Scalar<DataVector> chi_from_density(
      const Scalar<DataVector>& /*rest_mass_density*/) const = 0;
  /// @}

  /// @{
  /*!
   * Computes \f$\kappa p/\rho^2=(p/\rho^2)\partial p / \partial \epsilon\f$
   * from \f$\rho\f$, where \f$p\f$ is the pressure, \f$\rho\f$ is the rest mass
   * density, and \f$\epsilon\f$ is the specific internal energy.
   *
   * The reason for not returning just
   * \f$\kappa=\partial p / \partial \epsilon\f$ is to avoid division by zero
   * for small values of \f$\rho\f$ when assembling the speed of sound with
   * some equations of state.
   */
  virtual Scalar<double> kappa_times_p_over_rho_squared_from_density(
      const Scalar<double>& /*rest_mass_density*/) const = 0;
  virtual Scalar<DataVector> kappa_times_p_over_rho_squared_from_density(
      const Scalar<DataVector>& /*rest_mass_density*/) const = 0;

  /// The lower bound of the electron fraction that is valid for this EOS
  virtual double electron_fraction_lower_bound() const { return 0.0; }

  /// The upper bound of the electron fraction that is valid for this EOS
  virtual double electron_fraction_upper_bound() const { return 1.0; }

  /// The lower bound of the rest mass density that is valid for this EOS
  virtual double rest_mass_density_lower_bound() const = 0;

  /// The upper bound of the rest mass density that is valid for this EOS
  virtual double rest_mass_density_upper_bound() const = 0;

  /// The lower bound of the temperature that is valid for this EOS
  virtual double temperature_lower_bound() const { return 0.0; };

  /// The upper bound of the temperature that is valid for this EOS
  virtual double temperature_upper_bound() const {
    return std::numeric_limits<double>::max();
  };
  /// The lower bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  virtual double specific_internal_energy_lower_bound(
      double rest_mass_density) const = 0;

  /// The upper bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  virtual double specific_internal_energy_upper_bound(
      double rest_mass_density) const = 0;

  /// The lower bound of the specific enthalpy that is valid for this EOS
  virtual double specific_enthalpy_lower_bound() const = 0;

  /// The vacuum mass of a baryon for this EOS
  virtual double baryon_mass() const {
    return hydro::units::geometric::default_baryon_mass;
  }
};

/*!
 * \ingroup EquationsOfStateGroup
 * \brief Base class for equations of state which need two independent
 * thermodynamic variables in order to determine the pressure.
 *
 * The template parameter `IsRelativistic` is `true` for relativistic equations
 * of state and `false` for non-relativistic equations of state.
 */
template <bool IsRelativistic>
class EquationOfState<IsRelativistic, 2> : public PUP::able {
 public:
  static constexpr bool is_relativistic = IsRelativistic;
  static constexpr size_t thermodynamic_dim = 2;
  using creatable_classes =
      typename detail::DerivedClasses<IsRelativistic, 2>::type;

  EquationOfState() = default;
  EquationOfState(const EquationOfState&) = default;
  EquationOfState& operator=(const EquationOfState&) = default;
  EquationOfState(EquationOfState&&) = default;
  EquationOfState& operator=(EquationOfState&&) = default;
  ~EquationOfState() override = default;

  explicit EquationOfState(CkMigrateMessage* msg) : PUP::able(msg) {}

  WRAPPED_PUPable_abstract(EquationOfState);  // NOLINT

  virtual inline std::unique_ptr<EquationOfState<IsRelativistic, 2>> get_clone()
      const = 0;

  virtual bool is_equal(
      const EquationOfState<IsRelativistic, 2>& rhs) const = 0;

  virtual std::unique_ptr<EquationOfState<IsRelativistic, 3>>
  promote_to_3d_eos() const = 0;

  /// \brief Returns `true` if the EOS is barotropic
  virtual bool is_barotropic() const = 0;

  /// @{
  /*!
   * Computes the electron fraction in beta-equilibrium \f$Y_e^{\rm eq}\f$ from
   * the rest mass density \f$\rho\f$ and the temperature \f$T\f$.
   */
  virtual Scalar<double> equilibrium_electron_fraction_from_density_temperature(
      const Scalar<double>& rest_mass_density,
      const Scalar<double>& /*temperature*/) const {
    return make_with_value<Scalar<double>>(rest_mass_density, 0.1);
  }

  virtual Scalar<DataVector>
  equilibrium_electron_fraction_from_density_temperature(
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& /*temperature*/) const {
    return make_with_value<Scalar<DataVector>>(rest_mass_density, 0.1);
  }
  /// @}

  /// @{
  /*!
   * Computes the pressure \f$p\f$ from the rest mass density \f$\rho\f$ and the
   * specific internal energy \f$\epsilon\f$.
   */
  virtual Scalar<double> pressure_from_density_and_energy(
      const Scalar<double>& /*rest_mass_density*/,
      const Scalar<double>& /*specific_internal_energy*/) const = 0;
  virtual Scalar<DataVector> pressure_from_density_and_energy(
      const Scalar<DataVector>& /*rest_mass_density*/,
      const Scalar<DataVector>& /*specific_internal_energy*/) const = 0;
  /// @}

  /// @{
  /*!
   * Computes the pressure \f$p\f$ from the rest mass density \f$\rho\f$ and the
   * specific enthalpy \f$h\f$.
   */
  virtual Scalar<double> pressure_from_density_and_enthalpy(
      const Scalar<double>& /*rest_mass_density*/,
      const Scalar<double>& /*specific_enthalpy*/) const = 0;
  virtual Scalar<DataVector> pressure_from_density_and_enthalpy(
      const Scalar<DataVector>& /*rest_mass_density*/,
      const Scalar<DataVector>& /*specific_enthalpy*/) const = 0;
  /// @}

  /// @{
  /*!
   * Computes the specific internal energy \f$\epsilon\f$ from the rest mass
   * density \f$\rho\f$ and the pressure \f$p\f$.
   */
  virtual Scalar<double> specific_internal_energy_from_density_and_pressure(
      const Scalar<double>& /*rest_mass_density*/,
      const Scalar<double>& /*pressure*/) const = 0;
  virtual Scalar<DataVector> specific_internal_energy_from_density_and_pressure(
      const Scalar<DataVector>& /*rest_mass_density*/,
      const Scalar<DataVector>& /*pressure*/) const = 0;
  /// @}

  /// @{
  /*!
   * Computes the temperature \f$T\f$ from the rest mass
   * density \f$\rho\f$ and the specific internal energy \f$\epsilon\f$.
   */
  virtual Scalar<double> temperature_from_density_and_energy(
      const Scalar<double>& /*rest_mass_density*/,
      const Scalar<double>& /*specific_internal_energy*/) const = 0;
  virtual Scalar<DataVector> temperature_from_density_and_energy(
      const Scalar<DataVector>& /*rest_mass_density*/,
      const Scalar<DataVector>& /*specific_internal_energy*/) const = 0;
  /// @}

  /// @{
  /*!
   * Computes the specific internal energy \f$\epsilon\f$ from the rest mass
   * density \f$\rho\f$ and the temperature \f$T\f$.
   */
  virtual Scalar<double> specific_internal_energy_from_density_and_temperature(
      const Scalar<double>& /*rest_mass_density*/,
      const Scalar<double>& /*temperature*/) const = 0;
  virtual Scalar<DataVector>
  specific_internal_energy_from_density_and_temperature(
      const Scalar<DataVector>& /*rest_mass_density*/,
      const Scalar<DataVector>& /*temperature*/) const = 0;
  /// @}

  /// @{
  /*!
   * Computes \f$\chi=\partial p / \partial \rho |_{\epsilon}\f$ from the
   * \f$\rho\f$ and \f$\epsilon\f$, where \f$p\f$ is the pressure, \f$\rho\f$ is
   * the rest mass density, and \f$\epsilon\f$ is the specific internal energy.
   */
  virtual Scalar<double> chi_from_density_and_energy(
      const Scalar<double>& /*rest_mass_density*/,
      const Scalar<double>& /*specific_internal_energy*/) const = 0;
  virtual Scalar<DataVector> chi_from_density_and_energy(
      const Scalar<DataVector>& /*rest_mass_density*/,
      const Scalar<DataVector>& /*specific_internal_energy*/) const = 0;
  /// @}

  /// @{
  /*!
   * Computes \f$\kappa p/\rho^2=(p/\rho^2)\partial p / \partial \epsilon
   * |_{\rho}\f$ from \f$\rho\f$ and \f$\epsilon\f$, where \f$p\f$ is the
   * pressure, \f$\rho\f$ is the rest mass density, and \f$\epsilon\f$ is the
   * specific internal energy.
   *
   * The reason for not returning just
   * \f$\kappa=\partial p / \partial \epsilon\f$ is to avoid division by zero
   * for small values of \f$\rho\f$ when assembling the speed of sound with
   * some equations of state.
   */
  virtual Scalar<double> kappa_times_p_over_rho_squared_from_density_and_energy(
      const Scalar<double>& /*rest_mass_density*/,
      const Scalar<double>& /*specific_internal_energy*/) const = 0;
  virtual Scalar<DataVector>
  kappa_times_p_over_rho_squared_from_density_and_energy(
      const Scalar<DataVector>& /*rest_mass_density*/,
      const Scalar<DataVector>& /*specific_internal_energy*/) const = 0;
  /// @}

  /// The lower bound of the electron fraction that is valid for this EOS
  virtual double electron_fraction_lower_bound() const { return 0.0; }

  /// The upper bound of the electron fraction that is valid for this EOS
  virtual double electron_fraction_upper_bound() const { return 1.0; }

  /// The lower bound of the rest mass density that is valid for this EOS
  virtual double rest_mass_density_lower_bound() const = 0;

  /// The upper bound of the rest mass density that is valid for this EOS
  virtual double rest_mass_density_upper_bound() const = 0;

  /// The lower bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  virtual double specific_internal_energy_lower_bound(
      const double rest_mass_density) const = 0;

  /// The upper bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  virtual double specific_internal_energy_upper_bound(
      const double rest_mass_density) const = 0;

  /// The lower bound of the temperature that is valid for this EOS
  virtual double temperature_lower_bound() const { return 0.0; };

  /// The upper bound of the temperature that is valid for this EOS
  virtual double temperature_upper_bound() const {
    return std::numeric_limits<double>::max();
  };

  /// The lower bound of the specific enthalpy that is valid for this EOS
  virtual double specific_enthalpy_lower_bound() const = 0;

  /// The vacuum mass of a baryon for this EOS
  virtual double baryon_mass() const {
    return hydro::units::geometric::default_baryon_mass;
  }
};

/*!
 * \ingroup EquationsOfStateGroup
 * \brief Base class for equations of state which need three independent
 * thermodynamic variables in order to determine the pressure.
 *
 * The template parameter `IsRelativistic` is `true` for relativistic equations
 * of state and `false` for non-relativistic equations of state.
 */
template <bool IsRelativistic>
class EquationOfState<IsRelativistic, 3> : public PUP::able {
 public:
  static constexpr bool is_relativistic = IsRelativistic;
  static constexpr size_t thermodynamic_dim = 3;
  using creatable_classes =
      typename detail::DerivedClasses<IsRelativistic, 3>::type;

  EquationOfState() = default;
  EquationOfState(const EquationOfState&) = default;
  EquationOfState& operator=(const EquationOfState&) = default;
  EquationOfState(EquationOfState&&) = default;
  EquationOfState& operator=(EquationOfState&&) = default;
  ~EquationOfState() override = default;

  explicit EquationOfState(CkMigrateMessage* msg) : PUP::able(msg) {}

  WRAPPED_PUPable_abstract(EquationOfState);  // NOLINT

  virtual inline std::unique_ptr<EquationOfState<IsRelativistic, 3>> get_clone()
      const = 0;

  virtual bool is_equal(
      const EquationOfState<IsRelativistic, 3>& rhs) const = 0;

  virtual std::unique_ptr<EquationOfState<IsRelativistic, 3>>
  promote_to_3d_eos() {
    return this->get_clone();
  }

  /// \brief Returns `true` if the EOS is barotropic
  virtual bool is_barotropic() const = 0;

  /// @{
  /*!
   * Computes the pressure \f$p\f$ from the rest mass density \f$\rho\f$, the
   * specific internal energy \f$\epsilon\f$ and electron fraction \f$Y_e\f$.
   */
  virtual Scalar<double> pressure_from_density_and_energy(
      const Scalar<double>& /*rest_mass_density*/,
      const Scalar<double>& /*specific_internal_energy*/,
      const Scalar<double>& /*electron_fraction*/) const = 0;
  virtual Scalar<DataVector> pressure_from_density_and_energy(
      const Scalar<DataVector>& /*rest_mass_density*/,
      const Scalar<DataVector>& /*specific_internal_energy*/,
      const Scalar<DataVector>& /*electron_fraction*/) const = 0;
  /// @}

  /// @{
  /*!
   * Computes the pressure \f$p\f$ from the rest mass density \f$\rho\f$, the
   * temperature \f$T\f$, and electron fraction \f$Y_e\f$.
   */
  virtual Scalar<double> pressure_from_density_and_temperature(
      const Scalar<double>& /*rest_mass_density*/,
      const Scalar<double>& /*temperature*/,
      const Scalar<double>& /*electron_fraction*/) const = 0;
  virtual Scalar<DataVector> pressure_from_density_and_temperature(
      const Scalar<DataVector>& /*rest_mass_density*/,
      const Scalar<DataVector>& /*temperature*/,
      const Scalar<DataVector>& /*electron_fraction*/) const = 0;
  /// @}

  /// @{
  /*!
   * Computes the temperature \f$T\f$ from the rest mass
   * density \f$\rho\f$, the specific internal energy \f$\epsilon\f$,
   * and electron fraction \f$Y_e\f$.
   */
  virtual Scalar<double> temperature_from_density_and_energy(
      const Scalar<double>& /*rest_mass_density*/,
      const Scalar<double>& /*specific_internal_energy*/,
      const Scalar<double>& /*electron_fraction*/) const = 0;
  virtual Scalar<DataVector> temperature_from_density_and_energy(
      const Scalar<DataVector>& /*rest_mass_density*/,
      const Scalar<DataVector>& /*specific_internal_energy*/,
      const Scalar<DataVector>& /*electron_fraction*/) const = 0;
  /// @}

  /// @{
  /*!
   * Computes the specific internal energy \f$\epsilon\f$ from the rest mass
   * density \f$\rho\f$, the temperature \f$T\f$, and electron fraction
   * \f$Y_e\f$.
   */
  virtual Scalar<double> specific_internal_energy_from_density_and_temperature(
      const Scalar<double>& /*rest_mass_density*/,
      const Scalar<double>& /*temperature*/,
      const Scalar<double>& /*electron_fraction*/
  ) const = 0;
  virtual Scalar<DataVector>
  specific_internal_energy_from_density_and_temperature(
      const Scalar<DataVector>& /*rest_mass_density*/,
      const Scalar<DataVector>& /*temperature*/,
      const Scalar<DataVector>& /*electron_fraction*/
  ) const = 0;
  /// @}

  /// @{
  /*!
   * Computes adiabatic sound speed squared
   * \f[
   * c_s^2  \equiv \frac{\partial p}{\partial e} |_{s, Y_e} =
   * \frac{\rho}{h}\frac{\partial p}{\rho} |_{e, Y_e} +
   * \frac{\partial p}{\partial e}|_{\rho, Y_e}
   * \f].
   * With \f$p, e\f$ the pressure and energy density respectively,
   * \f$s\f$ the entropy density, \f$Y_e\f$ the electron fraction
   * and \f$\rho\f$ the rest-mass density.
   * Computed as a function of temperature, rest-mass density and electron
   * fraction. Note that this will break thermodynamic consistency if the
   * pressure and internal energy interpolated separately. The precise impact of
   * this will depend on the EoS and numerical scheme used for the evolution.
   */
  virtual Scalar<double> sound_speed_squared_from_density_and_temperature(
      const Scalar<double>& /*rest_mass_density*/,
      const Scalar<double>& /*temperature*/,
      const Scalar<double>& /*electron_fraction*/) const = 0;
  virtual Scalar<DataVector> sound_speed_squared_from_density_and_temperature(
      const Scalar<DataVector>& /*rest_mass_density*/,
      const Scalar<DataVector>& /*temperature*/,
      const Scalar<DataVector>& /*electron_fraction*/) const = 0;
  /// @}

  /// The lower bound of the electron fraction that is valid for this EOS
  virtual double electron_fraction_lower_bound() const = 0;

  /// The upper bound of the electron fraction that is valid for this EOS
  virtual double electron_fraction_upper_bound() const = 0;

  /// The lower bound of the rest mass density that is valid for this EOS
  virtual double rest_mass_density_lower_bound() const = 0;

  /// The upper bound of the rest mass density that is valid for this EOS
  virtual double rest_mass_density_upper_bound() const = 0;

  /// The lower bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$ and electron fraction \f$Y_e\f$.
  virtual double specific_internal_energy_lower_bound(
      const double rest_mass_density, const double electron_fraction) const = 0;

  /// The upper bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$ and electron fraction \f$Y_e\f$.
  virtual double specific_internal_energy_upper_bound(
      const double rest_mass_density, const double electron_fraction) const = 0;

  /// The lower bound of the specific enthalpy that is valid for this EOS
  virtual double specific_enthalpy_lower_bound() const = 0;

  /// The lower bound of the temperature that is valid for this EOS
  virtual double temperature_lower_bound() const = 0;

  /// The upper bound of the temperature that is valid for this EOS
  virtual double temperature_upper_bound() const = 0;

  /// The vacuum mass of a baryon for this EOS
  virtual double baryon_mass() const {
    return hydro::units::geometric::default_baryon_mass;
  }
};

/// Compare two equations of state for equality
template <bool IsRelLhs, bool IsRelRhs, size_t ThermoDimLhs,
          size_t ThermoDimRhs>
bool operator==(const EquationOfState<IsRelLhs, ThermoDimLhs>& lhs,
                const EquationOfState<IsRelRhs, ThermoDimRhs>& rhs) {
  if constexpr (IsRelLhs == IsRelRhs and ThermoDimLhs == ThermoDimRhs) {
    return typeid(lhs) == typeid(rhs) and lhs.is_equal(rhs);
  } else {
    return false;
  }
}
template <bool IsRelLhs, bool IsRelRhs, size_t ThermoDimLhs,
          size_t ThermoDimRhs>
bool operator!=(const EquationOfState<IsRelLhs, ThermoDimLhs>& lhs,
                const EquationOfState<IsRelRhs, ThermoDimRhs>& rhs) {
  return not(lhs == rhs);
}
}  // namespace EquationsOfState

/// \cond
#define EQUATION_OF_STATE_FUNCTIONS_1D                      \
  (pressure_from_density, rest_mass_density_from_enthalpy,  \
   specific_internal_energy_from_density, chi_from_density, \
   kappa_times_p_over_rho_squared_from_density)

#define EQUATION_OF_STATE_FUNCTIONS_2D                                   \
  (pressure_from_density_and_energy, pressure_from_density_and_enthalpy, \
   specific_internal_energy_from_density_and_pressure,                   \
   temperature_from_density_and_energy,                                  \
   specific_internal_energy_from_density_and_temperature,                \
   chi_from_density_and_energy,                                          \
   kappa_times_p_over_rho_squared_from_density_and_energy)

#define EQUATION_OF_STATE_FUNCTIONS_3D                                      \
  (pressure_from_density_and_energy, pressure_from_density_and_temperature, \
   temperature_from_density_and_energy,                                     \
   specific_internal_energy_from_density_and_temperature,                   \
   sound_speed_squared_from_density_and_temperature)

#define EQUATION_OF_STATE_ARGUMENTS_EXPAND(z, n, type) \
  BOOST_PP_COMMA_IF(n) const Scalar<type>&

#define EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS_HELPER(r, DIM,        \
                                                         FUNCTION_NAME) \
  Scalar<double> FUNCTION_NAME(BOOST_PP_REPEAT(                         \
      DIM, EQUATION_OF_STATE_ARGUMENTS_EXPAND, double)) const override; \
  Scalar<DataVector> FUNCTION_NAME(BOOST_PP_REPEAT(                     \
      DIM, EQUATION_OF_STATE_ARGUMENTS_EXPAND, DataVector)) const override;

/// \endcond

/*!
 * \ingroup EquationsOfStateGroup
 * \brief Macro used to generate forward declarations of member functions in
 * derived classes
 */
#define EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(DERIVED, DIM)            \
  BOOST_PP_LIST_FOR_EACH(                                                  \
      EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS_HELPER, DIM,               \
      BOOST_PP_TUPLE_TO_LIST(BOOST_PP_TUPLE_ELEM(                          \
          BOOST_PP_SUB(DIM, 1),                                            \
          (EQUATION_OF_STATE_FUNCTIONS_1D, EQUATION_OF_STATE_FUNCTIONS_2D, \
           EQUATION_OF_STATE_FUNCTIONS_3D))))                              \
                                                                           \
  /* clang-tidy: do not use non-const references */                        \
  void pup(PUP::er& p) override; /* NOLINT */                              \
                                                                           \
  explicit DERIVED(CkMigrateMessage* msg);

/// \cond
#define EQUATION_OF_STATE_FORWARD_ARGUMENTS(z, n, unused) \
  BOOST_PP_COMMA_IF(n) arg##n

#define EQUATION_OF_STATE_ARGUMENTS_EXPAND_NAMED(z, n, type) \
  BOOST_PP_COMMA_IF(n) const Scalar<type>& arg##n

#define EQUATION_OF_STATE_MEMBER_DEFINITIONS_HELPER(                        \
    TEMPLATE, DERIVED, DATA_TYPE, DIM, FUNCTION_NAME)                       \
  TEMPLATE                                                                  \
  Scalar<DATA_TYPE> DERIVED::FUNCTION_NAME(BOOST_PP_REPEAT(                 \
      DIM, EQUATION_OF_STATE_ARGUMENTS_EXPAND_NAMED, DATA_TYPE)) const {    \
    return FUNCTION_NAME##_impl(                                            \
        BOOST_PP_REPEAT(DIM, EQUATION_OF_STATE_FORWARD_ARGUMENTS, UNUSED)); \
  }

#define EQUATION_OF_STATE_MEMBER_DEFINITIONS_HELPER_2(r, ARGS, FUNCTION_NAME) \
  EQUATION_OF_STATE_MEMBER_DEFINITIONS_HELPER(                                \
      BOOST_PP_TUPLE_ELEM(0, ARGS), BOOST_PP_TUPLE_ELEM(1, ARGS),             \
      BOOST_PP_TUPLE_ELEM(2, ARGS), BOOST_PP_TUPLE_ELEM(3, ARGS),             \
      FUNCTION_NAME)
/// \endcond

#define EQUATION_OF_STATE_MEMBER_DEFINITIONS(TEMPLATE, DERIVED, DATA_TYPE, \
                                             DIM)                          \
  BOOST_PP_LIST_FOR_EACH(                                                  \
      EQUATION_OF_STATE_MEMBER_DEFINITIONS_HELPER_2,                       \
      (TEMPLATE, DERIVED, DATA_TYPE, DIM),                                 \
      BOOST_PP_TUPLE_TO_LIST(BOOST_PP_TUPLE_ELEM(                          \
          BOOST_PP_SUB(DIM, 1),                                            \
          (EQUATION_OF_STATE_FUNCTIONS_1D, EQUATION_OF_STATE_FUNCTIONS_2D, \
           EQUATION_OF_STATE_FUNCTIONS_3D))))

/// \cond
#define EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS_HELPER(r, DIM,        \
                                                              FUNCTION_NAME) \
  template <class DataType>                                                  \
  Scalar<DataType> FUNCTION_NAME##_impl(BOOST_PP_REPEAT(                     \
      DIM, EQUATION_OF_STATE_ARGUMENTS_EXPAND, DataType)) const;
/// \endcond

#define EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(DIM)                \
  BOOST_PP_LIST_FOR_EACH(                                                  \
      EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS_HELPER, DIM,          \
      BOOST_PP_TUPLE_TO_LIST(BOOST_PP_TUPLE_ELEM(                          \
          BOOST_PP_SUB(DIM, 1),                                            \
          (EQUATION_OF_STATE_FUNCTIONS_1D, EQUATION_OF_STATE_FUNCTIONS_2D, \
           EQUATION_OF_STATE_FUNCTIONS_3D))))
