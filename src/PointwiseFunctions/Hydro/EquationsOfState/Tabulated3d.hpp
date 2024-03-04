// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/control/expr_iif.hpp>
#include <boost/preprocessor/list/adt.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <limits>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "IO/H5/EosTable.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/Interpolation/MultiLinearSpanInterpolation.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace EquationsOfState {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief Nuclear matter equation of state in tabulated form.
 *
 * The equation of state takes the form
 *
 * \f[
 * p = p (T, rho, Y_e)
 * \f]
 *
 * where \f$\rho\f$ is the rest mass density, \f$T\f$ is the
 * temperature, and \f$Y_e\f$ is the electron fraction.
 * The temperature is given in units of MeV.
 */
template <bool IsRelativistic>
class Tabulated3D : public EquationOfState<IsRelativistic, 3> {
 public:
  static constexpr size_t thermodynamic_dim = 3;
  static constexpr bool is_relativistic = IsRelativistic;

  static constexpr Options::String help = {
      "A tabulated three-dimensional equation of state.\n"
      "The energy density, pressure and sound speed "
      "are tabulated as a function of density, electron_fraction and "
      "temperature."};

  struct TableFilename {
    using type = std::string;
    static constexpr Options::String help{"File name of the EOS table"};
  };

  struct TableSubFilename {
    using type = std::string;
    static constexpr Options::String help{
        "Subfile name of the EOS table, e.g., 'dd2'."};
  };

  using options = tmpl::list<TableFilename, TableSubFilename>;

  /// Fields stored in the table
  enum : size_t { Epsilon = 0, Pressure, CsSquared, DeltaMu, NumberOfVars };

  Tabulated3D() = default;
  Tabulated3D(const Tabulated3D&) = default;
  Tabulated3D& operator=(const Tabulated3D&) = default;
  Tabulated3D(Tabulated3D&&) = default;
  Tabulated3D& operator=(Tabulated3D&&) = default;
  ~Tabulated3D() override = default;

  explicit Tabulated3D(const std::string& filename,
                       const std::string& subfilename);

  explicit Tabulated3D(std::vector<double> electron_fraction,
                       std::vector<double> log_density,
                       std::vector<double> log_temperature,
                       std::vector<double> table_data, double energy_shift,
                       double enthalpy_minimum);

  explicit Tabulated3D(const h5::EosTable& spectre_eos);

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(Tabulated3D, 3)

  template <class DataType>
  void convert_to_table_quantities(
      const gsl::not_null<Scalar<DataType>*> converted_electron_fraction,
      const gsl::not_null<Scalar<DataType>*> log_rest_mass_density,
      const gsl::not_null<Scalar<DataType>*> log_temperature,
      const Scalar<DataType>& electron_fraction,
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& temperature) const {
    get(*converted_electron_fraction) = get(electron_fraction);
    get(*log_rest_mass_density) = get(rest_mass_density);
    get(*log_temperature) = get(temperature);

    // Enforce physicality of input
    // We reuse the same variables here, these are not log yet.
    enforce_physicality(*converted_electron_fraction, *log_rest_mass_density,
                        *log_temperature);

    // Table uses log T and log rho
    get(*log_rest_mass_density) = log(get(*log_rest_mass_density));
    get(*log_temperature) = log(get(*log_temperature));
  }

  std::unique_ptr<EquationOfState<IsRelativistic, 3>> get_clone()
      const override;

  void initialize(std::vector<double> electron_fraction,
                  std::vector<double> log_density,
                  std::vector<double> log_temperature,
                  std::vector<double> table_data, double energy_shift,
                  double enthalpy_minimum);


  void initialize(const h5::EosTable& spectre_eos);

  bool is_equal(const EquationOfState<IsRelativistic, 3>& rhs) const override;

  /// \brief Returns `true` if the EOS is barotropic
  bool is_barotropic() const override { return false; }

  bool operator==(const Tabulated3D<IsRelativistic>& rhs) const;

  bool operator!=(const Tabulated3D<IsRelativistic>& rhs) const;

  template <class DataType>
  Scalar<DataType> equilibrium_electron_fraction_from_density_temperature_impl(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& temperature) const;

  /// @{
  /*!
   * Computes the electron fraction in beta-equilibrium \f$Y_e^{\rm eq}\f$ from
   * the rest mass density \f$\rho\f$ and the temperature \f$T\f$.
   */
  Scalar<double> equilibrium_electron_fraction_from_density_temperature(
      const Scalar<double>& rest_mass_density,
      const Scalar<double>& temperature) const {
    return equilibrium_electron_fraction_from_density_temperature_impl<double>(
        rest_mass_density, temperature);
  }

  Scalar<DataVector> equilibrium_electron_fraction_from_density_temperature(
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& temperature) const {
    return equilibrium_electron_fraction_from_density_temperature_impl<
        DataVector>(rest_mass_density, temperature);
  }
  /// @}
  //

  template <typename DataType>
  void enforce_physicality(Scalar<DataType>& electron_fraction,
                           Scalar<DataType>& density,
                           Scalar<DataType>& temperature) const;

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<IsRelativistic, 3>), Tabulated3D);

  /// The lower bound of the electron fraction that is valid for this EOS
  double electron_fraction_lower_bound() const override {
    return table_electron_fraction_.front();
  }

  /// The upper bound of the electron fraction that is valid for this EOS
  double electron_fraction_upper_bound() const override {
    return table_electron_fraction_.back();
  }

  /// The lower bound of the rest mass density that is valid for this EOS
  double rest_mass_density_lower_bound() const override {
    return std::exp((table_log_density_.front()));
  }

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override {
    return std::exp((table_log_density_.back()));
  }

  /// The lower bound of the temperature that is valid for this EOS
  double temperature_lower_bound() const override {
    return std::exp((table_log_temperature_.front()));
  }

  /// The upper bound of the temperature that is valid for this EOS
  double temperature_upper_bound() const override {
    return std::exp((table_log_temperature_.back()));
  }

  /// The lower bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$ and electron fraction \f$Y_e\f$
  double specific_internal_energy_lower_bound(
      const double rest_mass_density,
      const double electron_fraction) const override;

  /// The upper bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_upper_bound(
      const double rest_mass_density,
      const double electron_fraction) const override;

  /// The lower bound of the specific enthalpy that is valid for this EOS
  double specific_enthalpy_lower_bound() const override {
    return enthalpy_minimum_;
  }

  /// The baryon mass for this EoS
  double baryon_mass() const override {
    return hydro::units::geometric::neutron_mass;
  }

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(3)

  void initialize_interpolator();

  /// Energy shift used to account for negative specific internal energies,
  /// which are only stored logarithmically
  double energy_shift_ = 0.;

  /// Enthalpy minium  across the table
  double enthalpy_minimum_ = 1.;

  /// Main interpolator for the EoS.
  /// The ordering is  \f$(\log T. \log \rho, Y_e)\f$.
  /// Assumed to be sorted in ascending order.
  intrp::UniformMultiLinearSpanInterpolation<3, NumberOfVars> interpolator_{};
  /// Electron fraction
  std::vector<double> table_electron_fraction_{};
  /// Logarithmic rest-mass denisty
  std::vector<double> table_log_density_{};
  /// Logarithmic temperature
  std::vector<double> table_log_temperature_{};
  /// Tabulate data. Entries are stated in the enum
  std::vector<double> table_data_{};

  /// Tolerance on upper bound for root finding
  static constexpr double upper_bound_tolerance_ = 0.9999;
};

/// \cond
template <bool IsRelativistic>
PUP::able::PUP_ID EquationsOfState::Tabulated3D<IsRelativistic>::my_PUP_ID = 0;
/// \endcond

}  // namespace EquationsOfState
