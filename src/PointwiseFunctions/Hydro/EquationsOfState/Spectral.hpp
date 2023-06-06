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
#include <cstddef>
#include <limits>
#include <pup.h>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep
#include "Utilities/Math.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace EquationsOfState {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief A spectral equation of state
 *
 * This equation of state is determined as a function of \f$x =
 * \ln(\rho/\rho_0)\f$ where \f$\rho\f$ is the rest mass density and
 * \f$\rho_0\f$ is the provided reference density.  The adiabatic
 * index \f$\Gamma(x)\f$ is defined such that
 * \f{equation}{
 * \frac{d \ln p}{dx} = \Gamma(x) = \sum_{n=0}^N
 * \gamma_n x^n
 * \f}
 *
 * for the set of spectral coefficinets \f$\gamma_n\f$ when
 * \f$0 < x < x_u = \ln(\rho_u/\rho_0)\f$, where \f$\rho_u\f$ is the provided
 * upper density.
 *
 * For \f$ x < 0 \f$, \f$ \Gamma(x) = \gamma_0 \f$.
 *
 * For \f$ x > x_u \f$, \f$ \Gamma(x) = \Gamma(x_u) \f$
 *
 *
 */
class Spectral : public EquationOfState<true, 1> {
 public:
  static constexpr size_t thermodynamic_dim = 1;
  static constexpr bool is_relativistic = true;

  struct ReferenceDensity {
    using type = double;
    static constexpr Options::String help = {"Reference density rho_0"};
    static double lower_bound() { return 0.0; }
  };

  struct ReferencePressure {
    using type = double;
    static constexpr Options::String help = {"Reference pressure p_0"};
    static double lower_bound() { return 0.0; }
  };

  struct Coefficients {
    using type = std::vector<double>;
    static constexpr Options::String help = {"Spectral coefficients gamma_i"};
  };

  struct UpperDensity {
    using type = double;
    static constexpr Options::String help = {"Upper density rho_u"};
    static double lower_bound() { return 0.0; }
  };

  static constexpr Options::String help = {
      "A spectral equation of state.  Defining x = log(rho/rho_0), Gamma(x) = "
      "Sum_i gamma_i x^i, then the pressure is determined from d(log P)/dx = "
      "Gamma(x) for x > 0.  For x < 0 the EOS is a polytrope with "
      "Gamma(x)=Gamma(0).  For x > x_u = log(rho_u/rho_0), Gamma(x) = "
      "Gamma(x_u).\n"
      "To get smooth equations of state, it is recommended that the second "
      "and third supplied coefficient should be 0. It is up to the user to "
      "choose coefficients that are physically reasonable, e.g. that "
      "satisfy causality."};

  using options = tmpl::list<ReferenceDensity, ReferencePressure, Coefficients,
                             UpperDensity>;

  Spectral() = default;
  Spectral(const Spectral&) = default;
  Spectral& operator=(const Spectral&) = default;
  Spectral(Spectral&&) = default;
  Spectral& operator=(Spectral&&) = default;
  ~Spectral() override = default;

  Spectral(double reference_density, double reference_pressure,
           std::vector<double> coefficients, double upper_density);

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(Spectral, 1)

  std::unique_ptr<EquationOfState<true, 1>> get_clone() const override;

  bool operator==(const Spectral& rhs) const;

  bool operator!=(const Spectral& rhs) const;

  bool is_equal(const EquationOfState<true, 1>& rhs) const override;

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<true, 1>), Spectral);

  /// The lower bound of the rest mass density that is valid for this EOS
  double rest_mass_density_lower_bound() const override { return 0.0; }

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override {
    return std::numeric_limits<double>::max();
  }

  /// The lower bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_lower_bound(
      const double /* rest_mass_density */) const override {
    return 0.0;
  }

  /// The upper bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_upper_bound(
      const double /* rest_mass_density */) const override {
    return std::numeric_limits<double>::max();
  }

  /// The lower bound of the specific enthalpy that is valid for this EOS
  double specific_enthalpy_lower_bound() const override { return 1.0; }

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(1)

  double gamma(const double x) const;
  double integral_of_gamma(const double x) const;
  double chi_from_density(const double density) const;
  double specific_internal_energy_from_density(const double density) const;
  double specific_enthalpy_from_density(const double density) const;
  double pressure_from_density(const double density) const;
  double pressure_from_log_density(const double x) const;
  double rest_mass_density_from_enthalpy(const double specific_enthalpy) const;

  double reference_density_ = std::numeric_limits<double>::signaling_NaN();
  double reference_pressure_ = std::numeric_limits<double>::signaling_NaN();
  std::vector<double> integral_coefficients_{};
  std::vector<double> gamma_coefficients_{};
  double x_max_ = std::numeric_limits<double>::signaling_NaN();
  double gamma_of_x_max_ = std::numeric_limits<double>::signaling_NaN();
  double integral_of_gamma_of_x_max_ =
      std::numeric_limits<double>::signaling_NaN();
  std::vector<double> table_of_specific_energies_{};
  // Information for Gaussian quadrature
  size_t number_of_quadrature_coefs_ =
      std::numeric_limits<size_t>::signaling_NaN();
  std::vector<double> quadrature_weights_{};
  std::vector<double> quadrature_points_{};
};

}  // namespace EquationsOfState
