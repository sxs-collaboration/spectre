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
 * \brief An equation of state given by parametrized enthalpy
 *
 * This equation of state is determined as a function of \f$x =
 * \ln(\rho/\rho_0)\f$ where \f$\rho\f$ is the rest mass density and
 * \f$\rho_0\f$ is the provided reference density.
 * The pseudo-enthalpy \f$h \equiv (p + rho  + u)/rho\f$
 * is expanded as
 *
 * \f{equation}
 * h(x) = \sum_i a_i x^i + \sum_j b_j \sin(jkx) + c_j \cos(jkx)
 * \f}
 *
 * This form allows for convenient calculation of thermodynamic
 * quantities for a cold equation of state. For example
 *
 * \f{equation}
 * h(x) = \frac{d e} {d \rho} |_{x = \log(\rho/\rho_0)}
 * \f}
 *
 * where \f$e\f$ is the total energy density.  At the same time \f$ dx =
 * d\rho/\rho \f$ so \f$ \rho_0 e^x dx = d \rho \f$ Therefore,
 *
 * \f{equation}
 * e(x) - e(x_0) = \int_{x_0}^x h(x') e^{x'} dx '
 * \f}
 *
 * This can be computed analytically because
 *
 * \f{equation}
 *  \int a_i \frac{x^i}{i!} e^{x} dx = \sum_{j \leq i} a_i (-1)^{i-j}
 * \frac{(x)^{j}}{j!}
 * + C \f}
 *
 * and
 *
 * \f{equation}
 * \int b_j \sin(j k x) e^x dx = b_j e^x \frac{\sin(jkx) - j k \cos(jkx)}{j^2
 * k^2 + 1} \f}
 *
 * \f{equation}
 * \int c_j \cos(j k x) e^x dx = b_j e^x \frac{\cos(jkx) + j k \sin(jkx)}{j^2
 * k^2 + 1} \f}
 *
 * From this most other thermodynamic quantities can be computed
 * analytically
 *
 * The internal energy density
 * \f{equation}
 * \epsilon(x)\rho(x) = e(x)  - \rho(x)
 * \f}
 *
 * The pressure
 * \f{equation}
 * p(x) = \rho(x) h(x) - e(x)
 * \f}
 *
 * The derivative of the pressure with respect to the rest mass density
 * \f{equation}
 * \chi(x) = \frac{dp}{d\rho} |_{x = x(\rho)} = \frac{dh}{dx}
 * \f}
 *
 * Below the minimum density, a spectral parameterization
 * is used.
 *
 *
 *
 */
template <typename LowDensityEoS>
class Enthalpy : public EquationOfState<true, 1> {
 private:
  struct Coefficients {
    std::vector<double> polynomial_coefficients;
    std::vector<double> sin_coefficients;
    std::vector<double> cos_coefficients;
    double trig_scale;
    double reference_density;
    bool has_exponential_prefactor;
    double exponential_external_constant;
    Coefficients() = default;
    ~Coefficients() = default;
    Coefficients(const Coefficients& coefficients) = default;
    Coefficients(std::vector<double> in_polynomial_coefficients,
                 std::vector<double> in_sin_coefficients,
                 std::vector<double> in_cos_coefficients, double in_trig_scale,
                 double in_reference_density,
                 double in_exponential_constant =
                     std::numeric_limits<double>::quiet_NaN());
    bool operator==(const Coefficients& rhs) const;

    Enthalpy<LowDensityEoS>::Coefficients compute_exponential_integral(
        const std::pair<double, double>& initial_condition,
        const double minimum_density);
    Enthalpy<LowDensityEoS>::Coefficients compute_derivative();
    void pup(PUP::er& p);
  };

 public:
  static constexpr size_t thermodynamic_dim = 1;
  static constexpr bool is_relativistic = true;

  static std::string name() {
    return "Enthalpy(" + pretty_type::name<LowDensityEoS>() + ")";
  }

  struct ReferenceDensity {
    using type = double;
    static constexpr Options::String help = {"Reference density rho_0"};
    static double lower_bound() { return 0.0; }
  };

  struct MinimumDensity {
    using type = double;
    static constexpr Options::String help = {
        "Minimum valid density rho_min,"
        " for this parametrization"};
    static double lower_bound() { return 0.0; }
  };
  struct MaximumDensity {
    using type = double;
    static constexpr Options::String help = {"Maximum density for this EoS"};
    static double lower_bound() { return 0.0; }
  };

  struct PolynomialCoefficients {
    using type = std::vector<double>;
    static constexpr Options::String help = {"Polynomial coefficients a_i"};
  };

  struct TrigScaling {
    using type = double;
    static constexpr Options::String help = {
        "Fundamental wavenumber of trig "
        "functions, k"};
    static double lower_bound() { return 0.0; }
  };

  struct SinCoefficients {
    using type = std::vector<double>;
    static constexpr Options::String help = {"Sine coefficients b_j"};
  };
  struct CosCoefficients {
    using type = std::vector<double>;
    static constexpr Options::String help = {"Cosine coefficients c_j"};
  };
  struct StitchedLowDensityEoS {
    using type = LowDensityEoS;
    static std::string name() {
      return pretty_type::short_name<LowDensityEoS>();
    }
    static constexpr Options::String help = {
        "Low density EoS stitched at the MinimumDensity"};
  };

  struct TransitionDeltaEpsilon {
    using type = double;
    static constexpr Options::String help = {
        "the change in internal energy across the low-"
        "to-high-density transition, generically 0.0"};
    static double lower_bound() { return 0.0; }
  };

  static constexpr Options::String help = {
      "An EoS with a parametrized value h(log(rho/rho_0)) with h the specific "
      "enthalpy and rho the baryon rest mass density.  The enthalpy is "
      "expanded as a sum of polynomial terms and trigonometric corrections. "
      "let x = log(rho/rho_0) in"
      "h(x) = \\sum_i a_ix^i + \\sum_j b_jsin(k * j * x) + c_jcos(k * j * x) "
      "Note that rho(x)(1+epsilon(x)) = int_0^x e^x' h((x') dx' can be "
      "computed "
      "analytically, and therefore so can "
      "P(x) = rho(x) * (h(x) - (1 + epsilon(x))) "};

  using options =
      tmpl::list<ReferenceDensity, MaximumDensity, MinimumDensity, TrigScaling,
                 PolynomialCoefficients, SinCoefficients, CosCoefficients,
                 StitchedLowDensityEoS, TransitionDeltaEpsilon>;

  Enthalpy() = default;
  Enthalpy(const Enthalpy&) = default;
  Enthalpy& operator=(const Enthalpy&) = default;
  Enthalpy(Enthalpy&&) = default;
  Enthalpy& operator=(Enthalpy&&) = default;
  ~Enthalpy() override = default;

  Enthalpy(double reference_density, double max_density, double min_density,
           double trig_scale,
           const std::vector<double>& polynomial_coefficients,
           const std::vector<double>& sin_coefficients,
           const std::vector<double>& cos_coefficients,
           const LowDensityEoS& low_density_eos,
           const double transition_delta_epsilon);

  std::unique_ptr<EquationOfState<true, 1>> get_clone() const override;

  bool is_equal(const EquationOfState<true, 1>& rhs) const override;

  bool operator==(const Enthalpy<LowDensityEoS>& rhs) const;

  bool operator!=(const Enthalpy<LowDensityEoS>& rhs) const;

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(Enthalpy, 1)

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<true, 1>), Enthalpy);

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

  SPECTRE_ALWAYS_INLINE
  bool in_low_density_domain(const double rest_mass_density) const {
    return rest_mass_density < minimum_density_;
  }

  double x_from_density(const double rest_mass_density) const;
  double density_from_x(const double x) const;
  double energy_density_from_log_density(const double x,
                                         const double rest_mass_density) const;
  static double evaluate_coefficients(
      const Enthalpy::Coefficients& coefficients, const double x,
      const double exponential_prefactor =
          std::numeric_limits<double>::signaling_NaN());
  static Enthalpy::Coefficients compute_pressure_coefficients(
      const typename Enthalpy::Coefficients& enthalpy,
      const typename Enthalpy::Coefficients& energy_density);

  double chi_from_density(const double rest_mass_density) const;
  double specific_internal_energy_from_density(
      const double rest_mass_density) const;
  double specific_enthalpy_from_density(const double rest_mass_density) const;
  double pressure_from_density(const double rest_mass_density) const;
  double pressure_from_log_density(const double x,
                                   const double rest_mass_density) const;
  double rest_mass_density_from_enthalpy(const double specific_enthalpy) const;

  double reference_density_ = std::numeric_limits<double>::signaling_NaN();
  double minimum_density_ = std::numeric_limits<double>::signaling_NaN();
  double maximum_density_ = std::numeric_limits<double>::signaling_NaN();
  double minimum_enthalpy_ = std::numeric_limits<double>::signaling_NaN();

  LowDensityEoS low_density_eos_;
  Coefficients coefficients_;
  Coefficients exponential_integral_coefficients_;
  Coefficients derivative_coefficients_;
  Coefficients pressure_coefficients_;
};

}  // namespace EquationsOfState
