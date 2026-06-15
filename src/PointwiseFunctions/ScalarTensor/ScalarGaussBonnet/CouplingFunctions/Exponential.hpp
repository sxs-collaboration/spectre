// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingFunctions/CouplingFunction.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::sgb::CouplingFunctions {

/*!
 * \brief Coupling function of exponential type.
 *
 * \details The functional form of this coupling is
 * \f$F[\Psi] = \lambda e^{\gamma \Psi}\f$, where both \f$\lambda\f$ and
 * \f$\gamma\f$ are dimensionless.
 *
 * In the case of exponential coupling the theory does not admit the Kerr metric
 * as a stationary solution, and spontaneous scalarization cannot take place.
 * For this type of coupling the theory is also referred to as
 * _Einstein-dilaton-Gauss-Bonnet_.
 */
class Exponential : public CouplingFunction {
 public:
  static constexpr Options::String help = {
      "Coupling function of exponential type"};

  struct Lambda {
    using type = double;
    static constexpr Options::String help = {
        "The numerical dimensionless coefficient in front of the exponential"};
    static std::string name() { return "lambda"; }
  };

  struct Gamma {
    using type = double;
    static constexpr Options::String help = {
        "The numerical dimensionless coefficient in the exponent"};
    static std::string name() { return "gamma"; }
  };

  using options = tmpl::list<Lambda, Gamma>;

  Exponential() = default;
  Exponential(double lambda, double gamma);
  Exponential(const Exponential&) = default;
  Exponential& operator=(const Exponential&) = default;
  Exponential(Exponential&&) = default;
  Exponential& operator=(Exponential&&) = default;
  ~Exponential() override = default;

  explicit Exponential(CkMigrateMessage* m) : CouplingFunction(m) {}
  WRAPPED_PUPable_decl_template(Exponential);
  void pup(PUP::er& p) override;

  double get_lambda() const { return lambda_; }
  double get_gamma() const { return gamma_; }

 protected:
  /*!
   * \brief Specialization of the function that evaluates the coupling function
   * on a scalar field profile.
   *
   * It returns the profile \f$F[\Psi] = \lambda e^{- \gamma \Psi}\f$.
   */
  void coupling_function_impl(
      gsl::not_null<Scalar<DataVector>*> function_values,
      const Scalar<DataVector>& scalar_field) const override;

  /*!
   * \brief Specialization of the function that evaluates the first functional
   * derivative of the coupling function on a scalar field profile.
   *
   * It returns the profile
   * \f$\frac{\delta F[\Psi]}{\delta \Psi}
   * = - \gamma \lambda e^{- \gamma \Psi}\f$.
   */
  void coupling_function_prime_impl(
      gsl::not_null<Scalar<DataVector>*> function_values,
      const Scalar<DataVector>& scalar_field) const override;

  /*!
   * \brief Specialization of the function that evaluates the second functional
   * derivative of the coupling function on a scalar field profile.
   *
   * It returns the profile
   * \f$\frac{\delta^2 F[\Psi]}{\delta \Psi^2}
   * = \gamma^2 \lambda e^{- \gamma \Psi}\f$.
   */
  void coupling_function_prime_prime_impl(
      gsl::not_null<Scalar<DataVector>*> function_values,
      const Scalar<DataVector>& scalar_field) const override;

 private:
  /*!
   * \brief Coupling constant \f$\lambda\f$, appearing as a global factor in
   * the coupling function.
   */
  double lambda_{std::numeric_limits<double>::signaling_NaN()};

  /*!
   * \brief Coupling constant \f$\gamma\f$, appearing in the exponent of
   * the coupling function.
   */
  double gamma_{std::numeric_limits<double>::signaling_NaN()};
};

}  // namespace ScalarTensor::sgb::CouplingFunctions
