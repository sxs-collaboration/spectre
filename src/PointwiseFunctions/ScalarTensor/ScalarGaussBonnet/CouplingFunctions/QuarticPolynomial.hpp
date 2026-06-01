// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingFunctions/CouplingFunction.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::sgb::CouplingFunctions {

/*!
 * \brief Coupling function of polynomial type with degree up to four:
 * \f$F[\Psi] = \sum_{i = 1}^4 a_i \Psi^i\f$.
 *
 * \details This expression can be used to model the weak-field limit of a
 * generic coupling. Also, by appropriately setting the values of the
 * coefficients \f$a_i\f$ it can be used to study different phenomenological
 * scenario. For example, the theory admits the Kerr metric with vanishing
 * scalar field as a stationary solution in the case \f$a_1 = 0\f$, and can
 * feature spontanous scalarization for sufficiently large values of \f$a_2\f$.
 * Furthermore, if \f$a_2 = 0\f$, the theory can feature nonlinear
 * scalarization. See \cite Doneva:2022ewd for a review on spontanous
 * scalarization.
 */
class QuarticPolynomial : public CouplingFunction {
 public:
  static constexpr Options::String help = {
      "Coupling function of polynomial type with degree up to four"};

  struct Linear {
    using type = double;
    static constexpr Options::String help = {
        "Coefficient of the linear term in the scalar field"};
  };

  struct Quadratic {
    using type = double;
    static constexpr Options::String help = {
        "Coefficient of the quadratic term in the scalar field"};
  };

  struct Cubic {
    using type = double;
    static constexpr Options::String help = {
        "Coefficient of the cubic term in the scalar field"};
  };

  struct Quartic {
    using type = double;
    static constexpr Options::String help = {
        "Coefficient of the quartic term in the scalar field"};
  };

  using options = tmpl::list<Linear, Quadratic, Cubic, Quartic>;

  QuarticPolynomial() = default;
  QuarticPolynomial(double linear, double quadratic, double cubic,
                    double quartic);
  QuarticPolynomial(const QuarticPolynomial&) = default;
  QuarticPolynomial& operator=(const QuarticPolynomial&) = default;
  QuarticPolynomial(QuarticPolynomial&&) = default;
  QuarticPolynomial& operator=(QuarticPolynomial&&) = default;
  ~QuarticPolynomial() override = default;

  explicit QuarticPolynomial(CkMigrateMessage* m) : CouplingFunction(m) {}
  WRAPPED_PUPable_decl_template(QuarticPolynomial);
  void pup(PUP::er& p) override;

  double get_linear() const { return linear_; }
  double get_quadratic() const { return quadratic_; }
  double get_cubic() const { return cubic_; }
  double get_quartic() const { return quartic_; }

 protected:
  /*!
   * \brief Specialization of the function that evaluates the coupling function
   * on a scalar field profile.
   *
   * It returns the profile
   * \f$F[\Psi] = a_1 \Psi + a_2 \Psi^2 + a_3 \Psi^3 + a_4 \Psi^4\f$.
   */
  void coupling_function_impl(
      gsl::not_null<Scalar<DataVector>*> function_values,
      const Scalar<DataVector>& scalar_field) const override;

  /*!
   * \brief Specialization of the function that evaluates the first functional
   * derivative of the coupling function on a scalar field profile.
   *
   * It returns the profile
   * \f$\frac{\delta F[\Psi]}{\delta \Psi} = a_1 + 2 a_2 \Psi + 3 a_3 \Psi^2
   * + 4 a_4 \Psi^3\f$.
   */
  void coupling_function_prime_impl(
      gsl::not_null<Scalar<DataVector>*> function_values,
      const Scalar<DataVector>& scalar_field) const override;

  /*!
   * \brief Specialization of the function that evaluates the second functional
   * derivative of the coupling function on a scalar field profile.
   *
   * It returns the profile
   * \f$\frac{\delta^2 F[\Psi]}{\delta \Psi^2} = 2 a_2 + 6 a_3 \Psi
   * + 12 a_4 \Psi^2\f$.
   */
  void coupling_function_prime_prime_impl(
      gsl::not_null<Scalar<DataVector>*> function_values,
      const Scalar<DataVector>& scalar_field) const override;

 private:
  /*!
   * \brief Coefficient of the term linear in the scalar field, \f$a_1\f$.
   */
  double linear_{std::numeric_limits<double>::signaling_NaN()};

  /*!
   * \brief Coefficient of the term quadratic in the scalar field, \f$a_2\f$.
   */
  double quadratic_{std::numeric_limits<double>::signaling_NaN()};

  /*!
   * \brief Coefficient of the term cubic in the scalar field, \f$a_3\f$.
   */
  double cubic_{std::numeric_limits<double>::signaling_NaN()};

  /*!
   * \brief Coefficient of the term quartic in the scalar field, \f$a_4\f$.
   */
  double quartic_{std::numeric_limits<double>::signaling_NaN()};
};

}  // namespace ScalarTensor::sgb::CouplingFunctions
