// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

/*!
 * \brief Holds the coupling functions for Einstein-scalar-Gauss-Bonnet gravity.
 */
namespace ScalarTensor::sgb::CouplingFunctions {

/*!
 * \brief Basic interface for the object defining the coupling function.
 *
 * \details This class is pure virtual and factory creatable, so that it acts as
 * a unique interface for all forms of coupling functions. The public member
 * functions that compute the profiles of \f$F[\Psi]\f$ and its
 * functional derivatives are: ``coupling_function``,
 * ``coupling_function_prime`` and ``coupling_function_prime_prime``; they
 * internally call ``coupling_function_impl``, ``coupling_function_prime_impl``
 * and ``coupling_function_prime_prime_impl``, which are pure virtual and
 * protected. Any class implementing a coupling function should inherit publicly
 * from this class and override only the latter three methods.
 */
class CouplingFunction : public PUP::able {
 public:
  WRAPPED_PUPable_abstract(CouplingFunction);  // NOLINT
  explicit CouplingFunction(CkMigrateMessage* m) : PUP::able(m) {}

  CouplingFunction() = default;
  ~CouplingFunction() override = default;

  /// @{
  /*!
   * \brief Evaluates the coupling function on a scalar field profile,
   * \f$F[\Psi]\f$.
   *
   * This methods act as an external interface and internally calls
   * ``coupling_function_impl``, which is the virtual function that needs to be
   * specialized for the different types of couplings.
   *
   * \see CouplingFunction::coupling_function_impl
   */
  void coupling_function(gsl::not_null<Scalar<DataVector>*> function_values,
                         const Scalar<DataVector>& scalar_field) const;

  Scalar<DataVector> coupling_function(
      const Scalar<DataVector>& scalar_field) const;
  /// @}

  /// @{
  /*!
   * \brief Evaluates the functional derivative of the coupling function,
   * \f$\frac{\delta F[\Psi]}{\delta \Psi} \f$, on a scalar field profile.
   *
   * This methods act as an external interface and internally calls
   * ``coupling_function_prime_impl``, which is the virtual function that needs
   * to be specialized for the different types of couplings.
   *
   * \see CouplingFunction::coupling_function_prime_impl
   */
  void coupling_function_prime(
      gsl::not_null<Scalar<DataVector>*> function_values,
      const Scalar<DataVector>& scalar_field) const;

  Scalar<DataVector> coupling_function_prime(
      const Scalar<DataVector>& scalar_field) const;
  /// @}

  /// @{
  /*!
   * \brief Evaluates the second functional derivative of the coupling
   * function, \f$\frac{\delta^2 F[\Psi]}{\delta \Psi^2} \f$, on a scalar field
   * profile.
   *
   * This methods act as an external interface and internally calls
   * ``coupling_function_prime_prime_impl``, which is the virtual function that
   * needs to be specialized for the different types of couplings.
   *
   * \see CouplingFunction::coupling_function_prime_prime_impl
   */
  void coupling_function_prime_prime(
      gsl::not_null<Scalar<DataVector>*> function_values,
      const Scalar<DataVector>& scalar_field) const;

  Scalar<DataVector> coupling_function_prime_prime(
      const Scalar<DataVector>& scalar_field) const;
  /// @}

 protected:
  /*!
   * \brief Evaluates the coupling function on a scalar field profile,
   * \f$F[\Psi]\f$.
   *
   * This is the virtual function that is called by
   * ``coupling_function``, and needs to be specialized for the different
   * types of couplings.
   *
   * \see CouplingFunction::coupling_function
   */
  virtual void coupling_function_impl(
      gsl::not_null<Scalar<DataVector>*> function_values,
      const Scalar<DataVector>& scalar_field) const = 0;

  /*!
   * \brief Implementation of the function that evaluates the first functional
   * derivative of the coupling function on a scalar field profile,
   * \f$\frac{\delta F[\Psi]}{\delta \Psi}\f$.
   *
   * This is the virtual function that is called by
   * ``coupling_function_prime``, and needs to be specialized for the different
   * types of couplings.
   *
   * \see CouplingFunction::coupling_function_prime
   */
  virtual void coupling_function_prime_impl(
      gsl::not_null<Scalar<DataVector>*> function_values,
      const Scalar<DataVector>& scalar_field) const = 0;

  /*!
   * \brief Implementation of the function that evaluates the second functional
   * derivative of the coupling function on a scalar field profile,
   * \f$\frac{\delta^2 F[\Psi]}{\delta \Psi^2}\f$.
   *
   * This is the virtual function that is called by
   * ``coupling_function_prime_prime``, and needs to be specialized for the
   * different types of couplings.
   *
   * \see CouplingFunction::coupling_function_prime_prime
   */
  virtual void coupling_function_prime_prime_impl(
      gsl::not_null<Scalar<DataVector>*> function_values,
      const Scalar<DataVector>& scalar_field) const = 0;
};

}  // namespace ScalarTensor::sgb::CouplingFunctions
