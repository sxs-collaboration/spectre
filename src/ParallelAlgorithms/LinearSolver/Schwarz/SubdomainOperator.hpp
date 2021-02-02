// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace LinearSolver::Schwarz {

/*!
 * \brief Abstract base class for the subdomain operator, i.e. the linear
 * operator restricted to an element-centered Schwarz subdomain
 *
 * A subdomain operator must implement these member function templates
 * - `operator()`: Applies the operator to the element-centered subdomain data.
 *   It must take these arguments, in this order:
 *   - `gsl::not_null<LinearSolver::Schwarz::ElementCenteredSubdomainData*>`:
 *     The data where the result of the operator applied to the operand should
 *     be written to
 *   - `LinearSolver::Schwarz::ElementCenteredSubdomainData`: The operand data
 *     to which the operator should be applied
 *   - `db::DataBox`: The element's DataBox. Can be used to retrieve any data on
 *     the subdomain geometry and to access and mutate persistent memory
 *     buffers.
 *
 * Since the subdomain operator has access to the element's DataBox, it can
 * retrieve any background data, i.e. data that does not depend on the variables
 * it operates on. Background data on overlap regions with other elements can be
 * either initialized in advance if possible or communicated with the
 * `LinearSolver::Schwarz::Actions::SendOverlapFields` and
 * `LinearSolver::Schwarz::Actions::ReceiveOverlapFields` actions. Note that
 * such communication should not be necessary between iterations of the Schwarz
 * solve, but only between successive solves, because background data should not
 * change during the solve. All variable data is passed to the operator as the
 * `operand` argument (see above).
 *
 * Here's an example of a subdomain operator that is the restriction of an
 * explicit global matrix:
 *
 * \snippet Test_SchwarzAlgorithm.cpp subdomain_operator
 */
template <size_t Dim>
class SubdomainOperator {
 public:
  static constexpr size_t volume_dim = Dim;

 protected:
  SubdomainOperator() = default;

  // Derived classes must implement operator() member function template
};

}  // namespace LinearSolver::Schwarz
