// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/Tags.hpp"

namespace LinearSolver::Schwarz {
/// Protocols related to the Schwarz solver
namespace protocols {

/*!
 * \brief A linear operator restricted to an element-centered Schwarz subdomain
 *
 * A subdomain operator must have these static member variables:
 * - `size_t volume_dim`: The number of spatial dimensions
 *
 * A subdomain operator must have these type aliases:
 * - `element_operator`: A DataBox-invokable that applies the operator to the
 * data on the central element of the subdomain. It must take these arguments,
 * in this order, and following the types of the `argument_tags` it specifies:
 *   - `LinearSolver::Schwarz::ElementCenteredSubdomainData`: The operand data
 *   to which the operator should be applied
 *   - `gsl::not_null<LinearSolver::Schwarz::ElementCenteredSubdomainData*>`:
 *   The data where the result of the operator applied to the operand should be
 *   written to
 *   - `gsl::not_null<SubdomainOperator*>`: An instance of the subdomain
 *   operator. Can be used to store buffers to avoid repeatedly allocating
 *   memory or to cache quantities for subsequent invocations of the
 *   `face_operator`.
 * - `face_operator`: A DataBox-invokable on interfaces that applies the
 * operator to the data on an element face. It must be specialized for
 * `domain::Tags::InternalDirections` and
 * `domain::Tags::BoundaryDirectionsInterior` and must take the same arguments
 * as the `element_operator` in addition to the `argument_tags` it specifies.
 *
 * Since the subdomain operator is implemented as DataBox-invokables, it can
 * retrieve any background data, i.e. data that does not depend on the variables
 * it operates on, from the DataBox. Background data on overlap regions with
 * other elements can be either initialized in advance if possible or
 * communicated with the `LinearSolver::Schwarz::Actions::SendOverlapFields` and
 * `LinearSolver::Schwarz::Actions::ReceiveOverlapFields` actions. Note that
 * such communication should not be necessary between iterations of the Schwarz
 * solve, but only between successive solves, because background data should not
 * change during the solve. All variable data is passed to the operator as an
 * argument (see above).
 *
 * Here's an example of a subdomain operator that is the restriction of an
 * explicit global matrix:
 *
 * \snippet Test_SchwarzAlgorithm.cpp subdomain_operator
 */
struct SubdomainOperator {
  template <typename ConformingType>
  struct test {
    static constexpr size_t volume_dim = ConformingType::volume_dim;
    using element_operator = typename ConformingType::element_operator;
    using face_operator_internal =
        typename ConformingType::template face_operator<
            domain::Tags::InternalDirections<volume_dim>>;
    using face_operator_external =
        typename ConformingType::template face_operator<
            domain::Tags::BoundaryDirectionsInterior<volume_dim>>;
  };
};

}  // namespace protocols
}  // namespace LinearSolver::Schwarz
