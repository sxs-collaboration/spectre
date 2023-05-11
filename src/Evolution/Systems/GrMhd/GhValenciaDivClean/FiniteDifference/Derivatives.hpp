// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
namespace evolution::dg::subcell {
class GhostData;
}  // namespace evolution::dg::subcell
/// \endcond

namespace grmhd::GhValenciaDivClean::fd {
/*!
 * \brief Compute partial derivatives of the spacetime variables \f$g_{ab}\f$,
 * \f$\Phi_{iab}\f$, and \f$\Pi_{ab}\f$.
 *
 * The derivatives are computed using 4th-order FD. If we want the ability to
 * take higher order derivatives we can add an argument. However, we then need
 * to make sure we have enough ghost cells. 4th-order derivatives require only 2
 * ghost cells, which is also what we need for 2nd-order reconstruction.
 * 6th-order derivatives need 3 ghost cells so we would need to send more data
 * or only use higher order derivatives when higher order reconstruction is
 * used.
 */
void spacetime_derivatives(
    gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::deriv,
        typename grmhd::GhValenciaDivClean::System::gradients_tags,
        tmpl::size_t<3>, Frame::Inertial>>*>
        result,
    const Variables<
        typename grmhd::GhValenciaDivClean::System::variables_tag::tags_list>&
        volume_evolved_variables,
    const FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        evolution::dg::subcell::GhostData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& all_ghost_data,
    const Mesh<3>& volume_mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_jacobian);
}  // namespace grmhd::GhValenciaDivClean::fd
