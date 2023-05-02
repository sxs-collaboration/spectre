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
 * \brief Apply a Kreiss-Oliger filter to \f$g_{ab}\f$, \f$\Phi_{iab}\f$, and
 * \f$\Pi_{ab}\f$.
 */
void spacetime_kreiss_oliger_filter(
    gsl::not_null<Variables<
        typename grmhd::GhValenciaDivClean::System::variables_tag::tags_list>*>
        result,
    const Variables<
        typename grmhd::GhValenciaDivClean::System::variables_tag::tags_list>&
        volume_evolved_variables,
    const FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        evolution::dg::subcell::GhostData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& all_ghost_data,
    const Mesh<3>& volume_mesh, size_t order, double epsilon);
}  // namespace grmhd::GhValenciaDivClean::fd
