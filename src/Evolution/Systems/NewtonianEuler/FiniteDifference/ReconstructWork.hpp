// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"

/// \cond
template <typename TagsList>
class Variables;
namespace gsl {
template <typename>
class not_null;
}  // namespace gsl
template <size_t Dim>
class Direction;
template <size_t Dim>
class ElementId;
template <size_t Dim>
class Element;
template <size_t Dim>
class Mesh;
namespace evolution::dg::subcell {
class NeighborData;
}  // namespace evolution::dg::subcell
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
/// \endcond

namespace NewtonianEuler::fd {
/*!
 * \brief Reconstructs the mass density, velocity, and pressure, then computes
 * the specific internal energy and conserved variables. All results are written
 * into `vars_on_lower_face` and `vars_on_upper_face`.
 */
template <typename PrimsTags, typename TagsList, size_t Dim,
          size_t ThermodynamicDim, typename F>
void reconstruct_prims_work(
    gsl::not_null<std::array<Variables<TagsList>, Dim>*> vars_on_lower_face,
    gsl::not_null<std::array<Variables<TagsList>, Dim>*> vars_on_upper_face,
    const F& reconstruct, const Variables<PrimsTags>& volume_prims,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
    const Element<Dim>& element,
    const FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
                       std::pair<Direction<Dim>, ElementId<Dim>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
        neighbor_data,
    const Mesh<Dim>& subcell_mesh, size_t ghost_zone_size);

/*!
 * \brief Reconstructs the mass density, velocity, and pressure, then computes
 * the specific internal energy and conserved variables. All results are written
 * into `vars_on_face`.
 *
 * This is used on DG elements to reconstruct their subcell neighbors' solution
 * on the shared faces.
 */
template <typename TagsList, typename PrimsTags, size_t Dim,
          size_t ThermodynamicDim, typename F0, typename F1>
void reconstruct_fd_neighbor_work(
    gsl::not_null<Variables<TagsList>*> vars_on_face,
    const F0& reconstruct_lower_neighbor, const F1& reconstruct_upper_neighbor,
    const Variables<PrimsTags>& subcell_volume_prims,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
    const Element<Dim>& element,
    const FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
                       std::pair<Direction<Dim>, ElementId<Dim>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
        neighbor_data,
    const Mesh<Dim>& subcell_mesh,
    const Direction<Dim>& direction_to_reconstruct, size_t ghost_zone_size);
}  // namespace NewtonianEuler::fd
