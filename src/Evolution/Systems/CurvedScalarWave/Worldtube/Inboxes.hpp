// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <unordered_map>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Grid;
}  // namespace Frame
/// \endcond

namespace CurvedScalarWave::Worldtube::Tags {
/*!
 * \brief Inbox of the worldtube singleton chare which receives quantities
 * projected onto spherical harmonics.
 *
 * \details Each element abutting the worldtube projects both $\Psi$ and the
 * time derivative of $\partial_t Psi$ onto spherical harmonics by integrating
 * it over the worldtube surface/boundary it is touching. These coefficients are
 * sent to this inbox.
 */
template <size_t Dim>
struct SphericalHarmonicsInbox
    : Parallel::InboxInserters::Map<SphericalHarmonicsInbox<Dim>> {
  using temporal_id = TimeStepId;
  using tags_list = tmpl::list<CurvedScalarWave::Tags::Psi,
                               ::Tags::dt<CurvedScalarWave::Tags::Psi>>;
  using type =
      std::map<temporal_id,
               std::unordered_map<ElementId<Dim>, Variables<tags_list>>>;
};

/*!
 * \brief Inbox of the element chares that contains the regular field $\Psi^R$
 * as well as its time and spatial derivative evaluated at the grid points of
 * abutting element faces.
 */
template <size_t Dim>
struct RegularFieldInbox
    : Parallel::InboxInserters::Value<RegularFieldInbox<Dim>> {
  using tags_to_send =
      tmpl::list<CurvedScalarWave::Tags::Psi,
                 ::Tags::dt<CurvedScalarWave::Tags::Psi>,
                 ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<Dim>,
                               Frame::Grid>>;
  using temporal_id = TimeStepId;
  using type = std::map<temporal_id, Variables<tags_to_send>>;
};
}  // namespace CurvedScalarWave::Worldtube::Tags
