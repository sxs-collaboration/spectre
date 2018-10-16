// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/RelativisticEuler/Valencia/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
template <class>
class Variables;
}  // namespace Tags

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to evolving the relativistic Euler system
namespace RelativisticEuler {
/// \brief The Valencia formulation of the relativistic Euler System
/// See Chapter 7 of Relativistic Hydrodynamics by Luciano Rezzolla and Olindo
/// Zanotti or http://iopscience.iop.org/article/10.1086/303604
namespace Valencia {

template <size_t Dim>
struct System {
  static constexpr bool is_in_flux_conservative_form = true;
  static constexpr bool has_primitive_and_conservative_vars = true;
  static constexpr size_t volume_dim = Dim;

  using variables_tag =
      Tags::Variables<tmpl::list<TildeD, TildeTau, TildeS<Dim>>>;
};

}  // namespace Valencia
}  // namespace RelativisticEuler
