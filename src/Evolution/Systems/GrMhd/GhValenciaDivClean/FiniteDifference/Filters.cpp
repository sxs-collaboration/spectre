// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Filters.hpp"

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/FillNeighborSpacetimeVariables.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "NumericalAlgorithms/FiniteDifference/Filter.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::GhValenciaDivClean::fd {
template <typename VariableTags>
void spacetime_kreiss_oliger_filter(
    const gsl::not_null<Variables<VariableTags>*> result,
    const Variables<VariableTags>& volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const Mesh<3>& volume_mesh, const size_t order, const double epsilon) {
  if (UNLIKELY(result->number_of_grid_points() !=
               volume_evolved_variables.number_of_grid_points())) {
    result->initialize(volume_evolved_variables.number_of_grid_points());
  }

  constexpr size_t number_of_gh_components = Variables<
      grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags>::
      number_of_independent_components;

  DirectionMap<3, gsl::span<const double>> ghost_cell_spacetime_vars{};
  using NeighborVariables =
      Variables<grmhd::GhValenciaDivClean::Tags::
                    primitive_grmhd_and_spacetime_reconstruction_tags>;
  using FirstGhTag = tmpl::front<
      grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags>;

  fill_neighbor_spacetime_variables<NeighborVariables, FirstGhTag>(
      make_not_null(&ghost_cell_spacetime_vars), all_ghost_data,
      number_of_gh_components);

  const auto volume_gh_vars =
      gsl::make_span(get<FirstGhTag>(volume_evolved_variables)[0].data(),
                     number_of_gh_components *
                         volume_evolved_variables.number_of_grid_points());

  auto filtered_gh_vars =
      gsl::make_span(get<FirstGhTag>(*result)[0].data(),
                     number_of_gh_components *
                         volume_evolved_variables.number_of_grid_points());
  ::fd::kreiss_oliger_filter(make_not_null(&filtered_gh_vars), volume_gh_vars,
                             ghost_cell_spacetime_vars, volume_mesh,
                             number_of_gh_components, order, epsilon);
}

#define NEUTRINO(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                             \
  template void spacetime_kreiss_oliger_filter(                            \
      const gsl::not_null<                                                 \
          Variables<typename grmhd::GhValenciaDivClean::System<NEUTRINO(   \
              data)>::variables_tag::tags_list>*>                          \
          result,                                                          \
      const Variables<typename grmhd::GhValenciaDivClean::System<NEUTRINO( \
          data)>::variables_tag::tags_list>& volume_evolved_variables,     \
      const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&        \
          all_ghost_data,                                                  \
      const Mesh<3>& volume_mesh, const size_t order, const double epsilon);

GENERATE_INSTANTIATIONS(INSTANTIATION,
                        (RadiationTransport::NoNeutrinos::System))

#undef INSTANTIATION
#undef NEUTRINO

}  // namespace grmhd::GhValenciaDivClean::fd
