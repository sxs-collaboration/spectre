// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructWork.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/Reconstruct.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::fd {
template <typename SpacetimeTagsToReconstruct,
          typename PrimTagsForReconstruction, typename PrimsTags,
          typename SpacetimeAndConsTags, typename TagsList,
          size_t ThermodynamicDim, typename HydroReconstructor,
          typename SpacetimeReconstructor,
          typename ComputeGrmhdSpacetimeVarsFromReconstructedSpacetimeTags,
          typename PrimsTagsSentByNeighbor>
void reconstruct_prims_work(
    const gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_upper_face,
    const HydroReconstructor& hydro_reconstructor,
    const SpacetimeReconstructor& spacetime_reconstructor,
    const ComputeGrmhdSpacetimeVarsFromReconstructedSpacetimeTags&
        spacetime_vars_for_grmhd,
    const Variables<PrimsTags>& volume_prims,
    const Variables<SpacetimeAndConsTags>& volume_spacetime_and_cons_vars,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        Variables<PrimsTagsSentByNeighbor>,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& neighbor_data,
    const Mesh<3>& subcell_mesh, const size_t ghost_zone_size,
    const bool compute_conservatives) {
  ASSERT(Mesh<3>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                 subcell_mesh.quadrature(0)) == subcell_mesh,
         "The subcell mesh should be isotropic but got " << subcell_mesh);
  const size_t volume_num_pts = subcell_mesh.number_of_grid_points();
  const size_t reconstructed_num_pts =
      (subcell_mesh.extents(0) + 1) *
      subcell_mesh.extents().slice_away(0).product();
  const size_t neighbor_num_pts =
      ghost_zone_size * subcell_mesh.extents().slice_away(0).product();
  size_t vars_in_neighbor_count = 0;
  tmpl::for_each<PrimTagsForReconstruction>([&element, &neighbor_data,
                                             neighbor_num_pts,
                                             &hydro_reconstructor,
                                             reconstructed_num_pts,
                                             volume_num_pts, &volume_prims,
                                             &vars_in_neighbor_count,
                                             &vars_on_lower_face,
                                             &vars_on_upper_face,
                                             &subcell_mesh](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    const typename tag::type* volume_tensor_ptr = nullptr;
    Variables<tmpl::list<
        hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>>
        lorentz_factor_times_v_I{};
    if constexpr (std::is_same_v<tag,
                                 hydro::Tags::LorentzFactorTimesSpatialVelocity<
                                     DataVector, 3>>) {
      // we need to handle the Wv^i reconstruction separately since we need to
      // first compute Wv^i in the volume (it's not one of our primitives from
      // the recovery). The components need to be stored contiguously, which is
      // why we have the Variables `lorentz_factor_times_v_I`
      const auto& spatial_velocity =
          get<hydro::Tags::SpatialVelocity<DataVector, 3>>(volume_prims);
      const auto& lorentz_factor =
          get<hydro::Tags::LorentzFactor<DataVector>>(volume_prims);
      lorentz_factor_times_v_I.initialize(get(lorentz_factor).size());
      auto& volume_tensor =
          get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
              lorentz_factor_times_v_I) = spatial_velocity;
      for (size_t i = 0; i < 3; ++i) {
        volume_tensor.get(i) *= get(lorentz_factor);
      }
      volume_tensor_ptr = &volume_tensor;
    } else {
      volume_tensor_ptr = &get<tag>(volume_prims);
    }

    const size_t number_of_variables = volume_tensor_ptr->size();
    const gsl::span<const double> volume_vars = gsl::make_span(
        (*volume_tensor_ptr)[0].data(), number_of_variables * volume_num_pts);
    std::array<gsl::span<double>, 3> upper_face_vars{};
    std::array<gsl::span<double>, 3> lower_face_vars{};
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(upper_face_vars, i) =
          gsl::make_span(get<tag>(gsl::at(*vars_on_upper_face, i))[0].data(),
                         number_of_variables * reconstructed_num_pts);
      gsl::at(lower_face_vars, i) =
          gsl::make_span(get<tag>(gsl::at(*vars_on_lower_face, i))[0].data(),
                         number_of_variables * reconstructed_num_pts);
    }

    DirectionMap<3, gsl::span<const double>> ghost_cell_vars{};
    for (const auto& direction : Direction<3>::all_directions()) {
      if (element.neighbors().contains(direction)) {
        const auto& neighbors_in_direction = element.neighbors().at(direction);
        ASSERT(neighbors_in_direction.size() == 1,
               "Currently only support one neighbor in each direction, but "
               "got "
                   << neighbors_in_direction.size() << " in direction "
                   << direction);
        ghost_cell_vars[direction] =
            gsl::make_span(get<tag>(neighbor_data.at(std::pair{
                               direction, *neighbors_in_direction.begin()}))[0]
                               .data(),
                           number_of_variables * neighbor_num_pts);
      } else {
        // retrieve boundary ghost data from neighbor_data
        ASSERT(
            element.external_boundaries().count(direction) == 1,
            "Element has neither neighbor nor external boundary to direction : "
                << direction);
        ghost_cell_vars[direction] = gsl::make_span(
            get<tag>(neighbor_data.at(
                std::pair{direction, ElementId<3>::external_boundary_id()}))[0]
                .data(),
            number_of_variables * neighbor_num_pts);
      }
    }

    hydro_reconstructor(make_not_null(&upper_face_vars),
                        make_not_null(&lower_face_vars), volume_vars,
                        ghost_cell_vars, subcell_mesh.extents(),
                        number_of_variables);

    vars_in_neighbor_count += number_of_variables;
  });
  tmpl::for_each<SpacetimeTagsToReconstruct>(
      [&element, &neighbor_data, neighbor_num_pts, &spacetime_reconstructor,
       reconstructed_num_pts, volume_num_pts, &volume_spacetime_and_cons_vars,
       &vars_in_neighbor_count, &vars_on_lower_face, &vars_on_upper_face,
       &subcell_mesh](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        const typename tag::type& volume_tensor =
            get<tag>(volume_spacetime_and_cons_vars);

        const size_t number_of_variables = volume_tensor.size();
        const gsl::span<const double> volume_vars = gsl::make_span(
            (volume_tensor)[0].data(), number_of_variables * volume_num_pts);
        std::array<gsl::span<double>, 3> upper_face_vars{};
        std::array<gsl::span<double>, 3> lower_face_vars{};
        for (size_t i = 0; i < 3; ++i) {
          gsl::at(upper_face_vars, i) = gsl::make_span(
              get<tag>(gsl::at(*vars_on_upper_face, i))[0].data(),
              number_of_variables * reconstructed_num_pts);
          gsl::at(lower_face_vars, i) = gsl::make_span(
              get<tag>(gsl::at(*vars_on_lower_face, i))[0].data(),
              number_of_variables * reconstructed_num_pts);
        }

        DirectionMap<3, gsl::span<const double>> ghost_cell_vars{};
        for (const auto& direction : Direction<3>::all_directions()) {
          if (element.neighbors().contains(direction)) {
            const auto& neighbors_in_direction =
                element.neighbors().at(direction);
            ASSERT(neighbors_in_direction.size() == 1,
                   "Currently only support one neighbor in each direction, but "
                   "got "
                       << neighbors_in_direction.size() << " in direction "
                       << direction);
            ghost_cell_vars[direction] = gsl::make_span(
                get<tag>(neighbor_data.at(
                    std::pair{direction, *neighbors_in_direction.begin()}))[0]
                    .data(),
                number_of_variables * neighbor_num_pts);
          } else {
            // retrieve boundary ghost data from neighbor_data
            ASSERT(element.external_boundaries().count(direction) == 1,
                   "Element has neither neighbor nor external boundary to "
                   "direction : "
                       << direction);
            ghost_cell_vars[direction] = gsl::make_span(
                get<tag>(neighbor_data.at(std::pair{
                    direction, ElementId<3>::external_boundary_id()}))[0]
                    .data(),
                number_of_variables * neighbor_num_pts);
          }
        }

        spacetime_reconstructor(make_not_null(&upper_face_vars),
                                make_not_null(&lower_face_vars), volume_vars,
                                ghost_cell_vars, subcell_mesh.extents(),
                                number_of_variables);

        vars_in_neighbor_count += number_of_variables;
      });

  for (size_t i = 0; i < 3; ++i) {
    if constexpr (tmpl::size<SpacetimeTagsToReconstruct>::value != 0) {
      spacetime_vars_for_grmhd(make_not_null(&gsl::at(*vars_on_lower_face, i)));
      spacetime_vars_for_grmhd(make_not_null(&gsl::at(*vars_on_upper_face, i)));
    }

    if (compute_conservatives) {
      ValenciaDivClean::fd::compute_conservatives_for_reconstruction(
          make_not_null(&gsl::at(*vars_on_lower_face, i)), eos);
      ValenciaDivClean::fd::compute_conservatives_for_reconstruction(
          make_not_null(&gsl::at(*vars_on_upper_face, i)), eos);
    }
  }
}

template <
    typename SpacetimeTagsToReconstruct, typename PrimTagsForReconstruction,
    typename PrimsTagsSentByNeighbor, typename TagsList, typename PrimsTags,
    size_t ThermodynamicDim, typename LowerHydroReconstructor,
    typename LowerSpacetimeReconstructor, typename UpperHydroReconstructor,
    typename UpperSpacetimeReconstructor,
    typename ComputeGrmhdSpacetimeVarsFromReconstructedSpacetimeTags>
void reconstruct_fd_neighbor_work(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
    const LowerHydroReconstructor& reconstruct_lower_neighbor_hydro,
    const LowerSpacetimeReconstructor& reconstruct_lower_neighbor_spacetime,
    const UpperHydroReconstructor& reconstruct_upper_neighbor_hydro,
    const UpperSpacetimeReconstructor& reconstruct_upper_neighbor_spacetime,
    const ComputeGrmhdSpacetimeVarsFromReconstructedSpacetimeTags&
        spacetime_vars_for_grmhd,
    const Variables<PrimsTags>& subcell_volume_prims,
    const Variables<
        grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags>&
        subcell_volume_spacetime_vars,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        evolution::dg::subcell::GhostData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& ghost_data,
    const Mesh<3>& subcell_mesh, const Direction<3>& direction_to_reconstruct,
    const size_t ghost_zone_size, const bool compute_conservatives) {
  const std::pair mortar_id{
      direction_to_reconstruct,
      *element.neighbors().at(direction_to_reconstruct).begin()};
  Index<3> ghost_data_extents = subcell_mesh.extents();
  ghost_data_extents[direction_to_reconstruct.dimension()] = ghost_zone_size;
  Variables<PrimsTagsSentByNeighbor> neighbor_prims{
      ghost_data_extents.product()};
  {
    ASSERT(ghost_data.contains(mortar_id),
           "The neighbor data does not contain the mortar: ("
               << mortar_id.first << ',' << mortar_id.second << ")");
    const DataVector& neighbor_data_on_mortar =
        ghost_data.at(mortar_id).neighbor_ghost_data_for_reconstruction();
    std::copy(neighbor_data_on_mortar.begin(),
              std::next(neighbor_data_on_mortar.begin(),
                        static_cast<std::ptrdiff_t>(
                            neighbor_prims.number_of_independent_components *
                            ghost_data_extents.product())),
              neighbor_prims.data());
  }

  tmpl::for_each<PrimTagsForReconstruction>(
      [&direction_to_reconstruct, &ghost_data_extents, &neighbor_prims,
       &reconstruct_lower_neighbor_hydro, &reconstruct_upper_neighbor_hydro,
       &subcell_mesh, &subcell_volume_prims, &vars_on_face](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        const typename tag::type* volume_tensor_ptr = nullptr;
        typename tag::type volume_tensor{};
        if constexpr (std::is_same_v<
                          tag, hydro::Tags::LorentzFactorTimesSpatialVelocity<
                                   DataVector, 3>>) {
          // we need to handle the Wv^i reconstruction separately since we need
          // to first compute Wv^i in the volume (it's not one of our primitives
          // from the recovery). The components need to be stored contiguously,
          // which is why we have the Variables `lorentz_factor_times_v_I`
          const auto& spatial_velocity =
              get<hydro::Tags::SpatialVelocity<DataVector, 3>>(
                  subcell_volume_prims);
          const auto& lorentz_factor =
              get<hydro::Tags::LorentzFactor<DataVector>>(subcell_volume_prims);
          volume_tensor = spatial_velocity;
          for (size_t i = 0; i < 3; ++i) {
            volume_tensor.get(i) *= get(lorentz_factor);
          }
          volume_tensor_ptr = &volume_tensor;
        } else {
          volume_tensor_ptr = &get<tag>(subcell_volume_prims);
        }

        const auto& tensor_neighbor = get<tag>(neighbor_prims);
        auto& tensor_on_face = get<tag>(*vars_on_face);
        if (direction_to_reconstruct.side() == Side::Upper) {
          for (size_t tensor_index = 0; tensor_index < tensor_on_face.size();
               ++tensor_index) {
            reconstruct_upper_neighbor_hydro(
                make_not_null(&tensor_on_face[tensor_index]),
                (*volume_tensor_ptr)[tensor_index],
                tensor_neighbor[tensor_index], subcell_mesh.extents(),
                ghost_data_extents, direction_to_reconstruct);
          }
        } else {
          for (size_t tensor_index = 0; tensor_index < tensor_on_face.size();
               ++tensor_index) {
            reconstruct_lower_neighbor_hydro(
                make_not_null(&tensor_on_face[tensor_index]),
                (*volume_tensor_ptr)[tensor_index],
                tensor_neighbor[tensor_index], subcell_mesh.extents(),
                ghost_data_extents, direction_to_reconstruct);
          }
        }
      });

  tmpl::for_each<SpacetimeTagsToReconstruct>(
      [&direction_to_reconstruct, &ghost_data_extents, &neighbor_prims,
       &reconstruct_lower_neighbor_spacetime,
       &reconstruct_upper_neighbor_spacetime, &subcell_mesh,
       &subcell_volume_spacetime_vars, &vars_on_face](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        const typename tag::type volume_tensor =
            get<tag>(subcell_volume_spacetime_vars);

        const auto& tensor_neighbor = get<tag>(neighbor_prims);
        auto& tensor_on_face = get<tag>(*vars_on_face);
        if (direction_to_reconstruct.side() == Side::Upper) {
          for (size_t tensor_index = 0; tensor_index < tensor_on_face.size();
               ++tensor_index) {
            reconstruct_upper_neighbor_spacetime(
                make_not_null(&tensor_on_face[tensor_index]),
                volume_tensor[tensor_index], tensor_neighbor[tensor_index],
                subcell_mesh.extents(), ghost_data_extents,
                direction_to_reconstruct);
          }
        } else {
          for (size_t tensor_index = 0; tensor_index < tensor_on_face.size();
               ++tensor_index) {
            reconstruct_lower_neighbor_spacetime(
                make_not_null(&tensor_on_face[tensor_index]),
                volume_tensor[tensor_index], tensor_neighbor[tensor_index],
                subcell_mesh.extents(), ghost_data_extents,
                direction_to_reconstruct);
          }
        }
      });

  if constexpr (tmpl::size<SpacetimeTagsToReconstruct>::value != 0) {
    spacetime_vars_for_grmhd(vars_on_face);
  }
  if (compute_conservatives) {
    ValenciaDivClean::fd::compute_conservatives_for_reconstruction(vars_on_face,
                                                                   eos);
  }
}
}  // namespace grmhd::GhValenciaDivClean::fd
