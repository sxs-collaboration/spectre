// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
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
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianEuler::fd {
template <typename PrimsTags, typename TagsList, size_t Dim,
          size_t ThermodynamicDim, typename F>
void reconstruct_prims_work(
    const gsl::not_null<std::array<Variables<TagsList>, Dim>*>
        vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, Dim>*>
        vars_on_upper_face,
    const F& reconstruct, const Variables<PrimsTags>& volume_prims,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
    const Element<Dim>& element,
    const FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
                       std::pair<Direction<Dim>, ElementId<Dim>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
        neighbor_data,
    const Mesh<Dim>& subcell_mesh, const size_t ghost_zone_size) {
  // Conservative vars tags
  using MassDensityCons = Tags::MassDensityCons;
  using EnergyDensity = Tags::EnergyDensity;
  using MomentumDensity = Tags::MomentumDensity<Dim>;

  // Primitive vars tags
  using MassDensity = Tags::MassDensity<DataVector>;
  using Velocity = Tags::Velocity<DataVector, Dim>;
  using SpecificInternalEnergy = Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = Tags::Pressure<DataVector>;

  using prim_tags_for_reconstruction =
      tmpl::list<MassDensity, Velocity, Pressure>;

  ASSERT(Mesh<Dim>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                   subcell_mesh.quadrature(0)) == subcell_mesh,
         "The subcell mesh should be isotropic but got " << subcell_mesh);
  const size_t volume_num_pts = subcell_mesh.number_of_grid_points();
  const size_t reconstructed_num_pts =
      (subcell_mesh.extents(0) + 1) *
      subcell_mesh.extents().slice_away(0).product();
  const size_t neighbor_num_pts =
      ghost_zone_size * subcell_mesh.extents().slice_away(0).product();
  size_t vars_in_neighbor_count = 0;
  tmpl::for_each<prim_tags_for_reconstruction>(
      [&element, &neighbor_data, neighbor_num_pts, &reconstruct,
       reconstructed_num_pts, volume_num_pts, &volume_prims,
       &vars_in_neighbor_count, &vars_on_lower_face, &vars_on_upper_face,
       &subcell_mesh](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        auto& volume_tensor = get<tag>(volume_prims);

        const size_t number_of_components = volume_tensor.size();
        const gsl::span<const double> volume_vars = gsl::make_span(
            volume_tensor[0].data(), number_of_components * volume_num_pts);
        std::array<gsl::span<double>, Dim> upper_face_vars{};
        std::array<gsl::span<double>, Dim> lower_face_vars{};
        for (size_t i = 0; i < Dim; ++i) {
          gsl::at(upper_face_vars, i) = gsl::make_span(
              get<tag>(gsl::at(*vars_on_upper_face, i))[0].data(),
              number_of_components * reconstructed_num_pts);
          gsl::at(lower_face_vars, i) = gsl::make_span(
              get<tag>(gsl::at(*vars_on_lower_face, i))[0].data(),
              number_of_components * reconstructed_num_pts);
        }

        DirectionMap<Dim, gsl::span<const double>> ghost_cell_vars{};
        for (const auto& direction : Direction<Dim>::all_directions()) {
          const auto& neighbors_in_direction =
              element.neighbors().at(direction);
          ASSERT(neighbors_in_direction.size() == 1,
                 "Currently only support one neighbor in each direction, but "
                 "got "
                     << neighbors_in_direction.size() << " in direction "
                     << direction);
          ghost_cell_vars[direction] = gsl::make_span(
              &neighbor_data
                   .at(std::pair{direction, *neighbors_in_direction.begin()})
                   .data_for_reconstruction[vars_in_neighbor_count *
                                            neighbor_num_pts],
              number_of_components * neighbor_num_pts);
        }

        reconstruct(make_not_null(&upper_face_vars),
                    make_not_null(&lower_face_vars), volume_vars,
                    ghost_cell_vars, subcell_mesh.extents(),
                    number_of_components);

        vars_in_neighbor_count += number_of_components;
      });

  for (size_t i = 0; i < Dim; ++i) {
    auto& vars_upper_face = gsl::at(*vars_on_upper_face, i);
    auto& vars_lower_face = gsl::at(*vars_on_lower_face, i);

    if constexpr (ThermodynamicDim == 2) {
      get<SpecificInternalEnergy>(vars_upper_face) =
          eos.specific_internal_energy_from_density_and_pressure(
              get<MassDensity>(vars_upper_face),
              get<Pressure>(vars_upper_face));
      get<SpecificInternalEnergy>(vars_lower_face) =
          eos.specific_internal_energy_from_density_and_pressure(
              get<MassDensity>(vars_lower_face),
              get<Pressure>(vars_lower_face));
    } else {
      get<SpecificInternalEnergy>(vars_upper_face) =
          eos.specific_internal_energy_from_density(
              get<MassDensity>(vars_upper_face));
      get<SpecificInternalEnergy>(vars_lower_face) =
          eos.specific_internal_energy_from_density(
              get<MassDensity>(vars_lower_face));
    }

    // Compute conserved variables on faces
    NewtonianEuler::ConservativeFromPrimitive<Dim>::apply(
        make_not_null(&get<MassDensityCons>(vars_upper_face)),
        make_not_null(&get<MomentumDensity>(vars_upper_face)),
        make_not_null(&get<EnergyDensity>(vars_upper_face)),
        get<MassDensity>(vars_upper_face), get<Velocity>(vars_upper_face),
        get<SpecificInternalEnergy>(vars_upper_face));
    NewtonianEuler::ConservativeFromPrimitive<Dim>::apply(
        make_not_null(&get<MassDensityCons>(vars_lower_face)),
        make_not_null(&get<MomentumDensity>(vars_lower_face)),
        make_not_null(&get<EnergyDensity>(vars_lower_face)),
        get<MassDensity>(vars_lower_face), get<Velocity>(vars_lower_face),
        get<SpecificInternalEnergy>(vars_lower_face));
  }
}

template <typename TagsList, typename PrimsTags, size_t Dim,
          size_t ThermodynamicDim, typename F0, typename F1>
void reconstruct_fd_neighbor_work(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
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
    const Direction<Dim>& direction_to_reconstruct,
    const size_t ghost_zone_size) {
  // Conservative vars tags
  using MassDensityCons = Tags::MassDensityCons;
  using EnergyDensity = Tags::EnergyDensity;
  using MomentumDensity = Tags::MomentumDensity<Dim>;

  // Primitive vars tags
  using MassDensity = Tags::MassDensity<DataVector>;
  using Velocity = Tags::Velocity<DataVector, Dim>;
  using SpecificInternalEnergy = Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = Tags::Pressure<DataVector>;

  using prim_tags_for_reconstruction =
      tmpl::list<MassDensity, Velocity, Pressure>;

  const std::pair mortar_id{
      direction_to_reconstruct,
      *element.neighbors().at(direction_to_reconstruct).begin()};
  Index<Dim> ghost_data_extents = subcell_mesh.extents();
  ghost_data_extents[direction_to_reconstruct.dimension()] = ghost_zone_size;
  Variables<prim_tags_for_reconstruction> neighbor_prims{
      ghost_data_extents.product()};
  {
    ASSERT(neighbor_data.contains(mortar_id),
           "The neighbor data does not contain the mortar: ("
               << mortar_id.first << ',' << mortar_id.second << ")");
    const auto& neighbor_data_in_direction = neighbor_data.at(mortar_id);
    std::copy(
        neighbor_data_in_direction.data_for_reconstruction.begin(),
        std::next(neighbor_data_in_direction.data_for_reconstruction.begin(),
                  static_cast<std::ptrdiff_t>(
                      neighbor_prims.number_of_independent_components *
                      ghost_data_extents.product())),
        neighbor_prims.data());
  }

  tmpl::for_each<prim_tags_for_reconstruction>(
      [&direction_to_reconstruct, &ghost_data_extents, &neighbor_prims,
       &reconstruct_lower_neighbor, &reconstruct_upper_neighbor, &subcell_mesh,
       &subcell_volume_prims, &vars_on_face](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        const auto& tensor_volume = get<tag>(subcell_volume_prims);
        const auto& tensor_neighbor = get<tag>(neighbor_prims);
        auto& tensor_on_face = get<tag>(*vars_on_face);
        if (direction_to_reconstruct.side() == Side::Upper) {
          for (size_t tensor_index = 0; tensor_index < tensor_on_face.size();
               ++tensor_index) {
            reconstruct_upper_neighbor(
                make_not_null(&tensor_on_face[tensor_index]),
                tensor_volume[tensor_index], tensor_neighbor[tensor_index],
                subcell_mesh.extents(), ghost_data_extents,
                direction_to_reconstruct);
          }
        } else {
          for (size_t tensor_index = 0; tensor_index < tensor_on_face.size();
               ++tensor_index) {
            reconstruct_lower_neighbor(
                make_not_null(&tensor_on_face[tensor_index]),
                tensor_volume[tensor_index], tensor_neighbor[tensor_index],
                subcell_mesh.extents(), ghost_data_extents,
                direction_to_reconstruct);
          }
        }
      });

  if constexpr (ThermodynamicDim == 2) {
    get<SpecificInternalEnergy>(*vars_on_face) =
        eos.specific_internal_energy_from_density_and_pressure(
            get<MassDensity>(*vars_on_face), get<Pressure>(*vars_on_face));
  } else {
    get<SpecificInternalEnergy>(*vars_on_face) =
        eos.specific_internal_energy_from_density(
            get<MassDensity>(*vars_on_face));
  }
  NewtonianEuler::ConservativeFromPrimitive<Dim>::apply(
      make_not_null(&get<MassDensityCons>(*vars_on_face)),
      make_not_null(&get<MomentumDensity>(*vars_on_face)),
      make_not_null(&get<EnergyDensity>(*vars_on_face)),
      get<MassDensity>(*vars_on_face), get<Velocity>(*vars_on_face),
      get<SpecificInternalEnergy>(*vars_on_face));
}
}  // namespace NewtonianEuler::fd
