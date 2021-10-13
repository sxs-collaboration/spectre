// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers::fd {
template <typename TagsList, typename Reconstructor>
void reconstruct_work(
    const gsl::not_null<std::array<Variables<TagsList>, 1>*> vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, 1>*> vars_on_upper_face,
    const Reconstructor& reconstruct,
    const Variables<tmpl::list<Tags::U>> volume_vars, const Element<1>& element,
    const FixedHashMap<maximum_number_of_neighbors(1) + 1,
                       std::pair<Direction<1>, ElementId<1>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<1>, ElementId<1>>>>
        neighbor_data,
    const Mesh<1>& subcell_mesh, const size_t ghost_zone_size) {
  const size_t volume_num_pts = subcell_mesh.number_of_grid_points();
  const size_t reconstructed_num_pts = volume_num_pts + 1;

  // create views/spans into the face data, which will be filled by the
  // reconstructor
  std::array<gsl::span<double>, 1> upper_face_vars{};
  std::array<gsl::span<double>, 1> lower_face_vars{};
  for (size_t i = 0; i < 1; i++) {
    gsl::at(upper_face_vars, i) =
        gsl::make_span(get<Tags::U>(gsl::at(*vars_on_upper_face, i))[0].data(),
                       reconstructed_num_pts);
    gsl::at(lower_face_vars, i) =
        gsl::make_span(get<Tags::U>(gsl::at(*vars_on_lower_face, i))[0].data(),
                       reconstructed_num_pts);
  }

  const size_t neighbor_num_ghost_fd_points = ghost_zone_size;

  // make span of ghost cell variables for each direction
  DirectionMap<1, gsl::span<const double>> ghost_cell_vars{};

  // for all Directions, make the pointers in ghost_cell_vars span to point
  // the received neighbor data
  for (const auto& direction : Direction<1>::all_directions()) {
    const auto& neighbors_in_direction = element.neighbors().at(direction);

    ASSERT(neighbors_in_direction.size() == 1,
           "Currently only support one neighbor in each direction, but "
           "got "
               << neighbors_in_direction.size() << " in direction "
               << direction);

    ghost_cell_vars[direction] = gsl::make_span(
        &neighbor_data.at(std::pair{direction, *neighbors_in_direction.begin()})
             .data_for_reconstruction[0],
        neighbor_num_ghost_fd_points);
  }

  // make span of volume variables
  auto& volume_tensor = get<Tags::U>(volume_vars);
  const gsl::span<const double> volume_vars_span =
      gsl::make_span((volume_tensor)[0].data(), volume_num_pts);

  // perform reconstruction
  reconstruct(make_not_null(&upper_face_vars), make_not_null(&lower_face_vars),
              volume_vars_span, ghost_cell_vars, subcell_mesh.extents(), 1);
}

template <typename TagsList, typename ReconstructLower,
          typename ReconstructUpper>
void reconstruct_fd_neighbor_work(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
    const ReconstructLower& reconstruct_lower_neighbor,
    const ReconstructUpper& reconstruct_upper_neighbor,
    const Variables<tmpl::list<Tags::U>>& subcell_volume_vars,
    const Element<1>& element,
    const FixedHashMap<maximum_number_of_neighbors(1) + 1,
                       std::pair<Direction<1>, ElementId<1>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<1>, ElementId<1>>>>
        neighbor_data,
    const Mesh<1>& subcell_mesh, const Direction<1>& direction_to_reconstruct,
    const size_t ghost_zone_size) {
  const std::pair mortar_id{
      direction_to_reconstruct,
      *element.neighbors().at(direction_to_reconstruct).begin()};

  Index<1> ghost_data_extents = subcell_mesh.extents();
  ghost_data_extents[direction_to_reconstruct.dimension()] = ghost_zone_size;

  // allocate Variable for storing data from neighbor
  Variables<tmpl::list<Tags::U>> neighbor_vars{ghost_data_extents.product()};

  {
    ASSERT(neighbor_data.contains(mortar_id),
           "The neighbor data does not contain the mortar: ("
               << mortar_id.first << ',' << mortar_id.second << ")");

    const auto& neighbor_data_in_direction = neighbor_data.at(mortar_id);
    std::copy(
        neighbor_data_in_direction.data_for_reconstruction.begin(),
        std::next(neighbor_data_in_direction.data_for_reconstruction.begin(),
                  static_cast<std::ptrdiff_t>(ghost_data_extents.product())),
        neighbor_vars.data());
  }

  const auto& tensor_volume = get<Tags::U>(subcell_volume_vars);
  const auto& tensor_neighbor = get<Tags::U>(neighbor_vars);
  auto& tensor_on_face = get<Tags::U>(*vars_on_face);

  if (direction_to_reconstruct.side() == Side::Upper) {
    for (size_t tensor_index = 0; tensor_index < tensor_on_face.size();
         ++tensor_index) {
      reconstruct_upper_neighbor(
          make_not_null(&tensor_on_face[tensor_index]),
          tensor_volume[tensor_index], tensor_neighbor[tensor_index],
          subcell_mesh.extents(), ghost_data_extents, direction_to_reconstruct);
    }
  } else {
    for (size_t tensor_index = 0; tensor_index < tensor_on_face.size();
         ++tensor_index) {
      reconstruct_lower_neighbor(
          make_not_null(&tensor_on_face[tensor_index]),
          tensor_volume[tensor_index], tensor_neighbor[tensor_index],
          subcell_mesh.extents(), ghost_data_extents, direction_to_reconstruct);
    }
  }
}
}  // namespace Burgers::fd
