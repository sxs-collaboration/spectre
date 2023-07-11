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
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::fd {

template <typename TagsList, typename Reconstructor>
void reconstruct_work(
    const gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_upper_face,
    const Reconstructor& reconstruct,
    const Variables<System::variables_tag::tags_list>& volume_evolved_vars,
    const tnsr::I<DataVector, 3, Frame::Inertial>& volume_tilde_j,
    const Element<3>& element,
    const FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        evolution::dg::subcell::GhostData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& neighbor_data,
    const Mesh<3>& subcell_mesh, const size_t ghost_zone_size) {
  ASSERT(Mesh<3>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                 subcell_mesh.quadrature(0)) == subcell_mesh,
         "The subcell mesh should be isotropic but got " << subcell_mesh);

  const size_t volume_num_pts = subcell_mesh.number_of_grid_points();
  const size_t reconstructed_num_pts =
      (subcell_mesh.extents(0) + 1) *
      subcell_mesh.extents().slice_away(0).product();
  const size_t neighbor_num_pts =
      ghost_zone_size * subcell_mesh.extents().slice_away(0).product();

  // We reconstruct the evolved variables (TildeE, TildeB, TildePsi, TildePhi,
  // TildeQ) which are contained in the function argument `volume_evolved_vars`,
  // and also reconstruct the generalized current density TildeJ which is passed
  // as the function argument `volume_tilde_j`.
  //
  // Core FD reconstruction routines (from NumericalAlgorithms) requires the
  // input data for reconstruction to be consecutive on memory. This is
  // true for `volume_evolved_vars` but not true for `volume_tilde_j` which is a
  // SpECTRE tensor. Therefore we need to allocate a single block of data
  // storing the values of TildeJ by means of a Variables object.
  using VarTildeJ = Variables<tmpl::list<ForceFree::Tags::TildeJ>>;
  const VarTildeJ var_tilde_j = [&volume_num_pts, &volume_tilde_j]() {
    DataVector buffer{VarTildeJ::number_of_independent_components *
                      volume_num_pts};
    VarTildeJ var{buffer.data(), buffer.size()};
    for (size_t i = 0; i < 3; ++i) {
      get<ForceFree::Tags::TildeJ>(var).get(i) = volume_tilde_j.get(i);
    }
    return var;
  }();

  size_t vars_in_neighbor_count = 0;
  tmpl::for_each<
      ForceFree::fd::tags_list_for_reconstruction>([&element, &neighbor_data,
                                                    neighbor_num_pts,
                                                    reconstructed_num_pts,
                                                    volume_num_pts,
                                                    &reconstruct,
                                                    &vars_in_neighbor_count,
                                                    &volume_evolved_vars,
                                                    &var_tilde_j,
                                                    &vars_on_lower_face,
                                                    &vars_on_upper_face,
                                                    &subcell_mesh](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;

    const typename tag::type* volume_tensor_ptr = [&volume_evolved_vars,
                                                   &var_tilde_j]() {
      if constexpr (std::is_same_v<tag, ForceFree::Tags::TildeJ>) {
        (void)volume_evolved_vars;  // avoid compiler warnings
        return &get<tag>(var_tilde_j);
      } else {
        (void)var_tilde_j;  // avoid compiler warnings
        return &get<tag>(volume_evolved_vars);
      }
    }();

    const size_t number_of_variables = volume_tensor_ptr->size();

    const gsl::span<const double> volume_vars_span = gsl::make_span(
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

        const DataVector& neighbor_data_dv =
            neighbor_data
                .at(std::pair{direction, *neighbors_in_direction.begin()})
                .neighbor_ghost_data_for_reconstruction();

        ASSERT(neighbor_data_dv.size() != 0,
               "The neighber data is empty in direction "
                   << direction << " on element id " << element.id());

        ghost_cell_vars[direction] = gsl::make_span(
            &neighbor_data_dv[vars_in_neighbor_count * neighbor_num_pts],
            number_of_variables * neighbor_num_pts);
      } else {
        // retrieve boundary ghost data from neighbor_data
        ASSERT(
            element.external_boundaries().count(direction) == 1,
            "Element has neither neighbor nor external boundary to direction : "
                << direction);

        const DataVector& neighbor_data_dv =
            neighbor_data
                .at(std::pair{direction, ElementId<3>::external_boundary_id()})
                .neighbor_ghost_data_for_reconstruction();

        ghost_cell_vars[direction] = gsl::make_span(
            &neighbor_data_dv[0], number_of_variables * neighbor_num_pts);
      }
    }

    reconstruct(make_not_null(&upper_face_vars),
                make_not_null(&lower_face_vars), volume_vars_span,
                ghost_cell_vars, subcell_mesh.extents(), number_of_variables);

    vars_in_neighbor_count += number_of_variables;
  });
}

template <typename TagsList, typename ReconstructLower,
          typename ReconstructUpper>
void reconstruct_fd_neighbor_work(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
    const ReconstructLower& reconstruct_lower_neighbor,
    const ReconstructUpper& reconstruct_upper_neighbor,
    const Variables<System::variables_tag::tags_list>&
        subcell_volume_evolved_vars,
    const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_volume_tilde_j,
    const Element<3>& element,
    const FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        evolution::dg::subcell::GhostData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& ghost_data,
    const Mesh<3>& subcell_mesh, const Direction<3>& direction_to_reconstruct,
    const size_t ghost_zone_size) {
  const std::pair mortar_id{
      direction_to_reconstruct,
      *element.neighbors().at(direction_to_reconstruct).begin()};

  Index<3> ghost_data_extents = subcell_mesh.extents();
  ghost_data_extents[direction_to_reconstruct.dimension()] = ghost_zone_size;

  Variables<ForceFree::fd::tags_list_for_reconstruction> neighbor_vars{};
  {
    ASSERT(ghost_data.contains(mortar_id),
           "The neighbor data does not contain the mortar: ("
               << mortar_id.first << ',' << mortar_id.second << ")");
    const DataVector& neighbor_data_on_mortar =
        ghost_data.at(mortar_id).neighbor_ghost_data_for_reconstruction();
    neighbor_vars.set_data_ref(
        const_cast<double*>(neighbor_data_on_mortar.data()),
        neighbor_vars.number_of_independent_components *
            ghost_data_extents.product());
  }

  tmpl::for_each<ForceFree::fd::tags_list_for_reconstruction>(
      [&direction_to_reconstruct, &ghost_data_extents, &neighbor_vars,
       &reconstruct_lower_neighbor, &reconstruct_upper_neighbor, &subcell_mesh,
       &subcell_volume_evolved_vars, &subcell_volume_tilde_j,
       &vars_on_face](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;

        const typename tag::type* volume_tensor_ptr =
            [&subcell_volume_evolved_vars, &subcell_volume_tilde_j]() {
              if constexpr (std::is_same_v<tag, ForceFree::Tags::TildeJ>) {
                (void)subcell_volume_evolved_vars;  // avoid compiler warnings
                return &subcell_volume_tilde_j;
              } else {
                (void)subcell_volume_tilde_j;  // avoid compiler warnings
                return &get<tag>(subcell_volume_evolved_vars);
              }
            }();

        const auto& tensor_neighbor = get<tag>(neighbor_vars);
        auto& tensor_on_face = get<tag>(*vars_on_face);

        if (direction_to_reconstruct.side() == Side::Upper) {
          for (size_t tensor_index = 0; tensor_index < tensor_on_face.size();
               ++tensor_index) {
            reconstruct_upper_neighbor(
                make_not_null(&tensor_on_face[tensor_index]),
                (*volume_tensor_ptr)[tensor_index],
                tensor_neighbor[tensor_index], subcell_mesh.extents(),
                ghost_data_extents, direction_to_reconstruct);
          }
        } else {
          for (size_t tensor_index = 0; tensor_index < tensor_on_face.size();
               ++tensor_index) {
            reconstruct_lower_neighbor(
                make_not_null(&tensor_on_face[tensor_index]),
                (*volume_tensor_ptr)[tensor_index],
                tensor_neighbor[tensor_index], subcell_mesh.extents(),
                ghost_data_extents, direction_to_reconstruct);
          }
        }
      });
}

}  // namespace ForceFree::fd
