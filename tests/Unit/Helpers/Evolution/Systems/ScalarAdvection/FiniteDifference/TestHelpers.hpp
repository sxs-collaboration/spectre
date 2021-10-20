// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers {
/*!
 * \brief Defines functions useful for testing advection subcell
 */
namespace ScalarAdvection::fd {
template <size_t Dim, typename F>
FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
             std::pair<Direction<Dim>, ElementId<Dim>>,
             evolution::dg::subcell::NeighborData,
             boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
compute_neighbor_data(const Mesh<Dim>& subcell_mesh,
                      const tnsr::I<DataVector, Dim, Frame::ElementLogical>&
                          volume_logical_coords,
                      const DirectionMap<Dim, Neighbors<Dim>>& neighbors,
                      const size_t ghost_zone_size,
                      const F& compute_variables_of_neighbor_data) {
  FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
               std::pair<Direction<Dim>, ElementId<Dim>>,
               evolution::dg::subcell::NeighborData,
               boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
      neighbor_data{};
  for (const auto& [direction, neighbors_in_direction] : neighbors) {
    REQUIRE(neighbors_in_direction.size() ==
            1);  // currently only support one neighbor in each direction
    const ElementId<Dim>& neighbor_id = *neighbors_in_direction.begin();

    auto neighbor_logical_coords = volume_logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    const auto neighbor_vars_for_reconstruction =
        compute_variables_of_neighbor_data(neighbor_logical_coords);

    DirectionMap<Dim, bool> directions_to_slice{};
    directions_to_slice[direction.opposite()] = true;
    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars_for_reconstruction.data(),
                       neighbor_vars_for_reconstruction.size()),
        subcell_mesh.extents(), ghost_zone_size, directions_to_slice);
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    neighbor_data[std::pair{direction, neighbor_id}].data_for_reconstruction =
        sliced_data.at(direction.opposite());
  }
  return neighbor_data;
}

template <size_t Dim, typename Reconstructor>
void test_reconstructor(const size_t points_per_dimension,
                        const Reconstructor& derived_reconstructor) {
  // 1. create the scalar field U to reconstruct being linear to coords
  // 2. send through reconstruction
  // 3. check if U was reconstructed correctly

  const ::ScalarAdvection::fd::Reconstructor<Dim>& reconstructor =
      derived_reconstructor;
  static_assert(
      tmpl::list_contains_v<
          typename ::ScalarAdvection::fd::Reconstructor<Dim>::creatable_classes,
          Reconstructor>);

  // create an element and its neighbor elements
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  for (size_t i = 0; i < 2 * Dim; ++i) {
    neighbors[gsl::at(Direction<Dim>::all_directions(), i)] =
        Neighbors<Dim>{{ElementId<Dim>{i + 1, {}}}, {}};
  }
  const Element<Dim> element{ElementId<Dim>{0, {}}, neighbors};

  // a simple, linear form of the scalar field U for testing purpose
  //   * U(x)   = x + 1       (for 1D)
  //   * U(x,y) = x + 2y + 1  (for 2D)
  using scalar_u = ::ScalarAdvection::Tags::U;
  const auto compute_solution = [](const auto& coords) {
    Variables<tmpl::list<scalar_u>> vars{get<0>(coords).size(), 0.0};
    for (size_t i = 0; i < Dim; ++i) {
      get(get<scalar_u>(vars)) += (i + 1) * coords.get(i);
    }
    get(get<scalar_u>(vars)) += 1.0;
    return vars;
  };

  // compute neighbor data
  const Mesh<Dim> subcell_mesh{points_per_dimension,
                               Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(subcell_mesh);
  const FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
                     std::pair<Direction<Dim>, ElementId<Dim>>,
                     evolution::dg::subcell::NeighborData,
                     boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
      neighbor_data = compute_neighbor_data(
          subcell_mesh, logical_coords, element.neighbors(),
          reconstructor.ghost_zone_size(), compute_solution);

  // create Variables on lower and upper faces to perform reconstruction
  using dg_package_data_argument_tags =
      tmpl::list<scalar_u,
                 ::Tags::Flux<scalar_u, tmpl::size_t<Dim>, Frame::Inertial>,
                 ::ScalarAdvection::Tags::VelocityField<Dim>>;
  const size_t reconstructed_num_pts =
      (subcell_mesh.extents(0) + 1) *
      subcell_mesh.extents().slice_away(0).product();
  std::array<Variables<dg_package_data_argument_tags>, Dim> vars_on_lower_face =
      make_array<Dim>(
          Variables<dg_package_data_argument_tags>(reconstructed_num_pts));
  std::array<Variables<dg_package_data_argument_tags>, Dim> vars_on_upper_face =
      make_array<Dim>(
          Variables<dg_package_data_argument_tags>(reconstructed_num_pts));

  Variables<tmpl::list<scalar_u>> volume_vars{
      subcell_mesh.number_of_grid_points()};
  volume_vars.assign_subset(compute_solution(logical_coords));

  // call the reconstruction
  dynamic_cast<const Reconstructor&>(reconstructor)
      .reconstruct(make_not_null(&vars_on_lower_face),
                   make_not_null(&vars_on_upper_face), volume_vars, element,
                   neighbor_data, subcell_mesh);

  for (size_t dim = 0; dim < Dim; ++dim) {
    // construct face-centered coordinates
    const auto basis = make_array<Dim>(Spectral::Basis::FiniteDifference);
    auto quadrature = make_array<Dim>(Spectral::Quadrature::CellCentered);
    auto extents = make_array<Dim>(points_per_dimension);
    gsl::at(extents, dim) = points_per_dimension + 1;
    gsl::at(quadrature, dim) = Spectral::Quadrature::FaceCentered;
    const Mesh<Dim> face_centered_mesh{extents, basis, quadrature};
    auto logical_coords_face_centered = logical_coordinates(face_centered_mesh);

    // check reconstructed values for reconstruct() function
    Variables<dg_package_data_argument_tags> expected_face_values{
        face_centered_mesh.number_of_grid_points()};
    expected_face_values.assign_subset(
        compute_solution(logical_coords_face_centered));

    CHECK_ITERABLE_APPROX(get<scalar_u>(gsl::at(vars_on_lower_face, dim)),
                          get<scalar_u>(expected_face_values));
    CHECK_ITERABLE_APPROX(get<scalar_u>(gsl::at(vars_on_upper_face, dim)),
                          get<scalar_u>(expected_face_values));

    // check reconstructed values for reconstruct_fd_neighbor() function
    Variables<dg_package_data_argument_tags> vars_on_mortar_face{0};
    Variables<dg_package_data_argument_tags> expected_mortar_face_values{0};

    for (size_t upper_or_lower = 0; upper_or_lower < 2; ++upper_or_lower) {
      // iterate over upper and lower direction
      Direction<Dim> direction =
          gsl::at(Direction<Dim>::all_directions(), 2 * dim + upper_or_lower);

      if constexpr (Dim == 1) {
        // 1D mortar has only one face point
        vars_on_mortar_face.initialize(1);
        expected_mortar_face_values.initialize(1);
      } else {
        const size_t num_mortar_face_pts =
            subcell_mesh.extents().slice_away(direction.dimension()).product();
        vars_on_mortar_face.initialize(num_mortar_face_pts);
        expected_mortar_face_values.initialize(num_mortar_face_pts);
      }

      // call the reconstruction
      dynamic_cast<const Reconstructor&>(reconstructor)
          .reconstruct_fd_neighbor(make_not_null(&vars_on_mortar_face),
                                   volume_vars, element, neighbor_data,
                                   subcell_mesh, direction);

      // slice face-centered variables to get the values on the mortar
      data_on_slice(make_not_null(&get<scalar_u>(expected_mortar_face_values)),
                    get<scalar_u>(expected_face_values),
                    face_centered_mesh.extents(), direction.dimension(),
                    direction.side() == Side::Lower
                        ? 0
                        : extents[direction.dimension()] - 1);

      CHECK_ITERABLE_APPROX(get<scalar_u>(vars_on_mortar_face),
                            get<scalar_u>(expected_mortar_face_values));
    }
  }
}

}  // namespace ScalarAdvection::fd
}  // namespace TestHelpers
