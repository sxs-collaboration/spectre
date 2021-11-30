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
#include "Evolution/Systems/Burgers/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers {
/*!
 * \brief Defines functions useful for testing Burgers subcell
 */
namespace Burgers::fd {
template <typename F>
FixedHashMap<maximum_number_of_neighbors(1) + 1,
             std::pair<Direction<1>, ElementId<1>>,
             evolution::dg::subcell::NeighborData,
             boost::hash<std::pair<Direction<1>, ElementId<1>>>>
compute_neighbor_data(
    const Mesh<1>& subcell_mesh,
    const tnsr::I<DataVector, 1, Frame::ElementLogical>& volume_logical_coords,
    const DirectionMap<1, Neighbors<1>>& neighbors,
    const size_t ghost_zone_size, const F& compute_variables_of_neighbor_data) {
  FixedHashMap<maximum_number_of_neighbors(1) + 1,
               std::pair<Direction<1>, ElementId<1>>,
               evolution::dg::subcell::NeighborData,
               boost::hash<std::pair<Direction<1>, ElementId<1>>>>
      neighbor_data{};
  for (const auto& [direction, neighbors_in_direction] : neighbors) {
    REQUIRE(neighbors_in_direction.size() ==
            1);  // currently only support one neighbor in each direction
    const ElementId<1>& neighbor_id = *neighbors_in_direction.begin();

    auto neighbor_logical_coords = volume_logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    const auto neighbor_vars_for_reconstruction =
        compute_variables_of_neighbor_data(neighbor_logical_coords);

    DirectionMap<1, bool> directions_to_slice{};
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

template <typename Reconstructor>
void test_reconstructor(const size_t num_pts,
                        const Reconstructor& derived_reconstructor) {
  // 1. create U that will be reconstructed, being linear to x
  // 2. send through reconstruction
  // 3. check if U was reconstructed correctly

  const ::Burgers::fd::Reconstructor& reconstructor = derived_reconstructor;
  static_assert(tmpl::list_contains_v<
                typename ::Burgers::fd::Reconstructor::creatable_classes,
                Reconstructor>);

  // create an element and its neighbor elements
  DirectionMap<1, Neighbors<1>> neighbors{};
  for (size_t i = 0; i < 2; ++i) {
    neighbors[gsl::at(Direction<1>::all_directions(), i)] =
        Neighbors<1>{{ElementId<1>{i + 1, {}}}, {}};
  }
  const Element<1> element{ElementId<1>{0, {}}, neighbors};

  // a simple, linear form of the scalar field U for testing purpose
  //   * U(x)   = x + 1       (for 1D)
  const auto compute_solution = [](const auto& coords) {
    Variables<tmpl::list<::Burgers::Tags::U>> vars{get<0>(coords).size(), 0.0};
    get(get<::Burgers::Tags::U>(vars)) += coords.get(0) + 1.0;
    return vars;
  };

  // compute neighbor data
  const Mesh<1> subcell_mesh{num_pts, Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(subcell_mesh);
  const FixedHashMap<maximum_number_of_neighbors(1) + 1,
                     std::pair<Direction<1>, ElementId<1>>,
                     evolution::dg::subcell::NeighborData,
                     boost::hash<std::pair<Direction<1>, ElementId<1>>>>
      neighbor_data = compute_neighbor_data(
          subcell_mesh, logical_coords, element.neighbors(),
          reconstructor.ghost_zone_size(), compute_solution);

  // create Variables on lower and upper faces to perform reconstruction
  using dg_package_data_argument_tags = tmpl::list<
      ::Burgers::Tags::U,
      ::Tags::Flux<::Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>;
  const size_t reconstructed_num_pts = subcell_mesh.number_of_grid_points() + 1;
  std::array<Variables<dg_package_data_argument_tags>, 1> vars_on_lower_face =
      make_array<1>(
          Variables<dg_package_data_argument_tags>(reconstructed_num_pts));
  std::array<Variables<dg_package_data_argument_tags>, 1> vars_on_upper_face =
      make_array<1>(
          Variables<dg_package_data_argument_tags>(reconstructed_num_pts));

  Variables<tmpl::list<::Burgers::Tags::U>> volume_vars{
      subcell_mesh.number_of_grid_points()};
  volume_vars.assign_subset(compute_solution(logical_coords));

  // call the reconstruction
  dynamic_cast<const Reconstructor&>(reconstructor)
      .reconstruct(make_not_null(&vars_on_lower_face),
                   make_not_null(&vars_on_upper_face), volume_vars, element,
                   neighbor_data, subcell_mesh);

  for (size_t dim = 0; dim < 1; ++dim) {
    // construct face-centered coordinates
    const auto basis = make_array<1>(Spectral::Basis::FiniteDifference);
    auto quadrature = make_array<1>(Spectral::Quadrature::CellCentered);
    auto extents = make_array<1>(num_pts);
    gsl::at(extents, dim) = num_pts + 1;
    gsl::at(quadrature, dim) = Spectral::Quadrature::FaceCentered;
    const Mesh<1> face_centered_mesh{extents, basis, quadrature};
    auto logical_coords_face_centered = logical_coordinates(face_centered_mesh);

    // check reconstructed values for reconstruct() function
    Variables<dg_package_data_argument_tags> expected_face_values{
        face_centered_mesh.number_of_grid_points()};
    expected_face_values.assign_subset(
        compute_solution(logical_coords_face_centered));

    CHECK_ITERABLE_APPROX(
        get<::Burgers::Tags::U>(gsl::at(vars_on_lower_face, dim)),
        get<::Burgers::Tags::U>(expected_face_values));
    CHECK_ITERABLE_APPROX(
        get<::Burgers::Tags::U>(gsl::at(vars_on_upper_face, dim)),
        get<::Burgers::Tags::U>(expected_face_values));

    // check reconstructed values for reconstruct_fd_neighbor() function
    Variables<dg_package_data_argument_tags> vars_on_mortar_face{0};
    Variables<dg_package_data_argument_tags> expected_mortar_face_values{0};

    for (size_t upper_or_lower = 0; upper_or_lower < 2; ++upper_or_lower) {
      // iterate over upper and lower direction
      Direction<1> direction =
          gsl::at(Direction<1>::all_directions(), 2 * dim + upper_or_lower);

      // 1D mortar has only one face point
      vars_on_mortar_face.initialize(1);
      expected_mortar_face_values.initialize(1);

      // call the reconstruction
      dynamic_cast<const Reconstructor&>(reconstructor)
          .reconstruct_fd_neighbor(make_not_null(&vars_on_mortar_face),
                                   volume_vars, element, neighbor_data,
                                   subcell_mesh, direction);

      // slice face-centered variables to get the values on the mortar
      data_on_slice(
          make_not_null(&get<::Burgers::Tags::U>(expected_mortar_face_values)),
          get<::Burgers::Tags::U>(expected_face_values),
          face_centered_mesh.extents(), direction.dimension(),
          direction.side() == Side::Lower ? 0
                                          : extents[direction.dimension()] - 1);

      CHECK_ITERABLE_APPROX(
          get<::Burgers::Tags::U>(vars_on_mortar_face),
          get<::Burgers::Tags::U>(expected_mortar_face_values));
    }
  }
}

}  // namespace Burgers::fd
}  // namespace TestHelpers
