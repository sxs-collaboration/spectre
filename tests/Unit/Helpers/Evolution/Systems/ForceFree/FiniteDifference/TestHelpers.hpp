// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers {

/*!
 * \brief Defines functions useful for testing subcell in ForceFree evolution
 * system
 */
namespace ForceFree::fd {

using GhostData = evolution::dg::subcell::GhostData;

template <typename F>
FixedHashMap<maximum_number_of_neighbors(3),
             std::pair<Direction<3>, ElementId<3>>, GhostData,
             boost::hash<std::pair<Direction<3>, ElementId<3>>>>
compute_ghost_data(
    const Mesh<3>& subcell_mesh,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& volume_logical_coords,
    const DirectionMap<3, Neighbors<3>>& neighbors,
    const size_t ghost_zone_size, const F& compute_variables_of_neighbor_data) {
  FixedHashMap<maximum_number_of_neighbors(3),
               std::pair<Direction<3>, ElementId<3>>, GhostData,
               boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      ghost_data{};

  for (const auto& [direction, neighbors_in_direction] : neighbors) {
    REQUIRE(neighbors_in_direction.size() == 1);
    const ElementId<3>& neighbor_id = *neighbors_in_direction.begin();
    auto neighbor_logical_coords = volume_logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    const auto neighbor_vars_for_reconstruction =
        compute_variables_of_neighbor_data(neighbor_logical_coords);

    DirectionMap<3, bool> directions_to_slice{};
    directions_to_slice[direction.opposite()] = true;
    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars_for_reconstruction.data(),
                       neighbor_vars_for_reconstruction.size()),
        subcell_mesh.extents(), ghost_zone_size,
        std::unordered_set{direction.opposite()}, 0);

    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));

    ghost_data[std::pair{direction, neighbor_id}] = GhostData{1};
    ghost_data.at(std::pair{direction, neighbor_id})
        .neighbor_ghost_data_for_reconstruction() =
        sliced_data.at(direction.opposite());
  }
  return ghost_data;
}

template <typename Reconstructor>
void test_reconstructor(const size_t points_per_dimension,
                        const Reconstructor& derived_reconstructor) {
  // 1. create the variables to be reconstructed (evolved variables and current
  //    density TildeJ) being linear to coords
  // 2. send through reconstruction
  // 3. check if evolved variables were reconstructed correctly

  const ::ForceFree::fd::Reconstructor& reconstructor = derived_reconstructor;
  static_assert(tmpl::list_contains_v<
                typename ::ForceFree::fd::Reconstructor::creatable_classes,
                Reconstructor>);

  // create an element and its neighbor elements
  DirectionMap<3, Neighbors<3>> neighbors{};
  for (size_t i = 0; i < 2 * 3; ++i) {
    neighbors[gsl::at(Direction<3>::all_directions(), i)] =
        Neighbors<3>{{ElementId<3>{i + 1, {}}}, {}};
  }
  const Element<3> element{ElementId<3>{0, {}}, neighbors};

  using TildeE = ::ForceFree::Tags::TildeE;
  using TildeB = ::ForceFree::Tags::TildeB;
  using TildePsi = ::ForceFree::Tags::TildePsi;
  using TildePhi = ::ForceFree::Tags::TildePhi;
  using TildeQ = ::ForceFree::Tags::TildeQ;
  using TildeJ = ::ForceFree::Tags::TildeJ;

  using cons_tags = tmpl::list<TildeE, TildeB, TildePsi, TildePhi, TildeQ>;
  using flux_tags = db::wrap_tags_in<::Tags::Flux, cons_tags, tmpl::size_t<3>,
                                     Frame::Inertial>;

  const Mesh<3> subcell_mesh{points_per_dimension,
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(subcell_mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < 3; ++i) {
    logical_coords.get(i) += 4.0 * i;
  }

  // a simple, linear variables for testing purpose
  const auto compute_solution = [](const auto& coords) {
    Variables<::ForceFree::fd::tags_list_for_reconstruction> vars{
        get<0>(coords).size(), 0.0};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        get<TildeE>(vars).get(j) += 1.0 * coords.get(i);
        get<TildeB>(vars).get(j) += 2.0 * coords.get(i);
        get<TildeJ>(vars).get(j) += 3.0 * coords.get(i);
      }
      get(get<TildePsi>(vars)) += 4.0 * coords.get(i);
      get(get<TildePhi>(vars)) += 5.0 * coords.get(i);
      get(get<TildeQ>(vars)) += 6.0 * coords.get(i);
    }
    get(get<TildePsi>(vars)) += 2.0;
    get(get<TildePhi>(vars)) += 3.0;
    get(get<TildeQ>(vars)) += 40.0;
    for (size_t j = 0; j < 3; ++j) {
      get<TildeE>(vars).get(j) += 1.0e-2 * (j + 1.0) + 10.0;
      get<TildeB>(vars).get(j) += 1.0e-2 * (j + 2.0) + 20.0;
      get<TildeJ>(vars).get(j) += 1.0e-2 * (j + 3.0) + 30.0;
    }
    return vars;
  };

  const size_t num_subcell_grid_pts = subcell_mesh.number_of_grid_points();

  Variables<::ForceFree::fd::tags_list_for_reconstruction>
      volume_vars_and_tilde_j{num_subcell_grid_pts};
  volume_vars_and_tilde_j.assign_subset(compute_solution(logical_coords));

  Variables<cons_tags> volume_vars =
      volume_vars_and_tilde_j.reference_subset<cons_tags>();
  const tnsr::I<DataVector, 3> volume_tilde_j =
      get<TildeJ>(volume_vars_and_tilde_j);

  // compute ghost data from neighbor
  const FixedHashMap<maximum_number_of_neighbors(3),
                     std::pair<Direction<3>, ElementId<3>>, GhostData,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      ghost_data =
          compute_ghost_data(subcell_mesh, logical_coords, element.neighbors(),
                             reconstructor.ghost_zone_size(), compute_solution);

  // create Variables on lower and upper faces to perform reconstruction
  const size_t reconstructed_num_pts =
      (subcell_mesh.extents(0) + 1) *
      subcell_mesh.extents().slice_away(0).product();
  using face_vars_tags = tmpl::append<
      tmpl::list<TildeJ>, cons_tags, flux_tags,
      tmpl::remove_duplicates<tmpl::push_back<tmpl::list<
          gr::Tags::Lapse<DataVector>,
          gr::Tags::Shift<DataVector, 3, Frame::Inertial>,
          gr::Tags::SpatialMetric<DataVector, 3>,
          gr::Tags::SqrtDetSpatialMetric<DataVector>,
          gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
          evolution::dg::Actions::detail::NormalVector<3>>>>>;
  tnsr::ii<DataVector, 3, Frame::Inertial> lower_face_spatial_metric{
      reconstructed_num_pts, 0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> upper_face_spatial_metric{
      reconstructed_num_pts, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    lower_face_spatial_metric.get(i, i) = 1.0 + 0.01 * i;
    upper_face_spatial_metric.get(i, i) = 1.0 - 0.01 * i;
  }
  const Scalar<DataVector> lower_face_sqrt_det_spatial_metric{
      sqrt(get(determinant(lower_face_spatial_metric)))};
  const Scalar<DataVector> upper_face_sqrt_det_spatial_metric{
      sqrt(get(determinant(upper_face_spatial_metric)))};

  std::array<Variables<face_vars_tags>, 3> vars_on_lower_face =
      make_array<3>(Variables<face_vars_tags>(reconstructed_num_pts));
  std::array<Variables<face_vars_tags>, 3> vars_on_upper_face =
      make_array<3>(Variables<face_vars_tags>(reconstructed_num_pts));
  for (size_t i = 0; i < 3; ++i) {
    get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
        gsl::at(vars_on_lower_face, i)) = lower_face_sqrt_det_spatial_metric;
    get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
        gsl::at(vars_on_upper_face, i)) = upper_face_sqrt_det_spatial_metric;

    get<gr::Tags::SpatialMetric<DataVector, 3>>(
        gsl::at(vars_on_lower_face, i)) = lower_face_spatial_metric;
    get<gr::Tags::SpatialMetric<DataVector, 3>>(
        gsl::at(vars_on_upper_face, i)) = upper_face_spatial_metric;
  }

  // Now we have everything to call the reconstruction
  dynamic_cast<const Reconstructor&>(reconstructor)
      .reconstruct(make_not_null(&vars_on_lower_face),
                   make_not_null(&vars_on_upper_face), volume_vars,
                   volume_tilde_j, element, ghost_data, subcell_mesh);

  for (size_t dim = 0; dim < 3; ++dim) {
    CAPTURE(dim);

    // construct face-centered coordinates
    const auto basis = make_array<3>(Spectral::Basis::FiniteDifference);
    auto quadrature = make_array<3>(Spectral::Quadrature::CellCentered);
    auto extents = make_array<3>(points_per_dimension);
    gsl::at(extents, dim) = points_per_dimension + 1;
    gsl::at(quadrature, dim) = Spectral::Quadrature::FaceCentered;
    const Mesh<3> face_centered_mesh{extents, basis, quadrature};
    auto logical_coords_face_centered = logical_coordinates(face_centered_mesh);
    for (size_t i = 1; i < 3; ++i) {
      logical_coords_face_centered.get(i) =
          logical_coords_face_centered.get(i) + 4.0 * i;
    }

    // check reconstructed values for reconstruct() function
    Variables<face_vars_tags> expected_face_values{
        face_centered_mesh.number_of_grid_points()};
    expected_face_values.assign_subset(
        compute_solution(logical_coords_face_centered));

    tmpl::for_each<::ForceFree::fd::tags_list_for_reconstruction>(
        [dim, &expected_face_values, &vars_on_lower_face,
         &vars_on_upper_face](auto tag_to_check_v) {
          using tag_to_check = tmpl::type_from<decltype(tag_to_check_v)>;
          CAPTURE(db::tag_name<tag_to_check>());
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(gsl::at(vars_on_lower_face, dim)),
              get<tag_to_check>(expected_face_values));
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(gsl::at(vars_on_upper_face, dim)),
              get<tag_to_check>(expected_face_values));
        });

    // Test reconstruct_fd_neighbor
    const size_t num_pts_on_mortar =
        face_centered_mesh.slice_away(dim).number_of_grid_points();

    Variables<face_vars_tags> upper_side_vars_on_mortar{num_pts_on_mortar};
    // Slice GR variables onto the mortar
    data_on_slice(
        make_not_null(&get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
            upper_side_vars_on_mortar)),
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(expected_face_values),
        face_centered_mesh.extents(), dim, face_centered_mesh.extents(dim) - 1);
    data_on_slice(
        make_not_null(&get<gr::Tags::SpatialMetric<DataVector, 3>>(
            upper_side_vars_on_mortar)),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(expected_face_values),
        face_centered_mesh.extents(), dim, face_centered_mesh.extents(dim) - 1);

    dynamic_cast<const Reconstructor&>(reconstructor)
        .reconstruct_fd_neighbor(make_not_null(&upper_side_vars_on_mortar),
                                 volume_vars, volume_tilde_j, element,
                                 ghost_data, subcell_mesh,
                                 Direction<3>{dim, Side::Upper});

    Variables<face_vars_tags> lower_side_vars_on_mortar{num_pts_on_mortar};
    // Slice GR variables onto the mortar
    data_on_slice(
        make_not_null(&get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
            lower_side_vars_on_mortar)),
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(expected_face_values),
        face_centered_mesh.extents(), dim, 0);
    data_on_slice(
        make_not_null(&get<gr::Tags::SpatialMetric<DataVector, 3>>(
            lower_side_vars_on_mortar)),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(expected_face_values),
        face_centered_mesh.extents(), dim, 0);

    dynamic_cast<const Reconstructor&>(reconstructor)
        .reconstruct_fd_neighbor(make_not_null(&lower_side_vars_on_mortar),
                                 volume_vars, volume_tilde_j, element,
                                 ghost_data, subcell_mesh,
                                 Direction<3>{dim, Side::Lower});

    tmpl::for_each<cons_tags>(
        [dim, &expected_face_values, &lower_side_vars_on_mortar,
         &face_centered_mesh, &upper_side_vars_on_mortar](auto tag_to_check_v) {
          using tag_to_check = tmpl::type_from<decltype(tag_to_check_v)>;
          CAPTURE(db::tag_name<tag_to_check>());
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(lower_side_vars_on_mortar),
              data_on_slice(get<tag_to_check>(expected_face_values),
                            face_centered_mesh.extents(), dim, 0));
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(upper_side_vars_on_mortar),
              data_on_slice(get<tag_to_check>(expected_face_values),
                            face_centered_mesh.extents(), dim,
                            face_centered_mesh.extents(dim) - 1));
        });
  }
}

}  // namespace ForceFree::fd
}  // namespace TestHelpers
