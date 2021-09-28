// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::NewtonianEuler::fd {
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
    REQUIRE(neighbors_in_direction.size() == 1);
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

namespace detail {
template <size_t Dim, size_t ThermodynamicDim, typename Reconstructor>
void test_prim_reconstructor_impl(
    const size_t points_per_dimension,
    const Reconstructor& derived_reconstructor,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos) {
  // 1. Create linear prims to reconstruct
  // 2. send through reconstruction
  // 3. check prims and cons were computed correctly
  namespace ne = ::NewtonianEuler;
  const ne::fd::Reconstructor<Dim>& reconstructor = derived_reconstructor;
  static_assert(tmpl::list_contains_v<
                typename ne::fd::Reconstructor<Dim>::creatable_classes,
                Reconstructor>);

  using MassDensityCons = ne::Tags::MassDensityCons;
  using EnergyDensity = ne::Tags::EnergyDensity;
  using MomentumDensity = ne::Tags::MomentumDensity<Dim>;

  // Primitive vars tags
  using MassDensity = ne::Tags::MassDensity<DataVector>;
  using Velocity = ne::Tags::Velocity<DataVector, Dim>;
  using SpecificInternalEnergy = ne::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = ne::Tags::Pressure<DataVector>;

  using prims_tags =
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>;
  using cons_tags = tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>;
  using flux_tags = db::wrap_tags_in<::Tags::Flux, cons_tags, tmpl::size_t<Dim>,
                                     Frame::Inertial>;
  using prim_tags_for_reconstruction =
      tmpl::list<MassDensity, Velocity, Pressure>;

  const Mesh<Dim> subcell_mesh{points_per_dimension,
                               Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(subcell_mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < Dim; ++i) {
    logical_coords.get(i) += 4.0 * i;
  }

  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  for (size_t i = 0; i < 2 * Dim; ++i) {
    neighbors[gsl::at(Direction<Dim>::all_directions(), i)] =
        Neighbors<Dim>{{ElementId<Dim>{i + 1, {}}}, {}};
  }
  const Element<Dim> element{ElementId<Dim>{0, {}}, neighbors};
  const auto compute_solution = [](const auto& coords) {
    Variables<prim_tags_for_reconstruction> vars{get<0>(coords).size(), 0.0};
    for (size_t i = 0; i < Dim; ++i) {
      get(get<MassDensity>(vars)) += coords.get(i);
      get(get<Pressure>(vars)) += coords.get(i);
      for (size_t j = 0;j <Dim;++j) {
        get<Velocity>(vars).get(j) += coords.get(i);
      }
    }
    get(get<MassDensity>(vars)) += 2.0;
    get(get<Pressure>(vars)) += 30.0;
    for (size_t j = 0; j < Dim; ++j) {
      get<Velocity>(vars).get(j) += 1.0e-2 * (j + 2.0) + 10.0;
    }
    return vars;
  };

  const FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
                     std::pair<Direction<Dim>, ElementId<Dim>>,
                     evolution::dg::subcell::NeighborData,
                     boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
      neighbor_data = compute_neighbor_data(
          subcell_mesh, logical_coords, element.neighbors(),
          reconstructor.ghost_zone_size(), compute_solution);

  const size_t reconstructed_num_pts =
      (subcell_mesh.extents(0) + 1) *
      subcell_mesh.extents().slice_away(0).product();

  using dg_package_data_argument_tags =
      tmpl::append<cons_tags, prims_tags, flux_tags>;
  std::array<Variables<dg_package_data_argument_tags>, Dim> vars_on_lower_face =
      make_array<Dim>(
          Variables<dg_package_data_argument_tags>(reconstructed_num_pts));
  std::array<Variables<dg_package_data_argument_tags>, Dim> vars_on_upper_face =
      make_array<Dim>(
          Variables<dg_package_data_argument_tags>(reconstructed_num_pts));

  Variables<prims_tags> volume_prims{subcell_mesh.number_of_grid_points()};
  volume_prims.assign_subset(compute_solution(logical_coords));

  // Now we have everything to call the reconstruction
  dynamic_cast<const Reconstructor&>(reconstructor).reconstruct(
      make_not_null(&vars_on_lower_face), make_not_null(&vars_on_upper_face),
      volume_prims, eos, element, neighbor_data, subcell_mesh);

  for (size_t dim = 0; dim < Dim; ++dim) {
    CAPTURE(dim);
    const auto basis = make_array<Dim>(Spectral::Basis::FiniteDifference);
    auto quadrature = make_array<Dim>(Spectral::Quadrature::CellCentered);
    auto extents = make_array<Dim>(points_per_dimension);
    gsl::at(extents, dim) = points_per_dimension + 1;
    gsl::at(quadrature, dim) = Spectral::Quadrature::FaceCentered;
    const Mesh<Dim> face_centered_mesh{extents, basis, quadrature};
    auto logical_coords_face_centered = logical_coordinates(face_centered_mesh);
    for (size_t i = 1; i < Dim; ++i) {
      logical_coords_face_centered.get(i) =
          logical_coords_face_centered.get(i) + 4.0 * i;
    }
    Variables<dg_package_data_argument_tags> expected_face_values{
        face_centered_mesh.number_of_grid_points()};
    expected_face_values.assign_subset(
        compute_solution(logical_coords_face_centered));
    if constexpr (ThermodynamicDim == 2) {
      get<SpecificInternalEnergy>(expected_face_values) =
          eos.specific_internal_energy_from_density_and_pressure(
              get<MassDensity>(expected_face_values),
              get<Pressure>(expected_face_values));
    } else {
      get<SpecificInternalEnergy>(expected_face_values) =
          eos.specific_internal_energy_from_density(
              get<MassDensity>(expected_face_values));
    }
    ne::ConservativeFromPrimitive<Dim>::apply(
        make_not_null(&get<MassDensityCons>(expected_face_values)),
        make_not_null(&get<MomentumDensity>(expected_face_values)),
        make_not_null(&get<EnergyDensity>(expected_face_values)),
        get<MassDensity>(expected_face_values),
        get<Velocity>(expected_face_values),
        get<SpecificInternalEnergy>(expected_face_values));

    tmpl::for_each<tmpl::append<cons_tags, prims_tags>>(
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
  }
}
}  // namespace detail

template <size_t Dim, typename Reconstructor>
void test_prim_reconstructor(const size_t points_per_dimension,
                             const Reconstructor& derived_reconstructor) {
  detail::test_prim_reconstructor_impl<Dim>(
      points_per_dimension, derived_reconstructor,
      EquationsOfState::IdealFluid<false>{1.4});
  detail::test_prim_reconstructor_impl<Dim>(
      points_per_dimension, derived_reconstructor,
      EquationsOfState::PolytropicFluid<false>{1.0, 2.0});
}
}  // namespace TestHelpers::NewtonianEuler::fd
