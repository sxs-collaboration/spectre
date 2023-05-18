// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::grmhd::GhValenciaDivClean::fd {
namespace detail {
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

inline Variables<::grmhd::GhValenciaDivClean::Tags::
                     primitive_grmhd_and_spacetime_reconstruction_tags>
compute_prim_solution(
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& coords) {
  using Rho = hydro::Tags::RestMassDensity<DataVector>;
  using ElectronFraction = hydro::Tags::ElectronFraction<DataVector>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using MagField = hydro::Tags::MagneticField<DataVector, 3>;
  using Phi = hydro::Tags::DivergenceCleaningField<DataVector>;
  using VelocityW =
      hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>;
  Variables<::grmhd::GhValenciaDivClean::Tags::
                primitive_grmhd_and_spacetime_reconstruction_tags>
      vars{get<0>(coords).size(), 0.0};
  for (size_t i = 0; i < 3; ++i) {
    get(get<Rho>(vars)) += coords.get(i);
    get(get<ElectronFraction>(vars)) += coords.get(i);
    get(get<Pressure>(vars)) += coords.get(i);
    get(get<Phi>(vars)) += coords.get(i);
    for (size_t j = 0; j < 3; ++j) {
      get<VelocityW>(vars).get(j) += coords.get(i);
      get<MagField>(vars).get(j) += coords.get(i);
    }
  }
  get(get<Rho>(vars)) += 2.0;
  get(get<ElectronFraction>(vars)) += 15.0;
  get(get<Pressure>(vars)) += 30.0;
  get(get<Phi>(vars)) += 50.0;
  for (size_t j = 0; j < 3; ++j) {
    get<VelocityW>(vars).get(j) += 1.0e-2 * (j + 2.0) + 10.0;
    get<MagField>(vars).get(j) += 1.0e-2 * (j + 2.0) + 60.0;
  }
  auto& spacetime_metric = get<gr::Tags::SpacetimeMetric<DataVector, 3>>(vars);
  spacetime_metric.get(0, 0) = -1.0;
  for (size_t j = 1; j < 4; ++j) {
    spacetime_metric.get(j, j) = 1.0;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t k = 0; k <= j; ++k) {
        spacetime_metric.get(j, k) += (k + 1) * j * 1.0e-3 * coords.get(i);
      }
    }
  }
  auto& phi = get<gh::Tags::Phi<DataVector, 3>>(vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < 4; ++a) {
      for (size_t b = a; b < 4; ++b) {
        phi.get(i, a, b) = (10 * i + 50 * a + 1000 * b + 1) * coords.get(i);
      }
    }
  }

  auto& pi = get<gh::Tags::Pi<DataVector, 3>>(vars);
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = a; b < 4; ++b) {
      pi.get(a, b) = (500 * a + 10000 * b + 1) * get<0>(coords);
      for (size_t i = 1; i < 3; ++i) {
        pi.get(a, b) += (500 * a + 10000 * b + 1 + i) * coords.get(i);
      }
    }
  }
  return vars;
}

inline Element<3> set_element(const bool skip_last = false) {
  DirectionMap<3, Neighbors<3>> neighbors{};
  for (size_t i = 0; i < 6; ++i) {
    if (skip_last and i == 5) {
      break;
    }
    neighbors[gsl::at(Direction<3>::all_directions(), i)] =
        Neighbors<3>{{ElementId<3>{i + 1, {}}}, {}};
  }
  return Element<3>{ElementId<3>{0, {}}, neighbors};
}

inline tnsr::I<DataVector, 3, Frame::ElementLogical> set_logical_coordinates(
    const Mesh<3>& subcell_mesh) {
  auto logical_coords = logical_coordinates(subcell_mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < 3; ++i) {
    logical_coords.get(i) += 4.0 * i;
  }
  return logical_coords;
}

template <size_t ThermodynamicDim, typename Reconstructor>
void test_prim_reconstructor_impl(
    const size_t points_per_dimension,
    const Reconstructor& derived_reconstructor,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos) {
  // 1. Create linear prims to reconstruct
  // 2. send through reconstruction
  // 3. check prims and cons were computed correctly
  namespace ghmhd = ::grmhd::GhValenciaDivClean;
  const ghmhd::fd::Reconstructor& reconstructor = derived_reconstructor;
  static_assert(
      tmpl::list_contains_v<
          typename ghmhd::fd::Reconstructor::creatable_classes, Reconstructor>);

  using Rho = hydro::Tags::RestMassDensity<DataVector>;
  using ElectronFraction = hydro::Tags::ElectronFraction<DataVector>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using Velocity = hydro::Tags::SpatialVelocity<DataVector, 3>;
  using MagField = hydro::Tags::MagneticField<DataVector, 3>;
  using Phi = hydro::Tags::DivergenceCleaningField<DataVector>;
  using VelocityW =
      hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>;
  using SpecificInternalEnergy =
      hydro::Tags::SpecificInternalEnergy<DataVector>;
  using SpecificEnthalpy = hydro::Tags::SpecificEnthalpy<DataVector>;
  using LorentzFactor = hydro::Tags::LorentzFactor<DataVector>;
  using SpacetimeMetric = gr::Tags::SpacetimeMetric<DataVector, 3>;
  using Lapse = gr::Tags::Lapse<DataVector>;
  using Shift = gr::Tags::Shift<DataVector, 3>;
  using SpatialMetric = gr::Tags::SpatialMetric<DataVector, 3>;
  using InverseSpatialMetric = gr::Tags::InverseSpatialMetric<DataVector, 3>;
  using SqrtDetSpatialMetric = gr::Tags::SqrtDetSpatialMetric<DataVector>;

  using prims_tags = hydro::grmhd_tags<DataVector>;
  using cons_tags = typename ghmhd::System::variables_tag::tags_list;
  using flux_tags =
      db::wrap_tags_in<::Tags::Flux, typename ghmhd::System::flux_variables,
                       tmpl::size_t<3>, Frame::Inertial>;
  using spacetime_tags =
      ::grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags;

  const Mesh<3> subcell_mesh{points_per_dimension,
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  const auto logical_coords = set_logical_coordinates(subcell_mesh);
  const Element<3> element = set_element(true);

  auto neighbors_for_data = element.neighbors();
  neighbors_for_data[gsl::at(Direction<3>::all_directions(), 5)] =
      Neighbors<3>{{ElementId<3>::external_boundary_id()}, {}};
  const FixedHashMap<maximum_number_of_neighbors(3),
                     std::pair<Direction<3>, ElementId<3>>, GhostData,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      ghost_data = compute_ghost_data(
          subcell_mesh, logical_coords, neighbors_for_data,
          reconstructor.ghost_zone_size(), compute_prim_solution);

  const size_t reconstructed_num_pts =
      (subcell_mesh.extents(0) + 1) *
      subcell_mesh.extents().slice_away(0).product();

  using fd_package_data_argument_tags = tmpl::remove_duplicates<tmpl::append<
      cons_tags,
      tmpl::push_back<
          prims_tags,
          hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>,
      flux_tags,
      tmpl::list<Lapse, Shift, SpatialMetric, SqrtDetSpatialMetric,
                 InverseSpatialMetric,
                 evolution::dg::Actions::detail::NormalVector<3>>>>;
  using dg_package_data_argument_tags = tmpl::remove_duplicates<tmpl::append<
      cons_tags,
      tmpl::push_back<
          prims_tags,
          hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>,
      flux_tags,
      tmpl::list<gh::ConstraintDamping::Tags::ConstraintGamma1,
                 gh::ConstraintDamping::Tags::ConstraintGamma2, Lapse, Shift,
                 SpatialMetric, SqrtDetSpatialMetric, InverseSpatialMetric,
                 evolution::dg::Actions::detail::NormalVector<3>>>>;

  std::array<Variables<fd_package_data_argument_tags>, 3> vars_on_lower_face =
      make_array<3>(
          Variables<fd_package_data_argument_tags>(reconstructed_num_pts));
  std::array<Variables<fd_package_data_argument_tags>, 3> vars_on_upper_face =
      make_array<3>(
          Variables<fd_package_data_argument_tags>(reconstructed_num_pts));

  Variables<prims_tags> volume_prims{subcell_mesh.number_of_grid_points()};
  Variables<cons_tags> volume_cons_vars{subcell_mesh.number_of_grid_points()};
  Variables<spacetime_tags> volume_spacetime_vars{
      subcell_mesh.number_of_grid_points()};
  {
    const auto volume_prims_for_recons = compute_prim_solution(logical_coords);
    tmpl::for_each<tmpl::list<Rho, ElectronFraction, Pressure, MagField, Phi>>(
        [&volume_prims, &volume_prims_for_recons](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          get<tag>(volume_prims) = get<tag>(volume_prims_for_recons);
        });
    get<gr::Tags::SpacetimeMetric<DataVector, 3>>(volume_cons_vars) =
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(volume_prims_for_recons);
    const auto spatial_metric = gr::spatial_metric(
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(volume_cons_vars));

    get(get<LorentzFactor>(volume_prims)) =
        sqrt(1.0 + get(dot_product(get<VelocityW>(volume_prims_for_recons),
                                   get<VelocityW>(volume_prims_for_recons),
                                   spatial_metric)));
    for (size_t i = 0; i < 3; ++i) {
      get<Velocity>(volume_prims).get(i) =
          get<VelocityW>(volume_prims_for_recons).get(i) /
          get(get<LorentzFactor>(volume_prims));
    }
    get<gr::Tags::SpacetimeMetric<DataVector, 3>>(volume_cons_vars) =
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(volume_prims_for_recons);
    get<gr::Tags::SpacetimeMetric<DataVector, 3>>(volume_spacetime_vars) =
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(volume_prims_for_recons);
    get<gh::Tags::Phi<DataVector, 3>>(volume_spacetime_vars) =
        get<gh::Tags::Phi<DataVector, 3>>(volume_prims_for_recons);
    get<gh::Tags::Pi<DataVector, 3>>(volume_spacetime_vars) =
        get<gh::Tags::Pi<DataVector, 3>>(volume_prims_for_recons);
  }

  // Now we have everything to call the reconstruction
  dynamic_cast<const Reconstructor&>(reconstructor)
      .reconstruct(make_not_null(&vars_on_lower_face),
                   make_not_null(&vars_on_upper_face), volume_prims,
                   volume_cons_vars, eos, element, ghost_data, subcell_mesh);

  for (size_t dim = 0; dim < 3; ++dim) {
    CAPTURE(dim);
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
    Variables<fd_package_data_argument_tags> expected_lower_face_values{
        face_centered_mesh.number_of_grid_points()};
    expected_lower_face_values.assign_subset(
        compute_prim_solution(logical_coords_face_centered));
    if constexpr (ThermodynamicDim == 2) {
      get<SpecificInternalEnergy>(expected_lower_face_values) =
          eos.specific_internal_energy_from_density_and_pressure(
              get<Rho>(expected_lower_face_values),
              get<Pressure>(expected_lower_face_values));
    } else {
      get<SpecificInternalEnergy>(expected_lower_face_values) =
          eos.specific_internal_energy_from_density(
              get<Rho>(expected_lower_face_values));
    }
    get<SpecificEnthalpy>(expected_lower_face_values) =
        hydro::relativistic_specific_enthalpy(
            get<Rho>(expected_lower_face_values),
            get<SpecificInternalEnergy>(expected_lower_face_values),
            get<Pressure>(expected_lower_face_values));

    gr::spatial_metric(
        make_not_null(&get<SpatialMetric>(expected_lower_face_values)),
        get<SpacetimeMetric>(expected_lower_face_values));
    determinant_and_inverse(
        make_not_null(&get<SqrtDetSpatialMetric>(expected_lower_face_values)),
        make_not_null(&get<InverseSpatialMetric>(expected_lower_face_values)),
        get<SpatialMetric>(expected_lower_face_values));
    get(get<SqrtDetSpatialMetric>(expected_lower_face_values)) =
        sqrt(get(get<SqrtDetSpatialMetric>(expected_lower_face_values)));
    gr::shift(make_not_null(&get<Shift>(expected_lower_face_values)),
              get<SpacetimeMetric>(expected_lower_face_values),
              get<InverseSpatialMetric>(expected_lower_face_values));
    gr::lapse(make_not_null(&get<Lapse>(expected_lower_face_values)),
              get<Shift>(expected_lower_face_values),
              get<SpacetimeMetric>(expected_lower_face_values));

    Variables<fd_package_data_argument_tags> expected_upper_face_values =
        expected_lower_face_values;
    gr::spatial_metric(
        make_not_null(&get<SpatialMetric>(expected_upper_face_values)),
        get<SpacetimeMetric>(expected_upper_face_values));
    determinant_and_inverse(
        make_not_null(&get<SqrtDetSpatialMetric>(expected_upper_face_values)),
        make_not_null(&get<InverseSpatialMetric>(expected_upper_face_values)),
        get<SpatialMetric>(expected_upper_face_values));
    get(get<SqrtDetSpatialMetric>(expected_upper_face_values)) =
        sqrt(get(get<SqrtDetSpatialMetric>(expected_upper_face_values)));
    gr::shift(make_not_null(&get<Shift>(expected_upper_face_values)),
              get<SpacetimeMetric>(expected_upper_face_values),
              get<InverseSpatialMetric>(expected_upper_face_values));
    gr::lapse(make_not_null(&get<Lapse>(expected_upper_face_values)),
              get<Shift>(expected_upper_face_values),
              get<SpacetimeMetric>(expected_upper_face_values));

    get(get<LorentzFactor>(expected_lower_face_values)) = sqrt(
        1.0 + get(dot_product(get<VelocityW>(expected_lower_face_values),
                              get<VelocityW>(expected_lower_face_values),
                              get<SpatialMetric>(expected_lower_face_values))));
    get(get<LorentzFactor>(expected_upper_face_values)) = sqrt(
        1.0 + get(dot_product(get<VelocityW>(expected_upper_face_values),
                              get<VelocityW>(expected_upper_face_values),
                              get<SpatialMetric>(expected_upper_face_values))));
    for (size_t i = 0; i < 3; ++i) {
      get<Velocity>(expected_lower_face_values).get(i) =
          get<VelocityW>(expected_lower_face_values).get(i) /
          get(get<LorentzFactor>(expected_lower_face_values));
      get<Velocity>(expected_upper_face_values).get(i) =
          get<VelocityW>(expected_upper_face_values).get(i) /
          get(get<LorentzFactor>(expected_upper_face_values));
    }

    namespace mhd = ::grmhd::ValenciaDivClean;
    mhd::ConservativeFromPrimitive::apply(
        make_not_null(&get<mhd::Tags::TildeD>(expected_lower_face_values)),
        make_not_null(&get<mhd::Tags::TildeYe>(expected_lower_face_values)),
        make_not_null(&get<mhd::Tags::TildeTau>(expected_lower_face_values)),
        make_not_null(&get<mhd::Tags::TildeS<>>(expected_lower_face_values)),
        make_not_null(&get<mhd::Tags::TildeB<>>(expected_lower_face_values)),
        make_not_null(&get<mhd::Tags::TildePhi>(expected_lower_face_values)),
        get<Rho>(expected_lower_face_values),
        get<ElectronFraction>(expected_lower_face_values),
        get<SpecificInternalEnergy>(expected_lower_face_values),
        get<Pressure>(expected_lower_face_values),
        get<Velocity>(expected_lower_face_values),
        get<LorentzFactor>(expected_lower_face_values),
        get<MagField>(expected_lower_face_values),
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
            expected_lower_face_values),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(expected_lower_face_values),
        get<Phi>(expected_lower_face_values));
    mhd::ConservativeFromPrimitive::apply(
        make_not_null(&get<mhd::Tags::TildeD>(expected_upper_face_values)),
        make_not_null(&get<mhd::Tags::TildeYe>(expected_upper_face_values)),
        make_not_null(&get<mhd::Tags::TildeTau>(expected_upper_face_values)),
        make_not_null(&get<mhd::Tags::TildeS<>>(expected_upper_face_values)),
        make_not_null(&get<mhd::Tags::TildeB<>>(expected_upper_face_values)),
        make_not_null(&get<mhd::Tags::TildePhi>(expected_upper_face_values)),
        get<Rho>(expected_upper_face_values),
        get<ElectronFraction>(expected_upper_face_values),
        get<SpecificInternalEnergy>(expected_upper_face_values),
        get<Pressure>(expected_upper_face_values),
        get<Velocity>(expected_upper_face_values),
        get<LorentzFactor>(expected_upper_face_values),
        get<MagField>(expected_upper_face_values),
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
            expected_upper_face_values),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(expected_upper_face_values),
        get<Phi>(expected_upper_face_values));

    using tags_to_test = tmpl::push_back<
        tmpl::append<typename mhd::System::variables_tag::tags_list,
                     prims_tags>,
        Lapse, Shift, SqrtDetSpatialMetric, SpatialMetric,
        InverseSpatialMetric>;
    tmpl::for_each<tags_to_test>([dim, &expected_lower_face_values,
                                  &expected_upper_face_values,
                                  &vars_on_lower_face,
                                  &vars_on_upper_face](auto tag_to_check_v) {
      using tag_to_check = tmpl::type_from<decltype(tag_to_check_v)>;
      CAPTURE(db::tag_name<tag_to_check>());
      CHECK_ITERABLE_APPROX(get<tag_to_check>(gsl::at(vars_on_lower_face, dim)),
                            get<tag_to_check>(expected_lower_face_values));
      CHECK_ITERABLE_APPROX(get<tag_to_check>(gsl::at(vars_on_upper_face, dim)),
                            get<tag_to_check>(expected_upper_face_values));
    });

    // Test reconstruct_fd_neighbor
    const size_t num_pts_on_mortar =
        face_centered_mesh.slice_away(dim).number_of_grid_points();
    Variables<dg_package_data_argument_tags> upper_side_vars_on_mortar{
        num_pts_on_mortar};
    if (dim != 2) {
      dynamic_cast<const Reconstructor&>(reconstructor)
          .reconstruct_fd_neighbor(make_not_null(&upper_side_vars_on_mortar),
                                   volume_prims, volume_spacetime_vars, eos,
                                   element, ghost_data, subcell_mesh,
                                   Direction<3>{dim, Side::Upper});
    }

    Variables<dg_package_data_argument_tags> lower_side_vars_on_mortar{
        num_pts_on_mortar};
    dynamic_cast<const Reconstructor&>(reconstructor)
        .reconstruct_fd_neighbor(make_not_null(&lower_side_vars_on_mortar),
                                 volume_prims, volume_spacetime_vars, eos,
                                 element, ghost_data, subcell_mesh,
                                 Direction<3>{dim, Side::Lower});

    tmpl::for_each<tmpl::append<tags_to_test, spacetime_tags>>(
        [dim, &expected_lower_face_values, &expected_upper_face_values,
         &lower_side_vars_on_mortar, &face_centered_mesh,
         &upper_side_vars_on_mortar](auto tag_to_check_v) {
          using tag_to_check = tmpl::type_from<decltype(tag_to_check_v)>;
          CAPTURE(db::tag_name<tag_to_check>());
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(lower_side_vars_on_mortar),
              data_on_slice(get<tag_to_check>(expected_lower_face_values),
                            face_centered_mesh.extents(), dim, 0));
          if (dim != 2) {
            CHECK_ITERABLE_APPROX(
                get<tag_to_check>(upper_side_vars_on_mortar),
                data_on_slice(get<tag_to_check>(expected_upper_face_values),
                              face_centered_mesh.extents(), dim,
                              face_centered_mesh.extents(dim) - 1));
          }
        });
  }
}
}  // namespace detail

template <typename Reconstructor>
void test_prim_reconstructor(const size_t points_per_dimension,
                             const Reconstructor& derived_reconstructor) {
  detail::test_prim_reconstructor_impl(points_per_dimension,
                                       derived_reconstructor,
                                       EquationsOfState::IdealFluid<true>{1.4});
  detail::test_prim_reconstructor_impl(
      points_per_dimension, derived_reconstructor,
      EquationsOfState::PolytropicFluid<true>{1.0, 2.0});
}
}  // namespace TestHelpers::grmhd::GhValenciaDivClean::fd
