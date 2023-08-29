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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::grmhd::ValenciaDivClean::fd {
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

template <size_t ThermodynamicDim, typename Reconstructor>
void test_prim_reconstructor_impl(
    const size_t points_per_dimension,
    const Reconstructor& derived_reconstructor,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos) {
  // 1. Create linear prims to reconstruct
  // 2. send through reconstruction
  // 3. check prims and cons were computed correctly
  namespace mhd = ::grmhd::ValenciaDivClean;
  const mhd::fd::Reconstructor& reconstructor = derived_reconstructor;
  static_assert(
      tmpl::list_contains_v<typename mhd::fd::Reconstructor::creatable_classes,
                            Reconstructor>);

  using Rho = hydro::Tags::RestMassDensity<DataVector>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using ElectronFraction = hydro::Tags::ElectronFraction<DataVector>;
  using Velocity = hydro::Tags::SpatialVelocity<DataVector, 3>;
  using MagField = hydro::Tags::MagneticField<DataVector, 3>;
  using Phi = hydro::Tags::DivergenceCleaningField<DataVector>;
  using VelocityW =
      hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>;
  using SpecificInternalEnergy =
      hydro::Tags::SpecificInternalEnergy<DataVector>;
  using Temperature = hydro::Tags::Temperature<DataVector>;
  using SpecificEnthalpy = hydro::Tags::SpecificEnthalpy<DataVector>;
  using LorentzFactor = hydro::Tags::LorentzFactor<DataVector>;

  using prims_tags = hydro::grmhd_tags<DataVector>;
  using cons_tags = typename mhd::System::variables_tag::tags_list;
  using flux_tags = db::wrap_tags_in<::Tags::Flux, cons_tags, tmpl::size_t<3>,
                                     Frame::Inertial>;
  using prim_tags_for_reconstruction =
      tmpl::list<Rho, ElectronFraction, Pressure, VelocityW, MagField, Phi>;

  const Mesh<3> subcell_mesh{points_per_dimension,
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(subcell_mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < 3; ++i) {
    logical_coords.get(i) += 4.0 * i;
  }

  DirectionMap<3, Neighbors<3>> neighbors{};
  for (size_t i = 0; i < 2 * 3; ++i) {
    neighbors[gsl::at(Direction<3>::all_directions(), i)] =
        Neighbors<3>{{ElementId<3>{i + 1, {}}}, {}};
  }
  const Element<3> element{ElementId<3>{0, {}}, neighbors};
  const auto compute_solution = [](const auto& coords) {
    Variables<prim_tags_for_reconstruction> vars{get<0>(coords).size(), 0.0};
    for (size_t i = 0; i < 3; ++i) {
      get(get<Rho>(vars)) += coords.get(i);
      get(get<Pressure>(vars)) += coords.get(i);
      get(get<ElectronFraction>(vars)) += coords.get(i);
      get(get<Phi>(vars)) += coords.get(i);
      for (size_t j = 0; j < 3; ++j) {
        get<VelocityW>(vars).get(j) += coords.get(i);
        get<MagField>(vars).get(j) += coords.get(i);
      }
    }
    get(get<Rho>(vars)) += 2.0;
    get(get<Pressure>(vars)) += 30.0;
    get(get<ElectronFraction>(vars)) += 40.0;
    get(get<Phi>(vars)) += 50.0;
    for (size_t j = 0; j < 3; ++j) {
      get<VelocityW>(vars).get(j) += 1.0e-2 * (j + 2.0) + 10.0;
      get<MagField>(vars).get(j) += 1.0e-2 * (j + 2.0) + 60.0;
    }
    return vars;
  };

  const FixedHashMap<maximum_number_of_neighbors(3),
                     std::pair<Direction<3>, ElementId<3>>,
                     evolution::dg::subcell::GhostData,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      ghost_data =
          compute_ghost_data(subcell_mesh, logical_coords, element.neighbors(),
                             reconstructor.ghost_zone_size(), compute_solution);

  const size_t reconstructed_num_pts =
      (subcell_mesh.extents(0) + 1) *
      subcell_mesh.extents().slice_away(0).product();

  using dg_package_data_argument_tags = tmpl::append<
      cons_tags,
      tmpl::push_back<
          prims_tags,
          hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>,
      flux_tags,
      tmpl::remove_duplicates<tmpl::push_back<tmpl::list<
          gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
          gr::Tags::SpatialMetric<DataVector, 3>,
          gr::Tags::SqrtDetSpatialMetric<DataVector>,
          gr::Tags::InverseSpatialMetric<DataVector, 3>,
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

  std::array<Variables<dg_package_data_argument_tags>, 3> vars_on_lower_face =
      make_array<3>(
          Variables<dg_package_data_argument_tags>(reconstructed_num_pts));
  std::array<Variables<dg_package_data_argument_tags>, 3> vars_on_upper_face =
      make_array<3>(
          Variables<dg_package_data_argument_tags>(reconstructed_num_pts));
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

  Variables<prims_tags> volume_prims{subcell_mesh.number_of_grid_points()};
  {
    const auto volume_prims_for_recons = compute_solution(logical_coords);
    tmpl::for_each<tmpl::list<Rho, ElectronFraction, Pressure, MagField, Phi>>(
        [&volume_prims, &volume_prims_for_recons](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          get<tag>(volume_prims) = get<tag>(volume_prims_for_recons);
        });
    // Metric is identity at cell center
    get(get<LorentzFactor>(volume_prims)) =
        sqrt(1.0 + get(dot_product(get<VelocityW>(volume_prims_for_recons),
                                   get<VelocityW>(volume_prims_for_recons))));
    for (size_t i = 0; i < 3; ++i) {
      get<Velocity>(volume_prims).get(i) =
          get<VelocityW>(volume_prims_for_recons).get(i) /
          get(get<LorentzFactor>(volume_prims));
    }
  }

  // Now we have everything to call the reconstruction
  if constexpr (Reconstructor::use_adaptive_order) {
    std::array<std::vector<std::uint8_t>, 3> reconstruction_order_storage{};
    std::optional<std::array<gsl::span<std::uint8_t>, 3>>
        reconstruction_order{};
    reconstruction_order.emplace();
    for (size_t i = 0; i < 3; ++i) {
      auto order_extents = subcell_mesh.extents();
      order_extents[i] += 2;
      gsl::at(reconstruction_order_storage, i).resize(order_extents.product());
      // Ensure we have reset the values to max so the min calls are fine.
      std::fill_n(gsl::at(reconstruction_order_storage, i).begin(),
                  order_extents.product(),
                  std::numeric_limits<std::uint8_t>::max());
      gsl::at(reconstruction_order.value(), i) = gsl::span<std::uint8_t>{
          gsl::at(reconstruction_order_storage, i).data(),
          gsl::at(reconstruction_order_storage, i).size()};
    }

    dynamic_cast<const Reconstructor&>(reconstructor)
        .reconstruct(make_not_null(&vars_on_lower_face),
                     make_not_null(&vars_on_upper_face),
                     make_not_null(&reconstruction_order), volume_prims, eos,
                     element, ghost_data, subcell_mesh);
    for (size_t d = 0; d < 3; ++d) {
      CAPTURE(d);
      for (size_t i = 0; i < gsl::at(reconstruction_order_storage, d).size();
           ++i) {
        CAPTURE(i);
        CHECK(gsl::at(reconstruction_order_storage, d)[i] >= 1);
        CHECK(gsl::at(reconstruction_order_storage, d)[i] <= 9);
      }
    }
  } else {
    dynamic_cast<const Reconstructor&>(reconstructor)
        .reconstruct(make_not_null(&vars_on_lower_face),
                     make_not_null(&vars_on_upper_face), volume_prims, eos,
                     element, ghost_data, subcell_mesh);
  }

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
    Variables<dg_package_data_argument_tags> expected_lower_face_values{
        face_centered_mesh.number_of_grid_points()};
    expected_lower_face_values.assign_subset(
        compute_solution(logical_coords_face_centered));
    if constexpr (ThermodynamicDim == 2) {
      get<SpecificInternalEnergy>(expected_lower_face_values) =
          eos.specific_internal_energy_from_density_and_pressure(
              get<Rho>(expected_lower_face_values),
              get<Pressure>(expected_lower_face_values));
      get<Temperature>(expected_lower_face_values) =
          eos.temperature_from_density_and_energy(
              get<Rho>(expected_lower_face_values),
              get<SpecificInternalEnergy>(expected_lower_face_values));
    } else {
      get<SpecificInternalEnergy>(expected_lower_face_values) =
          eos.specific_internal_energy_from_density(
              get<Rho>(expected_lower_face_values));
      get<Temperature>(expected_lower_face_values) =
          eos.temperature_from_density(get<Rho>(expected_lower_face_values));
    }
    get<SpecificEnthalpy>(expected_lower_face_values) =
        hydro::relativistic_specific_enthalpy(
            get<Rho>(expected_lower_face_values),
            get<SpecificInternalEnergy>(expected_lower_face_values),
            get<Pressure>(expected_lower_face_values));

    Variables<dg_package_data_argument_tags> expected_upper_face_values =
        expected_lower_face_values;
    get(get<LorentzFactor>(expected_lower_face_values)) =
        sqrt(1.0 + get(dot_product(get<VelocityW>(expected_lower_face_values),
                                   get<VelocityW>(expected_lower_face_values),
                                   lower_face_spatial_metric)));
    get(get<LorentzFactor>(expected_upper_face_values)) =
        sqrt(1.0 + get(dot_product(get<VelocityW>(expected_upper_face_values),
                                   get<VelocityW>(expected_upper_face_values),
                                   upper_face_spatial_metric)));
    for (size_t i = 0; i < 3; ++i) {
      get<Velocity>(expected_lower_face_values).get(i) =
          get<VelocityW>(expected_lower_face_values).get(i) /
          get(get<LorentzFactor>(expected_lower_face_values));
      get<Velocity>(expected_upper_face_values).get(i) =
          get<VelocityW>(expected_upper_face_values).get(i) /
          get(get<LorentzFactor>(expected_upper_face_values));
    }

    get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
        expected_lower_face_values) = lower_face_sqrt_det_spatial_metric;
    get<gr::Tags::SpatialMetric<DataVector, 3>>(expected_lower_face_values) =
        lower_face_spatial_metric;
    get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
        expected_upper_face_values) = upper_face_sqrt_det_spatial_metric;
    get<gr::Tags::SpatialMetric<DataVector, 3>>(expected_upper_face_values) =
        upper_face_spatial_metric;

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

    tmpl::for_each<tmpl::append<cons_tags, prims_tags>>(
        [dim, &expected_lower_face_values, &expected_upper_face_values,
         &vars_on_lower_face, &vars_on_upper_face](auto tag_to_check_v) {
          using tag_to_check = tmpl::type_from<decltype(tag_to_check_v)>;
          CAPTURE(db::tag_name<tag_to_check>());
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(gsl::at(vars_on_lower_face, dim)),
              get<tag_to_check>(expected_lower_face_values));
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(gsl::at(vars_on_upper_face, dim)),
              get<tag_to_check>(expected_upper_face_values));
        });

    // Test reconstruct_fd_neighbor
    const size_t num_pts_on_mortar =
        face_centered_mesh.slice_away(dim).number_of_grid_points();
    Variables<dg_package_data_argument_tags> upper_side_vars_on_mortar{
        num_pts_on_mortar};
    // Slice GR variables onto the mortar
    data_on_slice(
        make_not_null(&get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
            upper_side_vars_on_mortar)),
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
            expected_upper_face_values),
        face_centered_mesh.extents(), dim, face_centered_mesh.extents(dim) - 1);
    data_on_slice(
        make_not_null(&get<gr::Tags::SpatialMetric<DataVector, 3>>(
            upper_side_vars_on_mortar)),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(expected_upper_face_values),
        face_centered_mesh.extents(), dim, face_centered_mesh.extents(dim) - 1);

    dynamic_cast<const Reconstructor&>(reconstructor)
        .reconstruct_fd_neighbor(make_not_null(&upper_side_vars_on_mortar),
                                 volume_prims, eos, element, ghost_data,
                                 subcell_mesh, Direction<3>{dim, Side::Upper});

    Variables<dg_package_data_argument_tags> lower_side_vars_on_mortar{
        num_pts_on_mortar};
    // Slice GR variables onto the mortar
    data_on_slice(
        make_not_null(&get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
            lower_side_vars_on_mortar)),
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
            expected_lower_face_values),
        face_centered_mesh.extents(), dim, 0);
    data_on_slice(
        make_not_null(&get<gr::Tags::SpatialMetric<DataVector, 3>>(
            lower_side_vars_on_mortar)),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(expected_lower_face_values),
        face_centered_mesh.extents(), dim, 0);

    dynamic_cast<const Reconstructor&>(reconstructor)
        .reconstruct_fd_neighbor(make_not_null(&lower_side_vars_on_mortar),
                                 volume_prims, eos, element, ghost_data,
                                 subcell_mesh, Direction<3>{dim, Side::Lower});

    tmpl::for_each<tmpl::append<cons_tags, prims_tags>>(
        [dim, &expected_lower_face_values, &expected_upper_face_values,
         &lower_side_vars_on_mortar, &face_centered_mesh,
         &upper_side_vars_on_mortar](auto tag_to_check_v) {
          using tag_to_check = tmpl::type_from<decltype(tag_to_check_v)>;
          CAPTURE(db::tag_name<tag_to_check>());
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(lower_side_vars_on_mortar),
              data_on_slice(get<tag_to_check>(expected_lower_face_values),
                            face_centered_mesh.extents(), dim, 0));
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(upper_side_vars_on_mortar),
              data_on_slice(get<tag_to_check>(expected_upper_face_values),
                            face_centered_mesh.extents(), dim,
                            face_centered_mesh.extents(dim) - 1));
        });
  }
}
}  // namespace detail

template <typename Reconstructor>
void test_prim_reconstructor(const size_t points_per_dimension,
                             const Reconstructor& in_derived_reconstructor) {
  PUPable_reg(Reconstructor);
  const auto base_recons =
      serialize_and_deserialize(in_derived_reconstructor.get_clone());
  const auto& derived_reconstructor =
      dynamic_cast<const Reconstructor&>(*base_recons);

  detail::test_prim_reconstructor_impl(points_per_dimension,
                                       derived_reconstructor,
                                       EquationsOfState::IdealFluid<true>{1.4});
  detail::test_prim_reconstructor_impl(
      points_per_dimension, derived_reconstructor,
      EquationsOfState::PolytropicFluid<true>{1.0, 2.0});
}
}  // namespace TestHelpers::grmhd::ValenciaDivClean::fd
