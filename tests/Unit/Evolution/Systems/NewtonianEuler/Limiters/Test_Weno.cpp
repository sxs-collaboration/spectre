// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/CharacteristicHelpers.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Weno.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

void test_neuler_weno_option_parsing() noexcept {
  INFO("Testing option parsing");
  const auto sweno =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Weno<1>>(
          "Type: SimpleWeno\n"
          "VariablesToLimit: Characteristic\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: 0.0\n"
          "KxrcfConstant: None\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto sweno_nw =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Weno<1>>(
          "Type: SimpleWeno\n"
          "VariablesToLimit: Characteristic\n"
          "NeighborWeight: 0.02\n"
          "TvbConstant: 0.0\n"
          "KxrcfConstant: None\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto sweno_cons =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Weno<1>>(
          "Type: SimpleWeno\n"
          "VariablesToLimit: Conserved\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: 0.0\n"
          "KxrcfConstant: None\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto sweno_tvb =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Weno<1>>(
          "Type: SimpleWeno\n"
          "VariablesToLimit: Characteristic\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: 1.0\n"
          "KxrcfConstant: None\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto sweno_noflat =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Weno<1>>(
          "Type: SimpleWeno\n"
          "VariablesToLimit: Characteristic\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: 0.0\n"
          "KxrcfConstant: None\n"
          "ApplyFlattener: False\n"
          "DisableForDebugging: False");
  const auto sweno_disabled =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Weno<1>>(
          "Type: SimpleWeno\n"
          "VariablesToLimit: Characteristic\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: 0.0\n"
          "KxrcfConstant: None\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: True");
  const auto hweno =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Weno<1>>(
          "Type: Hweno\n"
          "VariablesToLimit: Characteristic\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: None\n"
          "KxrcfConstant: 0.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto hweno_kxrcf =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Weno<1>>(
          "Type: Hweno\n"
          "VariablesToLimit: Characteristic\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: None\n"
          "KxrcfConstant: 1.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");

  // Test operators == and !=
  CHECK(sweno == sweno);
  CHECK(sweno != sweno_nw);
  CHECK(sweno != sweno_cons);
  CHECK(sweno != sweno_tvb);
  CHECK(sweno != sweno_noflat);
  CHECK(sweno != sweno_disabled);
  CHECK(sweno != hweno);
  CHECK(hweno != hweno_kxrcf);

  const auto sweno_2d =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Weno<2>>(
          "Type: SimpleWeno\n"
          "VariablesToLimit: Characteristic\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: 0.0\n"
          "KxrcfConstant: None\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto sweno_3d =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Weno<3>>(
          "Type: SimpleWeno\n"
          "VariablesToLimit: Characteristic\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: 0.0\n"
          "KxrcfConstant: None\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");

  // Test that creation from options gives correct object
  const NewtonianEuler::Limiters::Weno<1> expected_sweno(
      Limiters::WenoType::SimpleWeno,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0.001, 0.0,
      {}, true);
  const NewtonianEuler::Limiters::Weno<1> expected_sweno_nw(
      Limiters::WenoType::SimpleWeno,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0.02, 0.0, {},
      true);
  const NewtonianEuler::Limiters::Weno<1> expected_sweno_cons(
      Limiters::WenoType::SimpleWeno,
      NewtonianEuler::Limiters::VariablesToLimit::Conserved, 0.001, 0.0, {},
      true);
  const NewtonianEuler::Limiters::Weno<1> expected_sweno_tvb(
      Limiters::WenoType::SimpleWeno,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0.001, 1.0,
      {}, true);
  const NewtonianEuler::Limiters::Weno<1> expected_sweno_noflat(
      Limiters::WenoType::SimpleWeno,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0.001, 0.0,
      {}, false);
  const NewtonianEuler::Limiters::Weno<1> expected_sweno_disabled(
      Limiters::WenoType::SimpleWeno,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0.001, 0.0,
      {}, true, true);
  const NewtonianEuler::Limiters::Weno<1> expected_hweno(
      Limiters::WenoType::Hweno,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0.001, {},
      0.0, true);
  const NewtonianEuler::Limiters::Weno<1> expected_hweno_kxrcf(
      Limiters::WenoType::Hweno,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0.001, {},
      1.0, true);
  const NewtonianEuler::Limiters::Weno<2> expected_sweno_2d(
      Limiters::WenoType::SimpleWeno,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0.001, 0.0,
      {}, true);
  const NewtonianEuler::Limiters::Weno<3> expected_sweno_3d(
      Limiters::WenoType::SimpleWeno,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0.001, 0.0,
      {}, true);

  CHECK(sweno == expected_sweno);
  CHECK(sweno_nw == expected_sweno_nw);
  CHECK(sweno_cons == expected_sweno_cons);
  CHECK(sweno_tvb == expected_sweno_tvb);
  CHECK(sweno_noflat == expected_sweno_noflat);
  CHECK(sweno_disabled == expected_sweno_disabled);
  CHECK(hweno == expected_hweno);
  CHECK(hweno_kxrcf == expected_hweno_kxrcf);
  CHECK(sweno_2d == expected_sweno_2d);
  CHECK(sweno_3d == expected_sweno_3d);
}

void test_neuler_weno_serialization() noexcept {
  INFO("Testing serialization");
  const NewtonianEuler::Limiters::Weno<1> weno(
      Limiters::WenoType::SimpleWeno,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0.001, 1., {},
      true);
  test_serialization(weno);
}

template <size_t Dim>
DirectionMap<Dim, std::optional<Variables<
                      tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                 evolution::dg::Tags::NormalCovector<Dim>>>>>
compute_normals_and_magnitudes(
    const Mesh<Dim>& mesh,
    const ElementMap<Dim, Frame::Inertial>& element_map) noexcept {
  DirectionMap<Dim, std::optional<Variables<
                        tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                   evolution::dg::Tags::NormalCovector<Dim>>>>>
      normals_and_magnitudes{};
  for (const auto& dir : Direction<Dim>::all_directions()) {
    const auto boundary_mesh = mesh.slice_away(dir.dimension());
    normals_and_magnitudes[dir] =
        Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                             evolution::dg::Tags::NormalCovector<Dim>>>(
            boundary_mesh.number_of_grid_points());
    auto& covector = get<evolution::dg::Tags::NormalCovector<Dim>>(
        normals_and_magnitudes[dir].value());
    auto& normal_magnitude = get<evolution::dg::Tags::MagnitudeOfNormal>(
        normals_and_magnitudes[dir].value());
    unnormalized_face_normal(make_not_null(&(covector)),
                             mesh.slice_away(dir.dimension()), element_map,
                             dir);
    normal_magnitude = magnitude(covector);
    for (size_t d = 0; d < Dim; ++d) {
      covector.get(d) /= get(normal_magnitude);
    }
  }
  return normals_and_magnitudes;
}

// Compare the specialized limiter to the generic limiter. The limiters are
// compared for limiting conserved/evolved and limiting characteristic
// variables.
//
// Note: comparing the specialized and generic limiters when limiting the
// characteristic variables is tedious to do thoroughly, because we need to
// call the generic limiter multiple times using characteristic fields w.r.t.
// multiple different unit vectors. In essence, the test would need to fully
// reimplement the specialized limiter. To avoid the test becoming so tedious,
// we test a simpler restricted problem, where in 2D and 3D we require
// input_momentum to be 0. With this simplification, the characteristic
// decomposition becomes independent of the choice of unit vector, and therefore
// the generic limiter can match the specialized limiter even when only one set
// of characteristic (e.g., w.r.t. \hat{x}) is used.
//
// However, this simplifying assumption doesn't play nicely with the HWENO
// limiter, because it uses the KXRCF TCI that relies on non-zero velocities to
// trigger the limiter... so we can use this function with HWENO only in 1D
// where we allow non-zero velocities.
template <size_t VolumeDim>
void test_neuler_vs_generic_weno_work(
    const Limiters::WenoType weno_type, const Scalar<DataVector>& input_density,
    const tnsr::I<DataVector, VolumeDim>& input_momentum,
    const Scalar<DataVector>& input_energy, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const Scalar<DataVector>& det_inv_logical_to_inertial_jacobian,
    const typename evolution::dg::Tags::NormalCovectorAndMagnitude<
        VolumeDim>::type& normals_and_magnitudes,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename NewtonianEuler::Limiters::Weno<VolumeDim>::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) noexcept {
  // Sanity check that
  // - input momentum satisfies simplifying assumptions
  // - limiter type is consistent with these simplifying assumptions
  if constexpr (VolumeDim > 1) {
    const auto zero_momentum = make_with_value<tnsr::I<DataVector, VolumeDim>>(
        get<0>(input_momentum), 0.);
    REQUIRE(input_momentum == zero_momentum);
    REQUIRE(weno_type == Limiters::WenoType::SimpleWeno);
  }

  CAPTURE(weno_type);

  const double neighbor_linear_weight = 0.001;
  const double tvb_constant = 0.;
  const double kxrcf_constant = 0.;
  const auto element = TestHelpers::Limiters::make_element<VolumeDim>();
  const EquationsOfState::IdealFluid<false> equation_of_state{5. / 3.};

  {
    INFO("Compare specialized vs generic Weno on conserved variables");
    auto density_generic = input_density;
    auto momentum_generic = input_momentum;
    auto energy_generic = input_energy;
    const Limiters::Weno<
        VolumeDim, tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                              NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                              NewtonianEuler::Tags::EnergyDensity>>
        weno_generic(weno_type, neighbor_linear_weight, tvb_constant);
    const bool activated_generic = weno_generic(
        make_not_null(&density_generic), make_not_null(&momentum_generic),
        make_not_null(&energy_generic), mesh, element, element_size,
        neighbor_data);

    std::optional<double> opt_tvb_constant{};
    std::optional<double> opt_kxrcf_constant{};
    if (weno_type == Limiters::WenoType::SimpleWeno) {
      opt_tvb_constant = tvb_constant;
    } else if (weno_type == Limiters::WenoType::Hweno) {
      opt_kxrcf_constant = kxrcf_constant;
    }
    const bool apply_flattener = false;

    auto density_specialized = input_density;
    auto momentum_specialized = input_momentum;
    auto energy_specialized = input_energy;
    const auto vars_to_limit =
        NewtonianEuler::Limiters::VariablesToLimit::Conserved;
    const NewtonianEuler::Limiters::Weno<VolumeDim> weno_specialized(
        weno_type, vars_to_limit, neighbor_linear_weight, opt_tvb_constant,
        opt_kxrcf_constant, apply_flattener);
    const bool activated_specialized = weno_specialized(
        make_not_null(&density_specialized),
        make_not_null(&momentum_specialized),
        make_not_null(&energy_specialized), mesh, element, element_size,
        det_inv_logical_to_inertial_jacobian, normals_and_magnitudes,
        equation_of_state, neighbor_data);

    CHECK(activated_generic);
    CHECK(activated_generic == activated_specialized);
    CHECK_ITERABLE_APPROX(density_generic, density_specialized);
    CHECK_ITERABLE_APPROX(momentum_generic, momentum_specialized);
    CHECK_ITERABLE_APPROX(energy_generic, energy_specialized);
  }

  {
    INFO("Compare specialized vs generic Weno on characteristic variables");
    // Cellwise means, for cons/char transforms
    const auto mean_density =
        Scalar<double>{mean_value(get(input_density), mesh)};
    const auto mean_momentum = [&input_momentum, &mesh]() noexcept {
      tnsr::I<double, VolumeDim> result{};
      for (size_t i = 0; i < VolumeDim; ++i) {
        result.get(i) = mean_value(input_momentum.get(i), mesh);
      }
      return result;
    }();
    const auto mean_energy =
        Scalar<double>{mean_value(get(input_energy), mesh)};

    // Compute characteristic transformation using x-direction...
    // Note that in 2D and 3D we know the fluid velocity is 0, so this
    // choice is arbitrary in these higher dimensions.
    const auto unit_vector = []() noexcept {
      auto components = make_array<VolumeDim>(0.);
      components[0] = 1.;
      return tnsr::i<double, VolumeDim>(components);
    }();
    const auto right_and_left =
        NewtonianEuler::Limiters::right_and_left_eigenvectors(
            mean_density, mean_momentum, mean_energy, equation_of_state,
            unit_vector);
    const auto& right = right_and_left.first;
    const auto& left = right_and_left.second;

    // Transform tensors to characteristics
    auto char_v_minus = input_density;
    auto char_v_momentum = input_momentum;
    auto char_v_plus = input_energy;
    NewtonianEuler::Limiters::characteristic_fields(
        make_not_null(&char_v_minus), make_not_null(&char_v_momentum),
        make_not_null(&char_v_plus), input_density, input_momentum,
        input_energy, left);

    using GenericWeno =
        Limiters::Weno<VolumeDim,
                       tmpl::list<NewtonianEuler::Tags::VMinus,
                                  NewtonianEuler::Tags::VMomentum<VolumeDim>,
                                  NewtonianEuler::Tags::VPlus>>;
    std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename GenericWeno::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
        neighbor_char_data{};
    for (const auto& [key, data] : neighbor_data) {
      neighbor_char_data[key].element_size = data.element_size;
      neighbor_char_data[key].mesh = data.mesh;
      NewtonianEuler::Limiters::characteristic_fields(
          make_not_null(&(neighbor_char_data[key].means)), data.means, left);
      neighbor_char_data[key].volume_data.initialize(
          mesh.number_of_grid_points());
      NewtonianEuler::Limiters::characteristic_fields(
          make_not_null(&(neighbor_char_data[key].volume_data)),
          data.volume_data, left);
    }

    const GenericWeno weno_generic(weno_type, neighbor_linear_weight,
                                   tvb_constant);
    const bool activated_generic = weno_generic(
        make_not_null(&char_v_minus), make_not_null(&char_v_momentum),
        make_not_null(&char_v_plus), mesh, element, element_size,
        neighbor_char_data);

    // Transform back to evolved variables
    // Note that because the fluid velocity is 0 and all sets of
    // characteristics are identical, we can skip the step of combining
    // different limiter results.
    auto density_generic = input_density;
    auto momentum_generic = input_momentum;
    auto energy_generic = input_energy;
    NewtonianEuler::Limiters::conserved_fields_from_characteristic_fields(
        make_not_null(&density_generic), make_not_null(&momentum_generic),
        make_not_null(&energy_generic), char_v_minus, char_v_momentum,
        char_v_plus, right);

    std::optional<double> opt_tvb_constant{};
    std::optional<double> opt_kxrcf_constant{};
    if (weno_type == Limiters::WenoType::SimpleWeno) {
      opt_tvb_constant = tvb_constant;
    } else if (weno_type == Limiters::WenoType::Hweno) {
      opt_kxrcf_constant = kxrcf_constant;
    }
    const bool apply_flattener = false;

    auto density_specialized = input_density;
    auto momentum_specialized = input_momentum;
    auto energy_specialized = input_energy;
    const auto vars_to_limit =
        NewtonianEuler::Limiters::VariablesToLimit::Characteristic;
    const NewtonianEuler::Limiters::Weno<VolumeDim> weno_specialized(
        weno_type, vars_to_limit, neighbor_linear_weight, opt_tvb_constant,
        opt_kxrcf_constant, apply_flattener);
    const bool activated_specialized = weno_specialized(
        make_not_null(&density_specialized),
        make_not_null(&momentum_specialized),
        make_not_null(&energy_specialized), mesh, element, element_size,
        det_inv_logical_to_inertial_jacobian, normals_and_magnitudes,
        equation_of_state, neighbor_data);

    CHECK(activated_generic);
    CHECK(activated_generic == activated_specialized);
    CHECK_ITERABLE_APPROX(density_generic, density_specialized);
    CHECK_ITERABLE_APPROX(momentum_generic, momentum_specialized);
    CHECK_ITERABLE_APPROX(energy_generic, energy_specialized);
  }
}

void test_neuler_vs_generic_weno_1d() noexcept {
  INFO("Testing NewtonianEuler::Limiters::Weno limiter in 1D");
  const auto mesh =
      Mesh<1>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto element = TestHelpers::Limiters::make_element<1>();
  const auto logical_coords = logical_coordinates(mesh);
  using Affine = domain::CoordinateMaps::Affine;
  const Affine xi_map{-1., 1., 3.7, 4.2};
  const auto coordmap =
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(xi_map);
  const ElementMap<1, Frame::Inertial> element_map(element.id(),
                                                   coordmap->get_clone());
  const auto element_size = size_of_element(element_map);
  const auto det_inv_logical_to_inertial_jacobian =
      determinant(element_map.inv_jacobian(logical_coords));
  const auto normals_and_magnitudes =
      compute_normals_and_magnitudes(mesh, element_map);

  const auto& x = get<0>(logical_coords);
  const auto mass_density_cons = [&x]() noexcept {
    return Scalar<DataVector>{{{1. + 0.2 * x + 0.05 * square(x)}}};
  }();
  const auto momentum_density = [&x]() noexcept {
    return tnsr::I<DataVector, 1>{{{0.2 - 0.3 * x}}};
  }();
  const auto energy_density = [&mesh]() noexcept {
    return Scalar<DataVector>{DataVector(mesh.number_of_grid_points(), 1.)};
  }();

  std::unordered_map<std::pair<Direction<1>, ElementId<1>>,
                     NewtonianEuler::Limiters::Weno<1>::PackagedData,
                     boost::hash<std::pair<Direction<1>, ElementId<1>>>>
      neighbor_data{};
  const std::array<std::pair<Direction<1>, ElementId<1>>, 2> dir_keys = {
      {{Direction<1>::lower_xi(), ElementId<1>(1)},
       {Direction<1>::upper_xi(), ElementId<1>(2)}}};
  const size_t num_pts = mesh.number_of_grid_points();
  for (const auto& dir_key : dir_keys) {
    neighbor_data[dir_key].element_size = element_size;
    neighbor_data[dir_key].mesh = mesh;
    neighbor_data[dir_key].volume_data.initialize(num_pts, 0.0);
  }

  // Note: the WENO algorithms are tested in detail elsewhere, so here we don't
  // go through the trouble of setting up realistic neighbor data that varies
  // over the neighbor's grid. Instead we pass in constant volume data because
  // it's a lot easier to set up.
  using rho = NewtonianEuler::Tags::MassDensityCons;
  get(get<rho>(neighbor_data[dir_keys[0]].volume_data)) = 0.4;
  get(get<rho>(neighbor_data[dir_keys[1]].volume_data)) = 1.1;
  using mean_rho = Tags::Mean<NewtonianEuler::Tags::MassDensityCons>;
  get(get<mean_rho>(neighbor_data[dir_keys[0]].means)) = 0.4;
  get(get<mean_rho>(neighbor_data[dir_keys[1]].means)) = 1.1;

  using rhou = NewtonianEuler::Tags::MomentumDensity<1>;
  get<0>(get<rhou>(neighbor_data[dir_keys[0]].volume_data)) = 0.2;
  get<0>(get<rhou>(neighbor_data[dir_keys[1]].volume_data)) = -0.1;
  using mean_rhou = Tags::Mean<NewtonianEuler::Tags::MomentumDensity<1>>;
  get<0>(get<mean_rhou>(neighbor_data[dir_keys[0]].means)) = 0.2;
  get<0>(get<mean_rhou>(neighbor_data[dir_keys[1]].means)) = -0.1;

  using eps = NewtonianEuler::Tags::EnergyDensity;
  get(get<eps>(neighbor_data[dir_keys[0]].volume_data)) = 1.;
  get(get<eps>(neighbor_data[dir_keys[1]].volume_data)) = 1.;
  using mean_eps = Tags::Mean<NewtonianEuler::Tags::EnergyDensity>;
  get(get<mean_eps>(neighbor_data[dir_keys[0]].means)) = 1.;
  get(get<mean_eps>(neighbor_data[dir_keys[1]].means)) = 1.;

  test_neuler_vs_generic_weno_work(
      Limiters::WenoType::SimpleWeno, mass_density_cons, momentum_density,
      energy_density, mesh, element_size, det_inv_logical_to_inertial_jacobian,
      normals_and_magnitudes, neighbor_data);

  test_neuler_vs_generic_weno_work(
      Limiters::WenoType::Hweno, mass_density_cons, momentum_density,
      energy_density, mesh, element_size, det_inv_logical_to_inertial_jacobian,
      normals_and_magnitudes, neighbor_data);
}

void test_neuler_vs_generic_weno_2d() noexcept {
  INFO("Testing NewtonianEuler::Limiters::Weno limiter in 2D");
  const auto mesh =
      Mesh<2>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto element = TestHelpers::Limiters::make_element<2>();
  const auto logical_coords = logical_coordinates(mesh);
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  const Affine xi_map{-1., 1., 3.7, 4.2};
  const Affine eta_map{-1., 1., 3.2, 4.2};
  const Affine2D map2d{xi_map, eta_map};
  const auto coordmap =
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(map2d);
  const ElementMap<2, Frame::Inertial> element_map(element.id(),
                                                   coordmap->get_clone());
  const auto element_size = size_of_element(element_map);
  const auto det_inv_logical_to_inertial_jacobian =
      determinant(element_map.inv_jacobian(logical_coords));
  const auto normals_and_magnitudes =
      compute_normals_and_magnitudes(mesh, element_map);

  const auto& x = get<0>(logical_coords);
  const auto& y = get<1>(logical_coords);
  const auto mass_density_cons = [&x, &y]() noexcept {
    return Scalar<DataVector>{
        {{1. + 0.2 * x - 0.1 * y + 0.05 * x * square(y)}}};
  }();
  const auto momentum_density = [&mesh]() noexcept {
    return tnsr::I<DataVector, 2>{DataVector(mesh.number_of_grid_points(), 0.)};
  }();
  const auto energy_density = [&x, &y]() noexcept {
    return Scalar<DataVector>{{{1.3 + 0.1 * y - 0.06 * square(x) * y}}};
  }();

  std::unordered_map<std::pair<Direction<2>, ElementId<2>>,
                     NewtonianEuler::Limiters::Weno<2>::PackagedData,
                     boost::hash<std::pair<Direction<2>, ElementId<2>>>>
      neighbor_data{};
  const std::array<std::pair<Direction<2>, ElementId<2>>, 4> dir_keys = {
      {{Direction<2>::lower_xi(), ElementId<2>(1)},
       {Direction<2>::upper_xi(), ElementId<2>(2)},
       {Direction<2>::lower_eta(), ElementId<2>(3)},
       {Direction<2>::upper_eta(), ElementId<2>(4)}}};
  const size_t num_pts = mesh.number_of_grid_points();
  for (const auto& dir_key : dir_keys) {
    neighbor_data[dir_key].element_size = element_size;
    neighbor_data[dir_key].mesh = mesh;
    neighbor_data[dir_key].volume_data.initialize(num_pts, 0.0);
  }

  using rho = NewtonianEuler::Tags::MassDensityCons;
  get(get<rho>(neighbor_data[dir_keys[0]].volume_data)) = 0.4;
  get(get<rho>(neighbor_data[dir_keys[1]].volume_data)) = 1.1;
  get(get<rho>(neighbor_data[dir_keys[2]].volume_data)) = 2.1;
  get(get<rho>(neighbor_data[dir_keys[3]].volume_data)) = 0.9;
  using mean_rho = Tags::Mean<NewtonianEuler::Tags::MassDensityCons>;
  get(get<mean_rho>(neighbor_data[dir_keys[0]].means)) = 0.4;
  get(get<mean_rho>(neighbor_data[dir_keys[1]].means)) = 1.1;
  get(get<mean_rho>(neighbor_data[dir_keys[2]].means)) = 2.1;
  get(get<mean_rho>(neighbor_data[dir_keys[3]].means)) = 0.9;

  for (const auto& dir_key : dir_keys) {
    using mean_rhou = Tags::Mean<NewtonianEuler::Tags::MomentumDensity<2>>;
    get<0>(get<mean_rhou>(neighbor_data[dir_key].means)) = 0.;
    get<1>(get<mean_rhou>(neighbor_data[dir_key].means)) = 0.;
    using mean_eps = Tags::Mean<NewtonianEuler::Tags::EnergyDensity>;
    get(get<mean_eps>(neighbor_data[dir_key].means)) = 1.;
  }

  test_neuler_vs_generic_weno_work(
      Limiters::WenoType::SimpleWeno, mass_density_cons, momentum_density,
      energy_density, mesh, element_size, det_inv_logical_to_inertial_jacobian,
      normals_and_magnitudes, neighbor_data);
}

void test_neuler_vs_generic_weno_3d() noexcept {
  INFO("Testing NewtonianEuler::Limiters::Weno limiter in 3D");
  const auto mesh =
      Mesh<3>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto element = TestHelpers::Limiters::make_element<3>();
  const auto logical_coords = logical_coordinates(mesh);
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const Affine xi_map{-1., 1., 3.7, 4.2};
  const Affine eta_map{-1., 1., 3.2, 4.2};
  const Affine zeta_map{-1., 1., 1.3, 2.1};
  const Affine3D map3d{xi_map, eta_map, zeta_map};
  const auto coordmap =
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(map3d);
  const ElementMap<3, Frame::Inertial> element_map(element.id(),
                                                   coordmap->get_clone());
  const auto element_size = size_of_element(element_map);
  const auto det_inv_logical_to_inertial_jacobian =
      determinant(element_map.inv_jacobian(logical_coords));
  const auto normals_and_magnitudes =
      compute_normals_and_magnitudes(mesh, element_map);

  const auto& x = get<0>(logical_coords);
  const auto& y = get<1>(logical_coords);
  const auto& z = get<2>(logical_coords);
  const auto mass_density_cons = [&x, &y, &z]() noexcept {
    return Scalar<DataVector>{{{1. + 0.2 * x - 0.1 * y + 0.4 * z}}};
  }();
  const auto momentum_density = [&mesh]() noexcept {
    return tnsr::I<DataVector, 3>{DataVector(mesh.number_of_grid_points(), 0.)};
  }();
  const auto energy_density = [&x, &y, &z]() noexcept {
    return Scalar<DataVector>{{{1.8 - 0.1 * square(x) * y * square(z)}}};
  }();

  std::unordered_map<std::pair<Direction<3>, ElementId<3>>,
                     NewtonianEuler::Limiters::Weno<3>::PackagedData,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      neighbor_data{};
  const std::array<std::pair<Direction<3>, ElementId<3>>, 6> dir_keys = {
      {{Direction<3>::lower_xi(), ElementId<3>(1)},
       {Direction<3>::upper_xi(), ElementId<3>(2)},
       {Direction<3>::lower_eta(), ElementId<3>(3)},
       {Direction<3>::upper_eta(), ElementId<3>(4)},
       {Direction<3>::lower_zeta(), ElementId<3>(5)},
       {Direction<3>::upper_zeta(), ElementId<3>(6)}}};
  const size_t num_pts = mesh.number_of_grid_points();
  for (const auto& dir_key : dir_keys) {
    neighbor_data[dir_key].element_size = element_size;
    neighbor_data[dir_key].mesh = mesh;
    neighbor_data[dir_key].volume_data.initialize(num_pts, 0.0);
  }

  using rho = NewtonianEuler::Tags::MassDensityCons;
  get(get<rho>(neighbor_data[dir_keys[0]].volume_data)) = 0.4;
  get(get<rho>(neighbor_data[dir_keys[1]].volume_data)) = 1.1;
  get(get<rho>(neighbor_data[dir_keys[2]].volume_data)) = 2.1;
  get(get<rho>(neighbor_data[dir_keys[3]].volume_data)) = 0.9;
  get(get<rho>(neighbor_data[dir_keys[4]].volume_data)) = 0.3;
  get(get<rho>(neighbor_data[dir_keys[5]].volume_data)) = 1.3;
  using mean_rho = Tags::Mean<NewtonianEuler::Tags::MassDensityCons>;
  get(get<mean_rho>(neighbor_data[dir_keys[0]].means)) = 0.4;
  get(get<mean_rho>(neighbor_data[dir_keys[1]].means)) = 1.1;
  get(get<mean_rho>(neighbor_data[dir_keys[2]].means)) = 2.1;
  get(get<mean_rho>(neighbor_data[dir_keys[3]].means)) = 0.9;
  get(get<mean_rho>(neighbor_data[dir_keys[4]].means)) = 0.3;
  get(get<mean_rho>(neighbor_data[dir_keys[5]].means)) = 1.3;

  for (const auto& dir_key : dir_keys) {
    using rhou = NewtonianEuler::Tags::MomentumDensity<3>;
    get<0>(get<rhou>(neighbor_data[dir_key].volume_data)) = 0.;
    get<1>(get<rhou>(neighbor_data[dir_key].volume_data)) = 0.;
    using mean_rhou = Tags::Mean<NewtonianEuler::Tags::MomentumDensity<3>>;
    get<0>(get<mean_rhou>(neighbor_data[dir_key].means)) = 0.;
    get<1>(get<mean_rhou>(neighbor_data[dir_key].means)) = 0.;
    using eps = NewtonianEuler::Tags::EnergyDensity;
    get(get<eps>(neighbor_data[dir_key].volume_data)) = 1.;
    using mean_eps = Tags::Mean<NewtonianEuler::Tags::EnergyDensity>;
    get(get<mean_eps>(neighbor_data[dir_key].means)) = 1.;
  }

  test_neuler_vs_generic_weno_work(
      Limiters::WenoType::SimpleWeno, mass_density_cons, momentum_density,
      energy_density, mesh, element_size, det_inv_logical_to_inertial_jacobian,
      normals_and_magnitudes, neighbor_data);
}

template <size_t VolumeDim>
void test_neuler_weno_flattener() noexcept {
  INFO("Testing flattener use in NewtonianEuler::Limiters::Weno limiter");
  CAPTURE(VolumeDim);

  const auto mesh = Mesh<VolumeDim>(2, Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto);
  // We use an element with no neighbors so the limiter does nothing
  const auto element = Element<VolumeDim>{ElementId<VolumeDim>(0), {}};
  const auto element_size = make_array<VolumeDim>(0.6);
  // inv_jac = logical volume / inertial volume
  const Scalar<DataVector> det_inv_logical_to_inertial_jacobian{DataVector(
      mesh.number_of_grid_points(), pow<VolumeDim>(2.) / pow<VolumeDim>(0.6))};
  const EquationsOfState::IdealFluid<false> equation_of_state{5. / 3.};

  const size_t num_points = mesh.number_of_grid_points();
  const auto input_density = [&num_points]() noexcept {
    // One negative value to trigger flattener
    Scalar<DataVector> density{DataVector(num_points, 0.8)};
    get(density)[0] = -0.2;
    return density;
  }();
  const auto input_momentum = [&num_points]() noexcept {
    return tnsr::I<DataVector, VolumeDim>{DataVector(num_points, 0.1)};
  }();
  const auto input_energy = [&num_points]() noexcept {
    return Scalar<DataVector>{DataVector(num_points, 1.4)};
  }();

  // Empty map because no neighbors
  const std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      typename NewtonianEuler::Limiters::Weno<VolumeDim>::PackagedData,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      neighbor_data{};

  // First a sanity check: there should be a negative density for this test to
  // make sense... otherwise we aren't testing anything useful
  const bool has_negative_density = min(get(input_density)) < 0.;
  REQUIRE(has_negative_density);

  // With flattener off, check the limiter does nothing (because no neighbors).
  //
  // Note that the post-limiter result is unphysical: the density is still
  // negative at one point. Because of this, we have to use the generic limiter
  // when testing the limiter does nothing: the specialized limiter has an
  // ASSERT that the post-limiting solution is valid which would fail here.
  auto density = input_density;
  auto momentum = input_momentum;
  auto energy = input_energy;
  const double tvb_constant = 0.;
  const Limiters::Weno<
      VolumeDim, tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                            NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                            NewtonianEuler::Tags::EnergyDensity>>
      weno_noflat(Limiters::WenoType::SimpleWeno, 0.001, tvb_constant, {});
  const bool activated_noflat = weno_noflat(
      make_not_null(&density), make_not_null(&momentum), make_not_null(&energy),
      mesh, element, element_size, neighbor_data);
  CHECK_FALSE(activated_noflat);

  // Not read from, so empty
  const DirectionMap<VolumeDim,
                     std::optional<Variables<tmpl::list<
                         evolution::dg::Tags::MagnitudeOfNormal,
                         evolution::dg::Tags::NormalCovector<VolumeDim>>>>>
      normals_and_magnitudes{};

  // With flattener on, check limiter activates
  density = input_density;
  momentum = input_momentum;
  energy = input_energy;
  const auto vars_to_limit =
      NewtonianEuler::Limiters::VariablesToLimit::Conserved;
  const NewtonianEuler::Limiters::Weno<VolumeDim> weno(
      Limiters::WenoType::SimpleWeno, vars_to_limit, 0.001, tvb_constant, {},
      true);
  const bool activated = weno(
      make_not_null(&density), make_not_null(&momentum), make_not_null(&energy),
      mesh, element, element_size, det_inv_logical_to_inertial_jacobian,
      normals_and_magnitudes, equation_of_state, neighbor_data);
  CHECK(activated);

  // Finally, check the negative density is brought back to positive
  const bool density_is_positive_everywhere = min(get(density)) > 0.;
  CHECK(density_is_positive_everywhere);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Limiters.Weno",
                  "[Limiters][Unit]") {
  test_neuler_weno_option_parsing();
  test_neuler_weno_serialization();

  // The package_data function for NewtonianEuler::Limiters::Weno is just a
  // direct call to the generic Weno package_data. We do not test it.

  // Compare specialized vs generic Weno limiters on simplified input data.
  // The SimpleWeno implementation is tested in all dims, but Hweno is tested
  // in 1D only because its TCI isn't compatible with the simplified input data.
  test_neuler_vs_generic_weno_1d();
  test_neuler_vs_generic_weno_2d();
  test_neuler_vs_generic_weno_3d();

  // Test the use of the post-processing flattener
  test_neuler_weno_flattener<1>();
  test_neuler_weno_flattener<2>();
  test_neuler_weno_flattener<3>();
}
