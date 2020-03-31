// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
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
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/CharacteristicHelpers.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Minmod.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

void test_neuler_minmod_option_parsing() noexcept {
  INFO("Testing option parsing");
  const auto lambda_pi1 =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Minmod<1>>(
          "Type: LambdaPi1\n"
          "VariablesToLimit: Characteristic\n"
          "TvbConstant: 0.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto lambda_pi1_cons =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Minmod<1>>(
          "Type: LambdaPi1\n"
          "VariablesToLimit: Conserved\n"
          "TvbConstant: 0.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto lambda_pi1_tvb =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Minmod<1>>(
          "Type: LambdaPi1\n"
          "VariablesToLimit: Characteristic\n"
          "TvbConstant: 1.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto lambda_pi1_noflat =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Minmod<1>>(
          "Type: LambdaPi1\n"
          "VariablesToLimit: Characteristic\n"
          "TvbConstant: 0.0\n"
          "ApplyFlattener: False\n"
          "DisableForDebugging: False");
  const auto lambda_pi1_disabled =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Minmod<1>>(
          "Type: LambdaPi1\n"
          "VariablesToLimit: Characteristic\n"
          "TvbConstant: 0.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: True");
  const auto muscl =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Minmod<1>>(
          "Type: Muscl\n"
          "VariablesToLimit: Characteristic\n"
          "TvbConstant: 0.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");

  // Test operators == and !=
  CHECK(lambda_pi1 == lambda_pi1);
  CHECK(lambda_pi1 != lambda_pi1_cons);
  CHECK(lambda_pi1 != lambda_pi1_tvb);
  CHECK(lambda_pi1 != lambda_pi1_noflat);
  CHECK(lambda_pi1 != lambda_pi1_disabled);
  CHECK(lambda_pi1 != muscl);

  const auto lambda_pin_2d =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Minmod<2>>(
          "Type: LambdaPiN\n"
          "VariablesToLimit: Characteristic\n"
          "TvbConstant: 0.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto lambda_pin_3d =
      TestHelpers::test_creation<NewtonianEuler::Limiters::Minmod<3>>(
          "Type: LambdaPiN\n"
          "VariablesToLimit: Characteristic\n"
          "TvbConstant: 10.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");

  // Test that creation from options gives correct object
  const NewtonianEuler::Limiters::Minmod<1> expected_lambda_pi1(
      Limiters::MinmodType::LambdaPi1,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0., true);
  const NewtonianEuler::Limiters::Minmod<1> expected_lambda_pi1_cons(
      Limiters::MinmodType::LambdaPi1,
      NewtonianEuler::Limiters::VariablesToLimit::Conserved, 0., true);
  const NewtonianEuler::Limiters::Minmod<1> expected_lambda_pi1_tvb(
      Limiters::MinmodType::LambdaPi1,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 1., true);
  const NewtonianEuler::Limiters::Minmod<1> expected_lambda_pi1_noflat(
      Limiters::MinmodType::LambdaPi1,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0., false);
  const NewtonianEuler::Limiters::Minmod<1> expected_lambda_pi1_disabled(
      Limiters::MinmodType::LambdaPi1,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0., true,
      true);
  const NewtonianEuler::Limiters::Minmod<1> expected_muscl(
      Limiters::MinmodType::Muscl,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0., true);
  const NewtonianEuler::Limiters::Minmod<2> expected_lambda_pin_2d(
      Limiters::MinmodType::LambdaPiN,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 0., true);
  const NewtonianEuler::Limiters::Minmod<3> expected_lambda_pin_3d(
      Limiters::MinmodType::LambdaPiN,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 10., true);

  CHECK(lambda_pi1 == expected_lambda_pi1);
  CHECK(lambda_pi1_cons == expected_lambda_pi1_cons);
  CHECK(lambda_pi1_tvb == expected_lambda_pi1_tvb);
  CHECK(lambda_pi1_noflat == expected_lambda_pi1_noflat);
  CHECK(lambda_pi1_disabled == expected_lambda_pi1_disabled);
  CHECK(muscl == expected_muscl);
  CHECK(lambda_pin_2d == expected_lambda_pin_2d);
  CHECK(lambda_pin_3d == expected_lambda_pin_3d);
}

void test_neuler_minmod_serialization() noexcept {
  INFO("Testing serialization");
  const NewtonianEuler::Limiters::Minmod<1> minmod(
      Limiters::MinmodType::LambdaPi1,
      NewtonianEuler::Limiters::VariablesToLimit::Characteristic, 1., true);
  test_serialization(minmod);
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
template <size_t VolumeDim>
void test_neuler_vs_generic_minmod_work(
    const Scalar<DataVector>& input_density,
    const tnsr::I<DataVector, VolumeDim>& input_momentum,
    const Scalar<DataVector>& input_energy, const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const Scalar<DataVector>& det_inv_logical_to_inertial_jacobian,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename NewtonianEuler::Limiters::Minmod<VolumeDim>::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) noexcept {
  // Sanity check that input momentum satisfies simplifying assumptions
  if constexpr (VolumeDim > 1) {
    const auto zero_momentum = make_with_value<tnsr::I<DataVector, VolumeDim>>(
        get<0>(input_momentum), 0.);
    REQUIRE(input_momentum == zero_momentum);
  }

  const double tvb_constant = 0.;
  const auto element = TestHelpers::Limiters::make_element<VolumeDim>();
  const EquationsOfState::IdealFluid<false> equation_of_state{5. / 3.};

  {
    INFO("Compare specialized vs generic Minmod on conserved variables");
    auto density_generic = input_density;
    auto momentum_generic = input_momentum;
    auto energy_generic = input_energy;
    const Limiters::Minmod<
        VolumeDim, tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                              NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                              NewtonianEuler::Tags::EnergyDensity>>
        minmod_generic(Limiters::MinmodType::LambdaPi1, tvb_constant);
    const bool activated_generic = minmod_generic(
        make_not_null(&density_generic), make_not_null(&momentum_generic),
        make_not_null(&energy_generic), mesh, element, logical_coords,
        element_size, neighbor_data);

    auto density_specialized = input_density;
    auto momentum_specialized = input_momentum;
    auto energy_specialized = input_energy;
    const auto vars_to_limit =
        NewtonianEuler::Limiters::VariablesToLimit::Conserved;
    const bool apply_flattener = false;
    const NewtonianEuler::Limiters::Minmod<VolumeDim> minmod_specialized(
        Limiters::MinmodType::LambdaPi1, vars_to_limit, tvb_constant,
        apply_flattener);
    const bool activated_specialized = minmod_specialized(
        make_not_null(&density_specialized),
        make_not_null(&momentum_specialized),
        make_not_null(&energy_specialized), mesh, element, logical_coords,
        element_size, det_inv_logical_to_inertial_jacobian, equation_of_state,
        neighbor_data);

    CHECK(activated_generic);
    CHECK(activated_generic == activated_specialized);
    CHECK_ITERABLE_APPROX(density_generic, density_specialized);
    CHECK_ITERABLE_APPROX(momentum_generic, momentum_specialized);
    CHECK_ITERABLE_APPROX(energy_generic, energy_specialized);
  }

  {
    INFO("Compare specialized vs generic Minmod on characteristic variables");
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
    // Note that in 2D and 3D we know the fluid velocity is 0, so this choice
    // is arbitrary in these higher dimensions.
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

    using GenericMinmod =
        Limiters::Minmod<VolumeDim,
                         tmpl::list<NewtonianEuler::Tags::VMinus,
                                    NewtonianEuler::Tags::VMomentum<VolumeDim>,
                                    NewtonianEuler::Tags::VPlus>>;
    std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename GenericMinmod::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
        neighbor_char_data{};
    for (const auto& [key, data] : neighbor_data) {
      neighbor_char_data[key].element_size = data.element_size;
      NewtonianEuler::Limiters::characteristic_fields(
          make_not_null(&(neighbor_char_data[key].means)), data.means, left);
    }

    const GenericMinmod minmod_generic(Limiters::MinmodType::LambdaPi1,
                                       tvb_constant);
    const bool activated_generic = minmod_generic(
        make_not_null(&char_v_minus), make_not_null(&char_v_momentum),
        make_not_null(&char_v_plus), mesh, element, logical_coords,
        element_size, neighbor_char_data);

    // Transform back to evolved variables
    // Note that because the fluid velocity is 0 and all sets of characteristics
    // are identical, we can skip the step of combining different limiter
    // results.
    auto density_generic = input_density;
    auto momentum_generic = input_momentum;
    auto energy_generic = input_energy;
    NewtonianEuler::Limiters::conserved_fields_from_characteristic_fields(
        make_not_null(&density_generic), make_not_null(&momentum_generic),
        make_not_null(&energy_generic), char_v_minus, char_v_momentum,
        char_v_plus, right);

    auto density_specialized = input_density;
    auto momentum_specialized = input_momentum;
    auto energy_specialized = input_energy;
    const auto vars_to_limit =
        NewtonianEuler::Limiters::VariablesToLimit::Characteristic;
    const bool apply_flattener = false;
    const NewtonianEuler::Limiters::Minmod<VolumeDim> minmod_specialized(
        Limiters::MinmodType::LambdaPi1, vars_to_limit, tvb_constant,
        apply_flattener);
    const bool activated_specialized = minmod_specialized(
        make_not_null(&density_specialized),
        make_not_null(&momentum_specialized),
        make_not_null(&energy_specialized), mesh, element, logical_coords,
        element_size, det_inv_logical_to_inertial_jacobian, equation_of_state,
        neighbor_data);

    CHECK(activated_generic);
    CHECK(activated_generic == activated_specialized);
    CHECK_ITERABLE_APPROX(density_generic, density_specialized);
    CHECK_ITERABLE_APPROX(momentum_generic, momentum_specialized);
    CHECK_ITERABLE_APPROX(energy_generic, energy_specialized);
  }
}

void test_neuler_vs_generic_minmod_1d() noexcept {
  INFO("Testing NewtonianEuler::Limiters::Minmod limiter in 1D");
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
                     NewtonianEuler::Limiters::Minmod<1>::PackagedData,
                     boost::hash<std::pair<Direction<1>, ElementId<1>>>>
      neighbor_data{};
  const std::array<std::pair<Direction<1>, ElementId<1>>, 2> dir_keys = {
      {{Direction<1>::lower_xi(), ElementId<1>(1)},
       {Direction<1>::upper_xi(), ElementId<1>(2)}}};
  for (const auto& dir_key : dir_keys) {
    neighbor_data[dir_key].element_size = element_size;
  }

  using mean_rho = Tags::Mean<NewtonianEuler::Tags::MassDensityCons>;
  get(get<mean_rho>(neighbor_data[dir_keys[0]].means)) = 0.4;
  get(get<mean_rho>(neighbor_data[dir_keys[1]].means)) = 1.1;

  using mean_rhou = Tags::Mean<NewtonianEuler::Tags::MomentumDensity<1>>;
  get<0>(get<mean_rhou>(neighbor_data[dir_keys[0]].means)) = 0.2;
  get<0>(get<mean_rhou>(neighbor_data[dir_keys[1]].means)) = -0.1;

  using mean_eps = Tags::Mean<NewtonianEuler::Tags::EnergyDensity>;
  get(get<mean_eps>(neighbor_data[dir_keys[0]].means)) = 1.;
  get(get<mean_eps>(neighbor_data[dir_keys[1]].means)) = 1.;

  test_neuler_vs_generic_minmod_work(
      mass_density_cons, momentum_density, energy_density, mesh, logical_coords,
      element_size, det_inv_logical_to_inertial_jacobian, neighbor_data);
}

void test_neuler_vs_generic_minmod_2d() noexcept {
  INFO("Testing NewtonianEuler::Limiters::Minmod limiter in 2D");
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

  const auto& x = get<0>(logical_coords);
  const auto& y = get<1>(logical_coords);
  const auto mass_density_cons = [&x, &y]() noexcept {
    return Scalar<DataVector>{
        {{1. + 0.2 * x - 0.1 * y + 0.05 * x * square(y)}}};
  }();
  const auto momentum_density = [&mesh]() noexcept {
    return tnsr::I<DataVector, 2>{DataVector(mesh.number_of_grid_points(), 0.)};
  }();
  const auto energy_density = [&mesh]() noexcept {
    return Scalar<DataVector>{DataVector(mesh.number_of_grid_points(), 1.)};
  }();

  std::unordered_map<std::pair<Direction<2>, ElementId<2>>,
                     NewtonianEuler::Limiters::Minmod<2>::PackagedData,
                     boost::hash<std::pair<Direction<2>, ElementId<2>>>>
      neighbor_data{};
  const std::array<std::pair<Direction<2>, ElementId<2>>, 4> dir_keys = {
      {{Direction<2>::lower_xi(), ElementId<2>(1)},
       {Direction<2>::upper_xi(), ElementId<2>(2)},
       {Direction<2>::lower_eta(), ElementId<2>(3)},
       {Direction<2>::upper_eta(), ElementId<2>(4)}}};
  for (const auto& dir_key : dir_keys) {
    neighbor_data[dir_key].element_size = element_size;
  }

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

  test_neuler_vs_generic_minmod_work(
      mass_density_cons, momentum_density, energy_density, mesh, logical_coords,
      element_size, det_inv_logical_to_inertial_jacobian, neighbor_data);
}

void test_neuler_vs_generic_minmod_3d() noexcept {
  INFO("Testing NewtonianEuler::Limiters::Minmod limiter in 3D");
  const auto mesh =
      Mesh<3>(std::array<size_t, 3>{{3, 3, 4}}, Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto);
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

  const auto& x = get<0>(logical_coords);
  const auto& y = get<1>(logical_coords);
  const auto& z = get<2>(logical_coords);
  const auto mass_density_cons = [&x, &y, &z]() noexcept {
    return Scalar<DataVector>{{{1. + 0.2 * x - 0.1 * y + 0.4 * z}}};
  }();
  const auto momentum_density = [&mesh]() noexcept {
    return tnsr::I<DataVector, 3>{DataVector(mesh.number_of_grid_points(), 0.)};
  }();
  const auto energy_density = [&mesh]() noexcept {
    return Scalar<DataVector>{DataVector(mesh.number_of_grid_points(), 1.)};
  }();

  std::unordered_map<std::pair<Direction<3>, ElementId<3>>,
                     NewtonianEuler::Limiters::Minmod<3>::PackagedData,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      neighbor_data{};
  const std::array<std::pair<Direction<3>, ElementId<3>>, 6> dir_keys = {
      {{Direction<3>::lower_xi(), ElementId<3>(1)},
       {Direction<3>::upper_xi(), ElementId<3>(2)},
       {Direction<3>::lower_eta(), ElementId<3>(3)},
       {Direction<3>::upper_eta(), ElementId<3>(4)},
       {Direction<3>::lower_zeta(), ElementId<3>(5)},
       {Direction<3>::upper_zeta(), ElementId<3>(6)}}};
  for (const auto& dir_key : dir_keys) {
    neighbor_data[dir_key].element_size = element_size;
  }

  using mean_rho = Tags::Mean<NewtonianEuler::Tags::MassDensityCons>;
  get(get<mean_rho>(neighbor_data[dir_keys[0]].means)) = 0.;
  get(get<mean_rho>(neighbor_data[dir_keys[1]].means)) = 0.;
  get(get<mean_rho>(neighbor_data[dir_keys[2]].means)) = 0.1;
  get(get<mean_rho>(neighbor_data[dir_keys[3]].means)) = 1.2;
  get(get<mean_rho>(neighbor_data[dir_keys[4]].means)) = 1.1;
  get(get<mean_rho>(neighbor_data[dir_keys[5]].means)) = 0.9;

  for (const auto& dir_key : dir_keys) {
    using mean_rhou = Tags::Mean<NewtonianEuler::Tags::MomentumDensity<3>>;
    get<0>(get<mean_rhou>(neighbor_data[dir_key].means)) = 0.;
    get<1>(get<mean_rhou>(neighbor_data[dir_key].means)) = 0.;
    get<2>(get<mean_rhou>(neighbor_data[dir_key].means)) = 0.;
    using mean_eps = Tags::Mean<NewtonianEuler::Tags::EnergyDensity>;
    get(get<mean_eps>(neighbor_data[dir_key].means)) = 1.;
  }

  test_neuler_vs_generic_minmod_work(
      mass_density_cons, momentum_density, energy_density, mesh, logical_coords,
      element_size, det_inv_logical_to_inertial_jacobian, neighbor_data);
}

template <size_t VolumeDim>
void test_neuler_minmod_flattener() noexcept {
  INFO("Testing flattener use in NewtonianEuler::Limiters::Minmod limiter");
  CAPTURE(VolumeDim);

  const auto mesh = Mesh<VolumeDim>(2, Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto);
  // We use an element with no neighbors so the limiter does nothing
  const auto element = Element<VolumeDim>{ElementId<VolumeDim>(0), {}};
  const auto logical_coords = logical_coordinates(mesh);
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
      typename NewtonianEuler::Limiters::Minmod<VolumeDim>::PackagedData,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      neighbor_data{};

  // First a sanity check: there should be a negative density for this test to
  // make sense... otherwise we aren't testing anything useful
  const bool has_negative_density = min(get(input_density)) < 0.;
  REQUIRE(has_negative_density);
  CHECK(true);

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
  const Limiters::Minmod<
      VolumeDim, tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                            NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                            NewtonianEuler::Tags::EnergyDensity>>
      minmod_noflat(Limiters::MinmodType::LambdaPi1, tvb_constant);
  const bool activated_noflat = minmod_noflat(
      make_not_null(&density), make_not_null(&momentum), make_not_null(&energy),
      mesh, element, logical_coords, element_size, neighbor_data);
  CHECK_FALSE(activated_noflat);

  // With flattener on, check limiter activates
  density = input_density;
  momentum = input_momentum;
  energy = input_energy;
  const auto vars_to_limit =
      NewtonianEuler::Limiters::VariablesToLimit::Conserved;
  const NewtonianEuler::Limiters::Minmod<VolumeDim> minmod(
      Limiters::MinmodType::LambdaPi1, vars_to_limit, tvb_constant, true);
  const bool activated = minmod(
      make_not_null(&density), make_not_null(&momentum), make_not_null(&energy),
      mesh, element, logical_coords, element_size,
      det_inv_logical_to_inertial_jacobian, equation_of_state, neighbor_data);
  CHECK(activated);

  // Finally, check the negative density is brought back to positive
  const bool density_is_positive_everywhere = min(get(density)) > 0.;
  CHECK(density_is_positive_everywhere);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Limiters.Minmod",
                  "[Limiters][Unit]") {
  test_neuler_minmod_option_parsing();
  test_neuler_minmod_serialization();

  // The package_data function for NewtonianEuler::Limiters::Minmod is just a
  // direct call to the generic Minmod package_data. We do not test it.

  // We test the action of NewtonianEuler::Limiters::Minmod by comparing it to
  // the action of the generic Minmod, for certain inputs where the comparison
  // is not too difficult. To fully test the specialized limiter, would need to
  // completely reimplement the logic within the test.
  test_neuler_vs_generic_minmod_1d();
  test_neuler_vs_generic_minmod_2d();
  test_neuler_vs_generic_minmod_3d();

  // Test the use of the post-processing flattener
  test_neuler_minmod_flattener<1>();
  test_neuler_minmod_flattener<2>();
  test_neuler_minmod_flattener<3>();
}
