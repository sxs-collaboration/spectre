// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Flattener.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

void test_flattener_1d() noexcept {
  INFO("Testing flatten_solution in 1D");
  const Mesh<1> mesh(4, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const EquationsOfState::IdealFluid<false> equation_of_state{5. / 3.};
  const Scalar<DataVector> det_logical_to_inertial_jacobian(
      DataVector{{2., 2., 2., 2.}});

  {
    INFO("Case: NoOp");
    Scalar<DataVector> mass_density_cons(DataVector{{0.8, 1.6, 2.2, 1.2}});
    tnsr::I<DataVector, 1> momentum_density(DataVector{{-1., 0., 0., -2.}});
    Scalar<DataVector> energy_density(DataVector{{1., 1.6, 1.9, 1.9}});

    const auto expected_mass_density_cons = mass_density_cons;
    const auto expected_momentum_density = momentum_density;
    const auto expected_energy_density = energy_density;

    const auto flattener_action = NewtonianEuler::Limiters::flatten_solution(
        make_not_null(&mass_density_cons), make_not_null(&momentum_density),
        make_not_null(&energy_density), mesh, det_logical_to_inertial_jacobian,
        equation_of_state);

    CHECK(flattener_action == NewtonianEuler::Limiters::FlattenerAction::NoOp);
    CHECK_ITERABLE_APPROX(mass_density_cons, expected_mass_density_cons);
    CHECK_ITERABLE_APPROX(momentum_density, expected_momentum_density);
    CHECK_ITERABLE_APPROX(energy_density, expected_energy_density);
  }

  {
    INFO("Case: ScaledSolution because of negative density");
    Scalar<DataVector> mass_density_cons(DataVector{{0.8, -0.25, 2.25, 1.2}});
    tnsr::I<DataVector, 1> momentum_density(DataVector{{-1., 0., 0., -2.}});
    Scalar<DataVector> energy_density(DataVector{{1., 1.6, 1.9, 1.9}});

    // Expect flattening factor (about mean) of 0.8 times the 0.95 safety = 0.76
    // density mean = 1.
    // momentum mean = -0.25
    // energy mean = 1.7
    const Scalar<DataVector> expected_mass_density_cons(
        DataVector{{0.848, 0.05, 1.95, 1.152}});
    const tnsr::I<DataVector, 1> expected_momentum_density(
        DataVector{{-0.82, -0.06, -0.06, -1.58}});
    const Scalar<DataVector> expected_energy_density(
        DataVector{{1.168, 1.624, 1.852, 1.852}});

    const auto flattener_action = NewtonianEuler::Limiters::flatten_solution(
        make_not_null(&mass_density_cons), make_not_null(&momentum_density),
        make_not_null(&energy_density), mesh, det_logical_to_inertial_jacobian,
        equation_of_state);

    CHECK(flattener_action ==
          NewtonianEuler::Limiters::FlattenerAction::ScaledSolution);
    CHECK_ITERABLE_APPROX(mass_density_cons, expected_mass_density_cons);
    CHECK_ITERABLE_APPROX(momentum_density, expected_momentum_density);
    CHECK_ITERABLE_APPROX(energy_density, expected_energy_density);
  }

  {
    INFO("Case: SetSolutionToMean because of negative pressure");
    Scalar<DataVector> mass_density_cons(DataVector{{0.8, -0.25, 2.25, 1.2}});
    tnsr::I<DataVector, 1> momentum_density(DataVector{{-1., 0., 0., -2.}});
    Scalar<DataVector> energy_density(DataVector{{1., 1.6, 1.9, 0.7}});

    // density mean = 1.
    // momentum mean = -0.25
    // energy mean = 1.6
    const auto expected_mass_density_cons =
        make_with_value<Scalar<DataVector>>(mass_density_cons, 1.);
    const auto expected_momentum_density =
        make_with_value<tnsr::I<DataVector, 1>>(momentum_density, -0.25);
    const auto expected_energy_density =
        make_with_value<Scalar<DataVector>>(energy_density, 1.6);

    const auto flattener_action = NewtonianEuler::Limiters::flatten_solution(
        make_not_null(&mass_density_cons), make_not_null(&momentum_density),
        make_not_null(&energy_density), mesh, det_logical_to_inertial_jacobian,
        equation_of_state);

    CHECK(flattener_action ==
          NewtonianEuler::Limiters::FlattenerAction::SetSolutionToMean);
    CHECK_ITERABLE_APPROX(mass_density_cons, expected_mass_density_cons);
    CHECK_ITERABLE_APPROX(momentum_density, expected_momentum_density);
    CHECK_ITERABLE_APPROX(energy_density, expected_energy_density);
  }

  {
    INFO("Case: ScaledSolution again, now with non-constant jacobian");
    const Scalar<DataVector> det_curved_jacobian(
        DataVector{{1., 1.1, 1.2, 1.3}});
    Scalar<DataVector> mass_density_cons(DataVector{{0.8, -0.25, 2.25, 1.2}});
    tnsr::I<DataVector, 1> momentum_density(DataVector{{-1., 0., 0., -2.}});
    Scalar<DataVector> energy_density(DataVector{{1., 1.6, 1.9, 1.9}});

    // With this jacobian, the means are (from Mathematica)
    // density mean = 2897 / 2769
    // momentum mean = -3 / 23
    // energy mean = 789 / 460
    // And the scaling factor (before 0.95 safety) is 2897 / 3587
    // Again from Mathematica, we get the expected results post-flattening...
    const Scalar<DataVector> expected_mass_density_cons(
        DataVector{{0.8581014826082916, 0.05248188405797101, 1.970623785368258,
                    1.165004186817938}});
    const tnsr::I<DataVector, 1> expected_momentum_density(
        DataVector{{-0.8279723882134762, -0.06071562768936134,
                    -0.06071562768936134, -1.595229148737591}});
    const Scalar<DataVector> expected_energy_density(
        DataVector{{1.166462012581666, 1.626816068896135, 1.856993097053369,
                    1.856993097053369}});

    const auto flattener_action = NewtonianEuler::Limiters::flatten_solution(
        make_not_null(&mass_density_cons), make_not_null(&momentum_density),
        make_not_null(&energy_density), mesh, det_curved_jacobian,
        equation_of_state);

    CHECK(flattener_action ==
          NewtonianEuler::Limiters::FlattenerAction::ScaledSolution);
    CHECK_ITERABLE_APPROX(mass_density_cons, expected_mass_density_cons);
    CHECK_ITERABLE_APPROX(momentum_density, expected_momentum_density);
    CHECK_ITERABLE_APPROX(energy_density, expected_energy_density);
  }
}

void test_flattener_2d() noexcept {
  INFO("Testing flatten_solution in 2D");
  const Mesh<2> mesh({{2, 3}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const EquationsOfState::IdealFluid<false> equation_of_state{5. / 3.};
  const Scalar<DataVector> det_logical_to_inertial_jacobian(
      DataVector{{2., 2.1, 2.1, 2.2, 2.2, 2.3}});

  {
    INFO("Case: NoOp");
    Scalar<DataVector> mass_density_cons(
        DataVector{{0.8, 1., 0.6, 1.3, 1.1, 1.6}});
    tnsr::I<DataVector, 2> momentum_density;
    get<0>(momentum_density) = DataVector{{-0.2, 0.1, 0.2, -0.1, 0.1, -0.2}};
    get<1>(momentum_density) = DataVector{{0.4, 0.3, 0.2, 0.3, 0.4, 0.7}};
    Scalar<DataVector> energy_density(DataVector{{1., 1.6, 1.4, 2., 1.8, 1.9}});

    const auto expected_mass_density_cons = mass_density_cons;
    const auto expected_momentum_density = momentum_density;
    const auto expected_energy_density = energy_density;

    const auto flattener_action = NewtonianEuler::Limiters::flatten_solution(
        make_not_null(&mass_density_cons), make_not_null(&momentum_density),
        make_not_null(&energy_density), mesh, det_logical_to_inertial_jacobian,
        equation_of_state);

    CHECK(flattener_action == NewtonianEuler::Limiters::FlattenerAction::NoOp);
    CHECK_ITERABLE_APPROX(mass_density_cons, expected_mass_density_cons);
    CHECK_ITERABLE_APPROX(momentum_density, expected_momentum_density);
    CHECK_ITERABLE_APPROX(energy_density, expected_energy_density);
  }

  {
    INFO("Case: ScaledSolution because of negative density");
    Scalar<DataVector> mass_density_cons(
        DataVector{{0.8, 1., -0.2, 1.3, 1.1, 1.6}});
    tnsr::I<DataVector, 2> momentum_density;
    get<0>(momentum_density) = DataVector{{-0.2, 0.1, 0.2, -0.1, 0.1, -0.2}};
    get<1>(momentum_density) = DataVector{{0.4, 0.3, 0.2, 0.3, 0.4, 0.7}};
    Scalar<DataVector> energy_density(DataVector{{1., 1.6, 1.4, 2., 1.8, 1.9}});

    const auto original_mass_density_cons = mass_density_cons;
    const auto original_momentum_density = momentum_density;
    const auto original_energy_density = energy_density;

    const auto flattener_action = NewtonianEuler::Limiters::flatten_solution(
        make_not_null(&mass_density_cons), make_not_null(&momentum_density),
        make_not_null(&energy_density), mesh, det_logical_to_inertial_jacobian,
        equation_of_state);

    // check 1) action, 2) positive density, 3) all fields changed
    CHECK(flattener_action ==
          NewtonianEuler::Limiters::FlattenerAction::ScaledSolution);
    CHECK(min(get(mass_density_cons)) > 0.);
    CHECK_FALSE(mass_density_cons == original_mass_density_cons);
    CHECK_FALSE(momentum_density == original_momentum_density);
    CHECK_FALSE(energy_density == original_energy_density);
  }

  {
    INFO("Case: SetSolutionToMean because of negative pressure");
    Scalar<DataVector> mass_density_cons(
        DataVector{{0.8, 1., 0.6, 1.3, 1.1, 1.6}});
    tnsr::I<DataVector, 2> momentum_density;
    get<0>(momentum_density) = DataVector{{-2.3, 1.9, 2.1, -3.2, 2.7, -3.2}};
    get<1>(momentum_density) = DataVector{{0.4, 0.3, 0.2, 0.3, 0.4, 0.7}};
    Scalar<DataVector> energy_density(DataVector{{1., 1.6, 1.4, 2., 1.8, 1.9}});

    // make sure initial pressure is in fact negative, because if not this isn't
    // testing the right thing
    const auto specific_internal_energy_before = Scalar<DataVector>{
        get(energy_density) / get(mass_density_cons) -
        0.5 * get(dot_product(momentum_density, momentum_density)) /
            square(get(mass_density_cons))};
    const auto pressure_before =
        equation_of_state.pressure_from_density_and_energy(
            mass_density_cons, specific_internal_energy_before);
    CHECK(min(get(pressure_before)) < 0.);

    const auto flattener_action = NewtonianEuler::Limiters::flatten_solution(
        make_not_null(&mass_density_cons), make_not_null(&momentum_density),
        make_not_null(&energy_density), mesh, det_logical_to_inertial_jacobian,
        equation_of_state);

    // check 1) action, 2) positive pressure, 3) all fields set to constant
    CHECK(flattener_action ==
          NewtonianEuler::Limiters::FlattenerAction::SetSolutionToMean);
    const auto specific_internal_energy = Scalar<DataVector>{
        get(energy_density) / get(mass_density_cons) -
        0.5 * get(dot_product(momentum_density, momentum_density)) /
            square(get(mass_density_cons))};
    const auto pressure = equation_of_state.pressure_from_density_and_energy(
        mass_density_cons, specific_internal_energy);
    CHECK(min(get(pressure)) > 0.);
    CHECK_ITERABLE_APPROX(mass_density_cons,
                          make_with_value<Scalar<DataVector>>(
                              get(pressure), get(mass_density_cons)[0]));
    CHECK_ITERABLE_APPROX(
        get<0>(momentum_density),
        DataVector(mesh.number_of_grid_points(), get<0>(momentum_density)[0]));
    CHECK_ITERABLE_APPROX(
        get<1>(momentum_density),
        DataVector(mesh.number_of_grid_points(), get<1>(momentum_density)[0]));
    CHECK_ITERABLE_APPROX(energy_density,
                          make_with_value<Scalar<DataVector>>(
                              get(pressure), get(energy_density)[0]));
  }
}

void test_flattener_3d() noexcept {
  INFO("Testing flatten_solution in 3D");
  const Mesh<3> mesh(2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const EquationsOfState::IdealFluid<false> equation_of_state{5. / 3.};
  const Scalar<DataVector> det_logical_to_inertial_jacobian(
      DataVector{{0.2, 0.1, 0.3, 0.2, 0.5, 0.6, 0.4, 0.5}});

  {
    INFO("Case: NoOp");
    Scalar<DataVector> mass_density_cons(
        DataVector{{1.4, 1.3, 1.2, 1.6, 1.8, 1.7, 1.3, 1.4}});
    tnsr::I<DataVector, 3> momentum_density;
    get<0>(momentum_density) = DataVector(8, 0.);
    get<1>(momentum_density) = DataVector(8, 0.);
    get<2>(momentum_density) =
        DataVector{{-0.3, -0.2, -0.4, -0.3, -0.8, -0.2, -0.3, -0.1}};
    Scalar<DataVector> energy_density(
        DataVector{{2.1, 2.3, 4.3, 3.5, 2.9, 1.8, 2.6, 2.9}});

    const auto expected_mass_density_cons = mass_density_cons;
    const auto expected_momentum_density = momentum_density;
    const auto expected_energy_density = energy_density;

    const auto flattener_action = NewtonianEuler::Limiters::flatten_solution(
        make_not_null(&mass_density_cons), make_not_null(&momentum_density),
        make_not_null(&energy_density), mesh, det_logical_to_inertial_jacobian,
        equation_of_state);

    CHECK(flattener_action == NewtonianEuler::Limiters::FlattenerAction::NoOp);
    CHECK_ITERABLE_APPROX(mass_density_cons, expected_mass_density_cons);
    CHECK_ITERABLE_APPROX(momentum_density, expected_momentum_density);
    CHECK_ITERABLE_APPROX(energy_density, expected_energy_density);
  }

  {
    INFO("Case: ScaledSolution because of negative density");
    Scalar<DataVector> mass_density_cons(
        DataVector{{1.4, 1.3, 1.2, 1.6, -1.6, 1.7, 1.3, 1.4}});
    tnsr::I<DataVector, 3> momentum_density;
    get<0>(momentum_density) = DataVector(8, 0.);
    get<1>(momentum_density) = DataVector(8, 0.);
    get<2>(momentum_density) =
        DataVector{{-0.3, -0.2, -0.4, -0.3, -0.8, -0.2, -0.3, -0.1}};
    Scalar<DataVector> energy_density(
        DataVector{{2.1, 2.3, 4.3, 3.5, 2.9, 1.8, 2.6, 2.9}});

    const auto original_mass_density_cons = mass_density_cons;
    const auto original_momentum_density = momentum_density;
    const auto original_energy_density = energy_density;

    const auto flattener_action = NewtonianEuler::Limiters::flatten_solution(
        make_not_null(&mass_density_cons), make_not_null(&momentum_density),
        make_not_null(&energy_density), mesh, det_logical_to_inertial_jacobian,
        equation_of_state);

    // check 1) action, 2) positive density, 3) all fields changed
    CHECK(flattener_action ==
          NewtonianEuler::Limiters::FlattenerAction::ScaledSolution);
    CHECK(min(get(mass_density_cons)) > 0.);
    CHECK_FALSE(mass_density_cons == original_mass_density_cons);
    CHECK_FALSE(momentum_density == original_momentum_density);
    CHECK_FALSE(energy_density == original_energy_density);
  }

  {
    INFO("Case: SetSolutionToMean because of negative pressure");
    Scalar<DataVector> mass_density_cons(
        DataVector{{1.4, 1.3, 1.2, 1.6, 1.8, 1.7, 1.3, 1.4}});
    tnsr::I<DataVector, 3> momentum_density;
    get<0>(momentum_density) = DataVector(8, -2.3);
    get<1>(momentum_density) = DataVector(8, 0.8);
    get<2>(momentum_density) =
        DataVector{{-0.3, -0.2, -0.4, -0.3, -0.8, -0.2, -0.3, -0.1}};
    Scalar<DataVector> energy_density(
        DataVector{{2.1, 2.3, 4.3, 3.5, 2.9, 1.8, 2.6, 2.9}});

    // make sure initial pressure is in fact negative, because if not this isn't
    // testing the right thing
    const auto specific_internal_energy_before = Scalar<DataVector>{
        get(energy_density) / get(mass_density_cons) -
        0.5 * get(dot_product(momentum_density, momentum_density)) /
            square(get(mass_density_cons))};
    const auto pressure_before =
        equation_of_state.pressure_from_density_and_energy(
            mass_density_cons, specific_internal_energy_before);
    CHECK(min(get(pressure_before)) < 0.);

    const auto flattener_action = NewtonianEuler::Limiters::flatten_solution(
        make_not_null(&mass_density_cons), make_not_null(&momentum_density),
        make_not_null(&energy_density), mesh, det_logical_to_inertial_jacobian,
        equation_of_state);

    // check 1) action, 2) positive pressure, 3) all fields set to constant
    CHECK(flattener_action ==
          NewtonianEuler::Limiters::FlattenerAction::SetSolutionToMean);
    const auto specific_internal_energy = Scalar<DataVector>{
        get(energy_density) / get(mass_density_cons) -
        0.5 * get(dot_product(momentum_density, momentum_density)) /
            square(get(mass_density_cons))};
    const auto pressure = equation_of_state.pressure_from_density_and_energy(
        mass_density_cons, specific_internal_energy);
    CHECK(min(get(pressure)) > 0.);
    CHECK_ITERABLE_APPROX(mass_density_cons,
                          make_with_value<Scalar<DataVector>>(
                              get(pressure), get(mass_density_cons)[0]));
    CHECK_ITERABLE_APPROX(
        get<0>(momentum_density),
        DataVector(mesh.number_of_grid_points(), get<0>(momentum_density)[0]));
    CHECK_ITERABLE_APPROX(
        get<1>(momentum_density),
        DataVector(mesh.number_of_grid_points(), get<1>(momentum_density)[0]));
    CHECK_ITERABLE_APPROX(energy_density,
                          make_with_value<Scalar<DataVector>>(
                              get(pressure), get(energy_density)[0]));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Limiters.Flattener",
                  "[Unit][Evolution]") {
  test_flattener_1d();
  test_flattener_2d();
  test_flattener_3d();
}
