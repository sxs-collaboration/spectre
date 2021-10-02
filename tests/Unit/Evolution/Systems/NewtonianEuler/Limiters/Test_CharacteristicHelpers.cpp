// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/CharacteristicHelpers.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

// This computes a unit vector. It is NOT uniformly distributed in angle,
// but for this test the angular distribution is not important.
template <size_t Dim>
tnsr::i<double, Dim> random_unit_vector(
    const gsl::not_null<std::mt19937*>& gen,
    const gsl::not_null<std::uniform_real_distribution<>*>& dist) {
  const double used_for_size = 0.;
  auto result =
      make_with_random_values<tnsr::i<double, Dim>>(gen, dist, used_for_size);
  double vector_magnitude = get(magnitude(result));
  // Though highly unlikely, the vector could have length nearly 0. If this
  // happens, we edit the vector to make it non-zero.
  if (vector_magnitude < 1e-3) {
    get<0>(result) += 0.9;
    vector_magnitude = get(magnitude(result));
  }
  // Make unit magnitude
  for (auto& n_i : result) {
    n_i /= vector_magnitude;
  }
  return result;
}

template <size_t Dim>
void prepare_hydro_data_for_test(
    const gsl::not_null<Scalar<DataVector>*>& mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>& momentum_density,
    const gsl::not_null<Scalar<DataVector>*>& energy_density,
    const gsl::not_null<Scalar<double>*>& mean_mass_density,
    const gsl::not_null<tnsr::I<double, Dim>*>& mean_momentum_density,
    const gsl::not_null<Scalar<double>*>& mean_energy_density,
    const gsl::not_null<Matrix*>& right, const gsl::not_null<Matrix*>& left,
    const Mesh<Dim>& mesh,
    const EquationsOfState::IdealFluid<false>& equation_of_state,
    const bool generate_random_vector = true) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-1., 1.);
  std::uniform_real_distribution<> distribution_positive(1e-3, 1.);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);
  const auto nn_distribution_positive = make_not_null(&distribution_positive);

  const auto unit_vector = [&generate_random_vector, &nn_generator,
                            &nn_distribution]() {
    if (generate_random_vector) {
      return random_unit_vector<Dim>(nn_generator, nn_distribution);
    } else {
      // Return unit vector pointing in the "last" dimension
      tnsr::i<double, Dim> result{{0.}};
      get<Dim - 1>(result) = 1.;
      return result;
    }
  }();

  // Derive all fluid quantities from the primitive variables
  const DataVector used_for_size(mesh.number_of_grid_points(), 0.);
  const auto density = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution_positive, used_for_size);
  const auto velocity = make_with_random_values<tnsr::I<DataVector, Dim>>(
      nn_generator, nn_distribution, used_for_size);
  const auto specific_internal_energy =
      make_with_random_values<Scalar<DataVector>>(
          nn_generator, nn_distribution_positive, used_for_size);

  const auto pressure = equation_of_state.pressure_from_density_and_energy(
      density, specific_internal_energy);

  *mass_density = density;
  *momentum_density = velocity;
  for (size_t i = 0; i < Dim; ++i) {
    momentum_density->get(i) *= get(density);
  }
  *energy_density = Scalar<DataVector>{
      get(density) * (get(specific_internal_energy) +
                      0.5 * get(dot_product(velocity, velocity)))};

  get(*mean_mass_density) = mean_value(get(*mass_density), mesh);
  for (size_t i = 0; i < Dim; ++i) {
    mean_momentum_density->get(i) = mean_value(momentum_density->get(i), mesh);
  }
  get(*mean_energy_density) = mean_value(get(*energy_density), mesh);

  const auto right_and_left =
      NewtonianEuler::Limiters::right_and_left_eigenvectors<Dim>(
          *mean_mass_density, *mean_momentum_density, *mean_energy_density,
          equation_of_state, unit_vector);
  *right = right_and_left.first;
  *left = right_and_left.second;
}

template <size_t Dim>
void test_characteristic_helpers() {
  INFO("Testing characteristic helpers");
  CAPTURE(Dim);
  const Mesh<Dim> mesh(3, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const EquationsOfState::IdealFluid<false> equation_of_state{5. / 3.};

  Scalar<DataVector> mass_density;
  tnsr::I<DataVector, Dim> momentum_density;
  Scalar<DataVector> energy_density;
  Scalar<double> mean_mass_density;
  tnsr::I<double, Dim> mean_momentum_density;
  Scalar<double> mean_energy_density;
  Matrix right;
  Matrix left;

  prepare_hydro_data_for_test(
      make_not_null(&mass_density), make_not_null(&momentum_density),
      make_not_null(&energy_density), make_not_null(&mean_mass_density),
      make_not_null(&mean_momentum_density),
      make_not_null(&mean_energy_density), make_not_null(&right),
      make_not_null(&left), mesh, equation_of_state);

  // First test that the tensor-transform helpers are inverses of each
  // other. This is a sanity check of the two tensor-transform helpers and
  // of right_and_left_eigenvectors.
  Scalar<DataVector> v_minus{};
  tnsr::I<DataVector, Dim> v_momentum{};
  Scalar<DataVector> v_plus{};
  NewtonianEuler::Limiters::characteristic_fields(
      make_not_null(&v_minus), make_not_null(&v_momentum),
      make_not_null(&v_plus), mass_density, momentum_density, energy_density,
      left);
  Scalar<DataVector> recovered_mass_density{};
  tnsr::I<DataVector, Dim> recovered_momentum_density{};
  Scalar<DataVector> recovered_energy_density{};
  NewtonianEuler::Limiters::conserved_fields_from_characteristic_fields(
      make_not_null(&recovered_mass_density),
      make_not_null(&recovered_momentum_density),
      make_not_null(&recovered_energy_density), v_minus, v_momentum, v_plus,
      right);
  CHECK_ITERABLE_APPROX(mass_density, recovered_mass_density);
  CHECK_ITERABLE_APPROX(momentum_density, recovered_momentum_density);
  CHECK_ITERABLE_APPROX(energy_density, recovered_energy_density);

  // Test that the cell-average helper matches the tensor helper
  tuples::TaggedTuple<Tags::Mean<NewtonianEuler::Tags::VMinus>,
                      Tags::Mean<NewtonianEuler::Tags::VMomentum<Dim>>,
                      Tags::Mean<NewtonianEuler::Tags::VPlus>>
      mean_char_vars;
  NewtonianEuler::Limiters::characteristic_fields(
      make_not_null(&mean_char_vars),
      {mean_mass_density, mean_momentum_density, mean_energy_density}, left);
  CHECK(get(get<Tags::Mean<NewtonianEuler::Tags::VMinus>>(mean_char_vars)) ==
        approx(mean_value(get(v_minus), mesh)));
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(get<Tags::Mean<NewtonianEuler::Tags::VMomentum<Dim>>>(mean_char_vars)
              .get(i) == approx(mean_value(v_momentum.get(i), mesh)));
  }
  CHECK(get(get<Tags::Mean<NewtonianEuler::Tags::VPlus>>(mean_char_vars)) ==
        approx(mean_value(get(v_plus), mesh)));

  // Test that the Variables helper matches the tensor helper
  Variables<tmpl::list<NewtonianEuler::Tags::VMinus,
                       NewtonianEuler::Tags::VMomentum<Dim>,
                       NewtonianEuler::Tags::VPlus>>
      char_vars(mesh.number_of_grid_points());
  Variables<tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                       NewtonianEuler::Tags::MomentumDensity<Dim>,
                       NewtonianEuler::Tags::EnergyDensity>>
      cons_vars(mesh.number_of_grid_points());
  get<NewtonianEuler::Tags::MassDensityCons>(cons_vars) = mass_density;
  get<NewtonianEuler::Tags::MomentumDensity<Dim>>(cons_vars) = momentum_density;
  get<NewtonianEuler::Tags::EnergyDensity>(cons_vars) = energy_density;
  NewtonianEuler::Limiters::characteristic_fields(make_not_null(&char_vars),
                                                  cons_vars, left);
  CHECK_ITERABLE_APPROX(get<NewtonianEuler::Tags::VMinus>(char_vars), v_minus);
  CHECK_ITERABLE_APPROX(get<NewtonianEuler::Tags::VMomentum<Dim>>(char_vars),
                        v_momentum);
  CHECK_ITERABLE_APPROX(get<NewtonianEuler::Tags::VPlus>(char_vars), v_plus);
}

template <size_t Dim>
void test_apply_limiter_to_char_fields() {
  INFO("Testing apply_limiter_to_characteristic_fields_in_all_directions");
  CAPTURE(Dim);
  const Mesh<Dim> mesh(3, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const EquationsOfState::IdealFluid<false> equation_of_state{5. / 3.};

  Scalar<DataVector> mass_density;
  tnsr::I<DataVector, Dim> momentum_density;
  Scalar<DataVector> energy_density;
  Scalar<double> mean_mass_density;
  tnsr::I<double, Dim> mean_momentum_density;
  Scalar<double> mean_energy_density;
  Matrix right;
  Matrix left;

  const bool generate_random_vector = false;
  prepare_hydro_data_for_test(
      make_not_null(&mass_density), make_not_null(&momentum_density),
      make_not_null(&energy_density), make_not_null(&mean_mass_density),
      make_not_null(&mean_momentum_density),
      make_not_null(&mean_energy_density), make_not_null(&right),
      make_not_null(&left), mesh, equation_of_state, generate_random_vector);

  // Check case where limiter does nothing
  const auto noop_expected_mass_density = mass_density;
  const auto noop_expected_momentum_density = momentum_density;
  const auto noop_expected_energy_density = energy_density;
  const auto noop_limiter_wrapper =
      [](const gsl::not_null<Scalar<DataVector>*> /*char_v_minus*/,
         const gsl::not_null<tnsr::I<DataVector, Dim>*> /*char_v_momentum*/,
         const gsl::not_null<Scalar<DataVector>*> /*char_v_plus*/,
         const Matrix& /*left*/) -> bool { return false; };

  const bool noop_result = NewtonianEuler::Limiters::
      apply_limiter_to_characteristic_fields_in_all_directions(
          make_not_null(&mass_density), make_not_null(&momentum_density),
          make_not_null(&energy_density), mesh, equation_of_state,
          noop_limiter_wrapper);
  CHECK_FALSE(noop_result);
  CHECK_ITERABLE_APPROX(mass_density, noop_expected_mass_density);
  CHECK_ITERABLE_APPROX(momentum_density, noop_expected_momentum_density);
  CHECK_ITERABLE_APPROX(energy_density, noop_expected_energy_density);

  // Check with a silly (and completely nonphysical) limiter that scales char
  // fields, which is relatively easy to check. The solution is set to 0 on the
  // first dims (d < Dim-1), and scaled by some integers on dim d == Dim-1
  size_t current_dim = 0;
  const auto silly_limiter_wrapper =
      [&current_dim](
          const gsl::not_null<Scalar<DataVector>*> char_v_minus,
          const gsl::not_null<tnsr::I<DataVector, Dim>*> char_v_momentum,
          const gsl::not_null<Scalar<DataVector>*> char_v_plus,
          const Matrix& /*left*/) -> bool {
    if (current_dim == Dim - 1) {
      get(*char_v_minus) *= 2.;
      // no change to char_v_momentum
      get(*char_v_plus) = 0.;
    } else {
      get(*char_v_minus) = 0.;
      for (size_t i = 0; i < Dim; ++i) {
        char_v_momentum->get(i) = 0.;
      }
      get(*char_v_plus) = 0.;
      current_dim++;
    }
    return true;
  };

  Matrix silly_limiter_action(Dim + 2, Dim + 2, 0.);
  silly_limiter_action(0, 0) = 2. / static_cast<double>(Dim);
  for (size_t i = 0; i < Dim; ++i) {
    silly_limiter_action(i + 1, i + 1) = 1. / static_cast<double>(Dim);
  }
  silly_limiter_action(Dim + 1, Dim + 1) = 0.;
  const Matrix silly = right * silly_limiter_action * left;

  auto expected_mass_density = mass_density;
  auto expected_momentum_density = momentum_density;
  auto expected_energy_density = energy_density;

  get(expected_mass_density) = silly(0, 0) * get(mass_density);
  for (size_t j = 0; j < Dim; ++j) {
    expected_momentum_density.get(j) = silly(j + 1, 0) * get(mass_density);
  }
  get(expected_energy_density) = silly(Dim + 1, 0) * get(mass_density);
  for (size_t i = 0; i < Dim; ++i) {
    get(expected_mass_density) += silly(0, i + 1) * momentum_density.get(i);
    for (size_t j = 0; j < Dim; ++j) {
      expected_momentum_density.get(j) +=
          silly(j + 1, i + 1) * momentum_density.get(i);
    }
    get(expected_energy_density) +=
        silly(Dim + 1, i + 1) * momentum_density.get(i);
  }
  get(expected_mass_density) += silly(0, Dim + 1) * get(energy_density);
  for (size_t j = 0; j < Dim; ++j) {
    expected_momentum_density.get(j) +=
        silly(j + 1, Dim + 1) * get(energy_density);
  }
  get(expected_energy_density) += silly(Dim + 1, Dim + 1) * get(energy_density);

  const bool silly_result = NewtonianEuler::Limiters::
      apply_limiter_to_characteristic_fields_in_all_directions(
          make_not_null(&mass_density), make_not_null(&momentum_density),
          make_not_null(&energy_density), mesh, equation_of_state,
          silly_limiter_wrapper);
  CHECK(silly_result);
  CHECK_ITERABLE_APPROX(mass_density, expected_mass_density);
  CHECK_ITERABLE_APPROX(momentum_density, expected_momentum_density);
  CHECK_ITERABLE_APPROX(energy_density, expected_energy_density);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Limiters.CharacteristicHelpers",
    "[Unit][Evolution]") {
  test_characteristic_helpers<1>();
  test_characteristic_helpers<2>();
  test_characteristic_helpers<3>();

  test_apply_limiter_to_char_fields<1>();
  test_apply_limiter_to_char_fields<2>();
  test_apply_limiter_to_char_fields<3>();
}
