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

template <size_t Dim>
void test_characteristic_helpers() noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-1., 1.);
  std::uniform_real_distribution<> distribution_positive(1e-3, 1.);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);
  const auto nn_distribution_positive = make_not_null(&distribution_positive);

  // This computes a unit vector. It is NOT uniformly distributed in angle,
  // but for this test the angular distribution is not important.
  const auto unit_vector = [&nn_generator, &nn_distribution]() noexcept {
    const double double_used_for_size = 0.;
    auto result = make_with_random_values<tnsr::i<double, Dim>>(
        nn_generator, nn_distribution, double_used_for_size);
    double vector_magnitude = get(magnitude(result));
    // Though highly unlikely, the vector could have length nearly 0. If this
    // happens, we edit the vector to make it non-zero.
    if (vector_magnitude < 1e-3) {
      get<0>(result) += 0.9;
      vector_magnitude = get(magnitude(result));
    }
    for (auto& n_i : result) {
      n_i /= vector_magnitude;
    }
    return result;
  }();

  const Mesh<Dim> mesh(3, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const DataVector used_for_size(pow<Dim>(3), 0.);

  // Derive all fluid quantities from the primitive variables
  const auto density = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution_positive, used_for_size);
  const auto velocity = make_with_random_values<tnsr::I<DataVector, Dim>>(
      nn_generator, nn_distribution, used_for_size);
  const auto specific_internal_energy =
      make_with_random_values<Scalar<DataVector>>(
          nn_generator, nn_distribution_positive, used_for_size);

  const EquationsOfState::IdealFluid<false> equation_of_state{5. / 3.};
  const auto pressure = equation_of_state.pressure_from_density_and_energy(
      density, specific_internal_energy);

  const Scalar<DataVector>& mass_density = density;
  const tnsr::I<DataVector, Dim> momentum_density = [&density,
                                                     &velocity]() noexcept {
    auto result = velocity;
    for (size_t i = 0; i < Dim; ++i) {
      result.get(i) *= get(density);
    }
    return result;
  }();
  const Scalar<DataVector> energy_density{
      get(density) * (get(specific_internal_energy) +
                      0.5 * get(dot_product(velocity, velocity)))};

  const Scalar<double> mean_mass_density{mean_value(get(mass_density), mesh)};
  const tnsr::I<double, Dim> mean_momentum_density = [&momentum_density,
                                                      &mesh]() noexcept {
    tnsr::I<double, Dim> result{};
    for (size_t i = 0; i < Dim; ++i) {
      result.get(i) = mean_value(momentum_density.get(i), mesh);
    }
    return result;
  }();
  const Scalar<double> mean_energy_density{
      mean_value(get(energy_density), mesh)};

  const auto right_and_left =
      NewtonianEuler::Limiters::right_and_left_eigenvectors<Dim>(
          mean_mass_density, mean_momentum_density, mean_energy_density,
          equation_of_state, unit_vector);
  const auto& right = right_and_left.first;
  const auto& left = right_and_left.second;

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

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Limiters.CharacteristicHelpers",
    "[Unit][Evolution]") {
  test_characteristic_helpers<1>();
  test_characteristic_helpers<2>();
  test_characteristic_helpers<3>();
}
