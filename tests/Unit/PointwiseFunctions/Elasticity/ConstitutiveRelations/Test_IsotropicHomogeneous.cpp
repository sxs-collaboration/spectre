// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <pup.h>
#include <random>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"
// IWYU pragma: no_forward_declare Tensor

namespace {

template <size_t Dim>
void test_type_traits() {
  INFO("Test type traits");
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim> relation{
      1., 2.};
  CHECK(relation ==
        Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>{1., 2.});
  CHECK(relation !=
        Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>{1., 1.});
  CHECK(relation !=
        Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>{2., 2.});
  test_serialization(relation);
  const auto created_relation = test_creation<
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>>(
      "  BulkModulus: 1.\n"
      "  ShearModulus: 2.\n");
  CHECK(created_relation == relation);
  Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim> moved_relation{
      1., 2.};
  test_move_semantics(std::move(moved_relation), relation);
}

template <size_t Dim>
void test_implementation(const double incompressibility,
                         const double rigidity) {
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim> relation{
      incompressibility, rigidity};
  pypp::check_with_random_values<1>(
      &Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>::stress,
      relation, "IsotropicHomogeneous", "stress", {{{-10.0, 10.0}}},
      std::tuple<double, double>{incompressibility, rigidity}, DataVector(5));
}

template <size_t Dim>
void test_implementation_suite() {
  INFO("Comparison to an independent Python implementation");
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Elasticity/ConstitutiveRelations");
  test_implementation<Dim>(1., 1.);
  // Values taken from:
  // http://homepages.engineering.auckland.ac.nz/~pkel015/SolidMechanicsBooks/Part_I/BookSM_Part_I/06_LinearElasticity/06_Linear_Elasticity_Complete.pdf
  // Iron: E=100, nu=0.29
  test_implementation<Dim>(79.3651, 38.7597);
  // Rubber: E=0.001, nu=0.4
  test_implementation<Dim>(0.00166667, 0.000357143);
  // Wood (fibre direction): E=17, nu=0.45
  test_implementation<Dim>(56.6667, 5.86207);
  // Wood (transverse direction): E=1, nu=0.79
  test_implementation<Dim>(-0.574713, 0.27933);
  // Parameters used for solving a Bowen-York momentum constraint:
  test_implementation<Dim>(0., 1.);
}

template <size_t Dim>
void test_identity(const tnsr::ii<DataVector, Dim>& random_strain,
                   const tnsr::I<DataVector, Dim>& random_inertial_coords) {
  INFO("Identity");
  // This relation should be the negative identity
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim> relation{
      1. / 3., 1. / 2.};
  const auto stress = relation.stress(random_strain, random_inertial_coords);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      CHECK_ITERABLE_APPROX(stress.get(i, j), -random_strain.get(i, j));
    }
  }
}

template <size_t Dim>
void test_trace(const tnsr::ii<DataVector, Dim>& random_strain,
                const tnsr::I<DataVector, Dim>& random_inertial_coords);

template <>
void test_trace<3>(const tnsr::ii<DataVector, 3>& random_strain,
                   const tnsr::I<DataVector, 3>& random_inertial_coords) {
  INFO("Trace");
  // This relation should result in a stress trace that is equal the negative
  // trace of the strain. The shear modulus should be irrelevant here.
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<3> relation{
      1. / 3., 10.};
  const auto stress = relation.stress(random_strain, random_inertial_coords);
  auto strain_trace = make_with_value<DataVector>(random_strain, 0.);
  auto stress_trace = make_with_value<DataVector>(random_strain, 0.);
  for (size_t i = 0; i < 3; i++) {
    strain_trace += random_strain.get(i, i);
    stress_trace += stress.get(i, i);
  }
  CHECK_ITERABLE_APPROX(stress_trace, -strain_trace);
}

template <>
void test_trace<2>(const tnsr::ii<DataVector, 2>& random_strain,
                   const tnsr::I<DataVector, 2>& random_inertial_coords) {
  INFO("Trace");
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<2> relation{
      1. / 3., 2.};
  const auto stress = relation.stress(random_strain, random_inertial_coords);
  auto strain_trace = make_with_value<DataVector>(random_strain, 0.);
  auto stress_trace = make_with_value<DataVector>(random_strain, 0.);
  for (size_t i = 0; i < 2; i++) {
    strain_trace += random_strain.get(i, i);
    stress_trace += stress.get(i, i);
  }
  // 2 (trace of delta_ij) * 9 * K * mu / (3 * K + 4 * mu) = 4 / 3
  CHECK_ITERABLE_APPROX(stress_trace, -4. / 3. * strain_trace);
}

template <size_t Dim>
void test_traceless(const tnsr::ii<DataVector, Dim>& random_strain,
                    const tnsr::I<DataVector, Dim>& random_inertial_coords) {
  INFO("Traceless");
  // This relation should result in a traceless stress.
  // The shear modulus should be irrelevant here.
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim> relation{
      0., 10.};
  const auto stress = relation.stress(random_strain, random_inertial_coords);
  auto trace = make_with_value<DataVector>(random_strain, 0.);
  for (size_t i = 0; i < Dim; i++) {
    trace += stress.get(i, i);
  }
  CHECK_ITERABLE_APPROX(trace, make_with_value<DataVector>(trace, 0.));
}

template <size_t Dim>
void test_analytically() {
  INFO("Comparison to analytic expressions");
  // Generate random strain data
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);
  const DataVector used_for_size{10};
  const auto random_strain = make_with_random_values<tnsr::ii<DataVector, Dim>>(
      nn_generator, nn_dist, used_for_size);
  auto random_strain_trace = make_with_value<DataVector>(used_for_size, 0.);
  for (size_t i = 0; i < Dim; i++) {
    random_strain_trace += random_strain.get(i, i);
  }
  const auto random_inertial_coords =
      make_with_random_values<tnsr::I<DataVector, Dim>>(nn_generator, nn_dist,
                                                        used_for_size);

  test_identity<Dim>(random_strain, random_inertial_coords);
  test_trace<Dim>(random_strain, random_inertial_coords);
  test_traceless<Dim>(random_strain, random_inertial_coords);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elasticity.ConstitutiveRelations.IsotropicHomogeneous",
                  "[PointwiseFunctions][Unit][Elasticity]") {
  {
    INFO("3D");
    test_type_traits<3>();
    test_implementation_suite<3>();
    test_analytically<3>();
  }
  {
    INFO("2D");
    test_type_traits<2>();
    test_implementation_suite<2>();
    test_analytically<2>();
  }
}
