// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <pup.h>
#include <random>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/CubicCrystal.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
// IWYU pragma: no_forward_declare Tensor

namespace {

void test_semantics() {
  INFO("Test type traits");
  const Elasticity::ConstitutiveRelations::CubicCrystal relation{3., 2., 1.};
  CHECK(relation ==
        Elasticity::ConstitutiveRelations::CubicCrystal{3., 2., 1.});
  CHECK(relation !=
        Elasticity::ConstitutiveRelations::CubicCrystal{2., 2., 2.});
  CHECK(relation !=
        Elasticity::ConstitutiveRelations::CubicCrystal{3., 0.2, 1.});
  test_serialization(relation);
  test_copy_semantics(relation);
  const auto created_relation = TestHelpers::test_creation<
      Elasticity::ConstitutiveRelations::CubicCrystal>(
      "C_11: 3.\n"
      "C_12: 2.\n"
      "C_44: 1.\n");
  CHECK(created_relation == relation);
  Elasticity::ConstitutiveRelations::CubicCrystal moved_relation{3., 2., 1.};
  test_move_semantics(std::move(moved_relation), relation);
}

void test_consistency(const tnsr::ii<DataVector, 3>& random_strain,
                      const tnsr::I<DataVector, 3>& random_inertial_coords) {
  INFO("Consistency between CubicCrystal and IsotropicHomogeneous");
  const double youngs_modulus = 1.;
  const double poisson_ratio = 1. / 3.;
  const double isotropic_bulk_modulus =
      youngs_modulus / (1. - 2. * poisson_ratio) / 3.;
  const double isotropic_shear_modulus =
      youngs_modulus / (1. + poisson_ratio) / 2.;
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<3>
      isotropic_homogeneous_relation{isotropic_bulk_modulus,
                                     isotropic_shear_modulus};
  const double c_11 = youngs_modulus * (1.- poisson_ratio) /
                      ((1. + poisson_ratio) * (1. - 2. * poisson_ratio));
  const double c_12 = youngs_modulus * poisson_ratio /
                      ((1. + poisson_ratio) * (1. - 2. * poisson_ratio));
  const double c_44 = isotropic_shear_modulus;
  // this should be isotropic homogeneous
  const Elasticity::ConstitutiveRelations::CubicCrystal
      cubic_crystalline_relation{c_11, c_12, c_44};
  tnsr::II<DataVector, 3> cubic_crystalline_stress{
      random_strain.begin()->size()};
  cubic_crystalline_relation.stress(make_not_null(&cubic_crystalline_stress),
                                    random_strain, random_inertial_coords);
  tnsr::II<DataVector, 3> isotropic_homogeneous_stress{
      random_strain.begin()->size()};
  isotropic_homogeneous_relation.stress(
      make_not_null(&isotropic_homogeneous_stress), random_strain,
      random_inertial_coords);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      CHECK_ITERABLE_APPROX(cubic_crystalline_stress.get(i, j),
                            isotropic_homogeneous_stress.get(i, j));
    }
  }
  // This relation should be the negative identity
  const Elasticity::ConstitutiveRelations::CubicCrystal relation{1., 0., 0.5};
  relation.stress(make_not_null(&cubic_crystalline_stress), random_strain,
                  random_inertial_coords);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      CHECK_ITERABLE_APPROX(cubic_crystalline_stress.get(i, j),
                            -random_strain.get(i, j));
    }
  }
}

void test_implementation(const double youngs_modulus,
                         const double poisson_ratio, double shear_modulus) {
  const Elasticity::ConstitutiveRelations::CubicCrystal relation{
      youngs_modulus, poisson_ratio, shear_modulus};
  pypp::check_with_random_values<1>(
      static_cast<void (Elasticity::ConstitutiveRelations::CubicCrystal::*)(
          gsl::not_null<tnsr::II<DataVector, 3>*>,
          const tnsr::ii<DataVector, 3>&, const tnsr::I<DataVector, 3>&)
                      const noexcept>(
          &Elasticity::ConstitutiveRelations::CubicCrystal::stress),
      relation, "CubicCrystal", {"stress"}, {{{-1., 1.}}},
      std::tuple<double, double, double>{youngs_modulus, poisson_ratio,
                                         shear_modulus},
      DataVector(5));
}

void test_implementation_suite() {
  INFO("Comparison to an independent Python implementation");
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Elasticity/ConstitutiveRelations");
  test_implementation(3., 2., 1.);
  // Values taken from:
  // R. E. Newnham: Properties of materials. Oxford University Press, 2005, ISBN
  // 978-0-19-852075-7 Diamond: c_11=1020, c_12=250, c_44=492
  test_implementation(1020., 250., 492.);
  // Silicon: c_11=166, c_12=64, c_44=80
  test_implementation(166., 64., 80.);
  // Germanium: c_11=130, c_12=49, c_44=67
  test_implementation(130., 49., 67.);
  // Lithium: c_11=13.5, c_12=11.4, c_44=8.8
  test_implementation(13.5, 11.4, 8.8);
  // Sodium: c_11=7.4, c_12=6.2, c_44=4.2
  test_implementation(7.4, 6.2, 4.2);
  // Potassium: c_11=3.7, c_12=3.1, c_44=1.9
  test_implementation(3.7, 3.1, 1.9);
  // NaCl: c_11=48.5, c_12=12.5, c_44=12.7
  test_implementation(48.5, 12.5, 12.7);
  // KCl: c_11=40.5, c_12=6.6, c_44=6.3
  test_implementation(40.5, 6.6, 6.3);
  // RbCl: c_11=36.3, c_12=6.2, c_44=4.7
  test_implementation(36.3, 6.2, 4.7);
}

void test_analytically() {
  INFO("Comparison to analytic expressions");
  // Generate random strain data
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);
  const DataVector used_for_size{10};
  const auto random_strain = make_with_random_values<tnsr::ii<DataVector, 3>>(
      nn_generator, nn_dist, used_for_size);
  const auto random_inertial_coords =
      make_with_random_values<tnsr::I<DataVector, 3>>(nn_generator, nn_dist,
                                                      used_for_size);

  test_consistency(random_strain, random_inertial_coords);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elasticity.ConstitutiveRelations.CubicCrystal",
                  "[PointwiseFunctions][Unit][Elasticity]") {
  {
    INFO("CubicyCrystal");
    test_semantics();
    test_analytically();
    test_implementation_suite();
  }
}
