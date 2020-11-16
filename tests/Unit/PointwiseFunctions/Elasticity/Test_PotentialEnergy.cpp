// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "PointwiseFunctions/Elasticity/PotentialEnergy.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <size_t Dim>
Scalar<DataVector> potential_energy_density(
    // The wrapper constructs a constitutive relation for use in
    // check_with_random_values.
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& strain,
    const tnsr::I<DataVector, Dim>& coordinates, const double bulk_modulus,
    const double shear_modulus) noexcept {
  Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>
      constitutive_relation{bulk_modulus, shear_modulus};
  return Elasticity::potential_energy_density<Dim>(
      strain, coordinates, std::move(constitutive_relation));
}

template <size_t Dim>
void test_elastic_potential_energy(const DataVector& used_for_size) {
  pypp::check_with_random_values<4>(
      &potential_energy_density<Dim>, "Elasticity.PotentialEnergy",
      "potential_energy_density", {{{-1., 1.}, {0., 1.}, {0., 1.}, {0., 1.}}},
      used_for_size);
}

template <size_t Dim>
void test_compute_tags(const DataVector& used_for_size) noexcept {
  CAPTURE(Dim);
  using strain_tag = Elasticity::Tags::Strain<Dim>;
  using energy_tag = Elasticity::Tags::PotentialEnergyDensity<Dim>;
  using energy_compute_tag =
      Elasticity::Tags::PotentialEnergyDensityCompute<Dim>;
  using constitutive_relation_tag = Elasticity::Tags::ConstitutiveRelation<
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>>;
  using coordinates_tag = domain::Tags::Coordinates<Dim, Frame::Inertial>;
  TestHelpers::db::test_compute_tag<energy_compute_tag>(
      "PotentialEnergyDensity");
  const size_t num_points = used_for_size.size();
  {
    INFO("Energy");
    const auto box =
        db::create<db::AddSimpleTags<strain_tag, constitutive_relation_tag,
                                     coordinates_tag>,
                   db::AddComputeTags<energy_compute_tag>>(
            tnsr::ii<DataVector, Dim>{num_points, 1.},
            Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>{3.,
                                                                         4.},
            tnsr::I<DataVector, Dim>{num_points, 2.});

    const auto expected_energy = Elasticity::potential_energy_density<Dim>(
        get<strain_tag>(box), get<coordinates_tag>(box),
        get<constitutive_relation_tag>(box));
    CHECK(get<energy_tag>(box) == expected_energy);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Elasticity",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"PointwiseFunctions"};
  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_elastic_potential_energy, (2, 3));
  CHECK_FOR_DATAVECTORS(test_compute_tags, (2, 3));
}
