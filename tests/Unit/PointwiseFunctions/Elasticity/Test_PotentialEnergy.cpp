// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "PointwiseFunctions/Elasticity/PotentialEnergy.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <size_t Dim>
Scalar<DataVector> potential_energy(  // This wrapper function is needed to
                                      // construct a constitutive relation for
                                      // the random values of its parameters.
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& strain,
    const tnsr::I<DataVector, Dim>& coordinates, const double bulk_modulus,
    const double shear_modulus) noexcept {
  Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>
      constitutive_relation{bulk_modulus, shear_modulus};
  return Elasticity::evaluate_potential_energy<Dim>(
      strain, coordinates, std::move(constitutive_relation));
}

template <size_t Dim>
void test_potential_energy(const DataVector& used_for_size) {
  pypp::check_with_random_values<4>(
      &potential_energy<Dim>, "PotentialEnergy", "potential_energy",
      {{{-1., 1.}, {0., 1.}, {0., 1.}, {0., 1.}}}, used_for_size);
}

template <size_t Dim>
void test_compute_tags(const DataVector& used_for_size) noexcept {
  using strain_tag = Elasticity::Tags::Strain<Dim>;
  using energy_tag = Elasticity::Tags::PotentialEnergy<Dim>;
  using energy_compute_tag = Elasticity::Tags::PotentialEnergyCompute<Dim>;
  using constitutive_relation_tag = Elasticity::Tags::ConstitutiveRelation<
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>>;
  using coordinates_tag = domain::Tags::Coordinates<Dim, Frame::Inertial>;
  const size_t num_points = used_for_size.size();
  {
    INFO("Energy" << Dim << "D");
    auto box =
        db::create<db::AddSimpleTags<strain_tag, constitutive_relation_tag,
                                     coordinates_tag>,
                   db::AddComputeTags<energy_compute_tag>>(
            tnsr::ii<DataVector, Dim>{num_points, 1.},
            Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>{3.,
                                                                         4.},
            tnsr::I<DataVector, Dim>{num_points, 2.});

    auto expected_energy = Elasticity::evaluate_potential_energy<Dim>(
        get<strain_tag>(box), get<coordinates_tag>(box),
        get<constitutive_relation_tag>(box));
    CHECK(get<energy_tag>(box) == expected_energy);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Elasticity",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Elasticity"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_potential_energy, (2, 3));

  CHECK_FOR_DATAVECTORS(test_compute_tags, (2, 3));
}
