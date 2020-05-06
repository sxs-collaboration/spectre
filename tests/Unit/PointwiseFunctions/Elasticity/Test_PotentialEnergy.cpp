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
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/BentBeam.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "PointwiseFunctions/Elasticity/PotentialEnergy.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

// This wrapper function is needed to construct a constitutive relation for the
// random values of its parameters.
template <size_t Dim>
Scalar<DataVector> potential_energy(
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

void test_energy_with_bent_beam() {
  using BentBeam = Elasticity::Solutions::BentBeam;
  const size_t dim = BentBeam::volume_dim;
  const double length = 5.;
  const double height = 1.;
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<dim>
      constitutive_relation{// Iron: E=100, nu=0.29
                            79.36507936507935, 38.75968992248062};
  const BentBeam& solution{length, height, 0.5, constitutive_relation};
  using AffineMap = ::domain::CoordinateMaps::Affine;
  using AffineMap2D =
      ::domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
  const ::domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap2D>
      coord_map{{{-1., 1., -length / 2., length / 2.},
                 {-1., 1., -height / 2., height / 2.}}};
  const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  // Since the solution is a polynomial of degree 2 it should be fully
  // characterized to machine precision on 3 grid points per dimension.
  const auto logical_coords = logical_coordinates(mesh);
  const auto inertial_coords = coord_map(logical_coords);
  auto jacobian = coord_map.jacobian(logical_coords);
  typename ::Elasticity::Tags::Strain<dim>::type strain =
      get<::Elasticity::Tags::Strain<dim>>(solution.variables(
          inertial_coords, tmpl::list<::Elasticity::Tags::Strain<dim>>{}));
  auto pointwise_energy = ::Elasticity::evaluate_potential_energy<dim>(
      strain, inertial_coords, constitutive_relation);
  const DataVector det_jacobian = get(determinant(jacobian));
  double potential_energy =
      definite_integral(det_jacobian * pointwise_energy[0], mesh);
  CHECK_ITERABLE_APPROX(potential_energy, solution.potential_energy());
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Elasticity",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Elasticity"};
  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_potential_energy, (2, 3));
  CHECK_FOR_DATAVECTORS(test_compute_tags, (2, 3));
  INFO("test_potential_energy_of_bent_beam");
  test_energy_with_bent_beam();
}
