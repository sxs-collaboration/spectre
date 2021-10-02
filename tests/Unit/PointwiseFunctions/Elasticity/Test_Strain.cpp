// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Elasticity/Strain.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

// Polynomial functions of max. degree 2, so the differentiation is exact on 3
// grid points
template <size_t Dim>
tnsr::I<DataVector, Dim> polynomial_displacement(
    const tnsr::I<DataVector, Dim>& x) {
  tnsr::I<DataVector, Dim> displacement{x.begin()->size()};
  get<0>(displacement) = square(get<0>(x)) + 2. * get<1>(x);
  get<1>(displacement) = square(get<1>(x)) + 3. * get<0>(x);
  if constexpr (Dim == 3) {
    get<2>(displacement) = square(get<2>(x)) + get<0>(x) + 4. * get<1>(x);
  }
  return displacement;
}

// This is the symmetrized gradient of `polynomial_displacement`
template <size_t Dim>
tnsr::ii<DataVector, Dim> polynomial_strain(const tnsr::I<DataVector, Dim>& x) {
  tnsr::ii<DataVector, Dim> strain{x.begin()->size()};
  get<0, 0>(strain) = 2. * get<0>(x);
  get<1, 1>(strain) = 2. * get<1>(x);
  get<0, 1>(strain) = 2.5;
  if constexpr (Dim == 3) {
    get<2, 2>(strain) = 2. * get<2>(x);
    get<0, 2>(strain) = 0.5;
    get<1, 2>(strain) = 2.;
  }
  return strain;
}

template <size_t Dim>
auto make_coord_map() {
  using AffineMap = domain::CoordinateMaps::Affine;
  if constexpr (Dim == 1) {
    return domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                 AffineMap>{{-1., 1., 0., M_PI}};
  } else if constexpr (Dim == 2) {
    using AffineMap2D =
        domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
    return domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                 AffineMap2D>{
        {{-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}}};
  } else {
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    return domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                 AffineMap3D>{
        {{-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}}};
  }
}

template <size_t Dim>
void test_strain() {
  CAPTURE(Dim);
  {
    INFO("Random-value tests");
    DataVector used_for_size{5};
    pypp::check_with_random_values<1>(
        static_cast<void (*)(gsl::not_null<tnsr::ii<DataVector, Dim>*>,
                             const tnsr::iJ<DataVector, Dim>&)>(
            &Elasticity::strain<DataVector, Dim>),
        "Strain", {"strain_flat"}, {{{-1., 1.}}}, used_for_size);
    pypp::check_with_random_values<1>(
        static_cast<void (*)(gsl::not_null<tnsr::ii<DataVector, Dim>*>,
                             const tnsr::iJ<DataVector, Dim>&,
                             const tnsr::ii<DataVector, Dim>&,
                             const tnsr::ijj<DataVector, Dim>&,
                             const tnsr::ijj<DataVector, Dim>&,
                             const tnsr::I<DataVector, Dim>&)>(
            &Elasticity::strain<DataVector, Dim>),
        "Strain", {"strain_curved"}, {{{-1., 1.}}}, used_for_size);
  }
  const Mesh<Dim> mesh{3, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);
  const auto coord_map = make_coord_map<Dim>();
  const auto inertial_coords = coord_map(logical_coords);
  const auto inv_jacobian = coord_map.inv_jacobian(logical_coords);
  const auto displacement = polynomial_displacement(inertial_coords);
  const auto expected_strain = polynomial_strain(inertial_coords);
  tnsr::ii<DataVector, Dim> strain{mesh.number_of_grid_points()};
  Elasticity::strain(make_not_null(&strain), displacement, mesh, inv_jacobian);
  for (size_t i = 0; i < strain.size(); ++i) {
    const auto component_name =
        strain.component_name(strain.get_tensor_index(i));
    CAPTURE(component_name);
    CHECK_ITERABLE_APPROX(strain[i], expected_strain[i]);
  }
  {
    INFO("Test the compute tag");
    TestHelpers::db::test_compute_tag<Elasticity::Tags::StrainCompute<Dim>>(
        "Strain");
    const auto box = db::create<
        db::AddSimpleTags<Elasticity::Tags::Displacement<Dim>,
                          domain::Tags::Mesh<Dim>,
                          domain::Tags::InverseJacobian<
                              Dim, Frame::ElementLogical, Frame::Inertial>>,
        db::AddComputeTags<Elasticity::Tags::StrainCompute<Dim>>>(
        displacement, mesh, inv_jacobian);
    CHECK(get<Elasticity::Tags::Strain<Dim>>(box) == strain);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Elasticity.Strain",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Elasticity"};
  test_strain<2>();
  test_strain<3>();
}
