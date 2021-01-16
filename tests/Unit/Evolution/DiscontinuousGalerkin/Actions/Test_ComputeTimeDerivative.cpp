// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <charm++.h>
#include <optional>
#include <pup.h>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeImpl.hpp"
#include "Parallel/CharmPupable.hpp"

namespace TestHelpers::evolution::dg::Actions {
namespace {
namespace WithInverseSpatialMetricTag {
template <size_t Dim>
struct InverseSpatialMetric : db::SimpleTag {
  using type = tnsr::II<DataVector, Dim, Frame::Inertial>;
};

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  using flux_variables = tmpl::list<Var1>;
  using inverse_spatial_metric_tag = InverseSpatialMetric<Dim>;
};

template <size_t Dim>
struct BoundaryTerms {
  using dg_package_field_tags = tmpl::list<Var1>;

  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<Scalar<DataVector>*> out_var1,

      const Scalar<DataVector>& var1,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_var1,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity)
      const noexcept {
    *out_normal_dot_flux_var1 = dot_product(normal_covector, flux_var1);
    *out_var1 = var1;

    const DataVector normal_magnitude =
        sqrt(get(dot_product(normal_covector, normal_vector)));
    CHECK_ITERABLE_APPROX(DataVector(normal_magnitude.size(), 1.0),
                          normal_magnitude);

    if (normal_dot_mesh_velocity.has_value()) {
      return max(1.0 - get(*normal_dot_mesh_velocity));
    }
    return 1.0;
  }
};

template <size_t Dim>
void test(const bool use_moving_mesh) {
  constexpr size_t number_of_grid_points = 5;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> dist{0.0, 1.0};

  Variables<tmpl::list<::Tags::NormalDotFlux<Var1>, Var1>> packaged_data{
      number_of_grid_points};
  BoundaryTerms<Dim> boundary_correction{};
  Variables<
      tmpl::list<Var1, ::Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>,
                 InverseSpatialMetric<Dim>,
                 ::evolution::dg::Actions::detail::NormalVector<Dim>>>
      projected_fields{number_of_grid_points};
  fill_with_random_values(make_not_null(&projected_fields), make_not_null(&gen),
                          make_not_null(&dist));

  tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal_covector{
      number_of_grid_points};
  fill_with_random_values(make_not_null(&unit_normal_covector),
                          make_not_null(&gen), make_not_null(&dist));
  auto& normal_vector =
      get<::evolution::dg::Actions::detail::NormalVector<Dim>>(
          projected_fields);
  for (size_t i = 0; i < Dim; ++i) {
    normal_vector.get(i) = 0.0;
    for (size_t j = 0; j < Dim; ++j) {
      normal_vector.get(i) +=
          unit_normal_covector.get(j) *
          get<InverseSpatialMetric<Dim>>(projected_fields).get(i, j);
    }
  }
  const DataVector magnitude_of_normal =
      sqrt(get(dot_product(normal_vector, unit_normal_covector)));
  for (size_t i = 0; i < Dim; ++i) {
    normal_vector.get(i) /= magnitude_of_normal;
    unit_normal_covector.get(i) /= magnitude_of_normal;
  }

  std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>> mesh_velocity;
  std::optional<Scalar<DataVector>> normal_dot_mesh_velocity;
  if (use_moving_mesh) {
    mesh_velocity =
        make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
            make_not_null(&gen), make_not_null(&dist), magnitude_of_normal);
    normal_dot_mesh_velocity =
        dot_product(unit_normal_covector, *mesh_velocity);
  }

  db::DataBox<tmpl::list<>> box{};
  const double max_speed =
      ::evolution::dg::Actions::detail::dg_package_data<System<Dim>>(
          make_not_null(&packaged_data), boundary_correction, projected_fields,
          unit_normal_covector, mesh_velocity, box, tmpl::list<>{},
          tmpl::list<Var1,
                     ::Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>>{});

  Variables<tmpl::list<::Tags::NormalDotFlux<Var1>, Var1>>
      expected_packaged_data{number_of_grid_points};
  const double expected_max_speed = boundary_correction.dg_package_data(
      make_not_null(&get<::Tags::NormalDotFlux<Var1>>(expected_packaged_data)),
      make_not_null(&get<Var1>(expected_packaged_data)),
      get<Var1>(projected_fields),
      get<::Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>>(
          projected_fields),
      unit_normal_covector,
      get<::evolution::dg::Actions::detail::NormalVector<Dim>>(
          projected_fields),
      mesh_velocity, normal_dot_mesh_velocity);

  CHECK(max_speed == approx(expected_max_speed));
  CHECK_ITERABLE_APPROX(
      get<::Tags::NormalDotFlux<Var1>>(packaged_data),
      get<::Tags::NormalDotFlux<Var1>>(expected_packaged_data));
  CHECK_ITERABLE_APPROX(get<Var1>(packaged_data),
                        get<Var1>(expected_packaged_data));
}
}  // namespace WithInverseSpatialMetricTag

template <SystemType system_type, UseBoundaryCorrection use_boundary_correction>
void test_wrapper() {
  test<system_type, use_boundary_correction, 1>();
  test<system_type, use_boundary_correction, 2>();
  test<system_type, use_boundary_correction, 3>();
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.ComputeTimeDerivative",
                  "[Unit][Evolution][Actions]") {
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                       domain::CoordinateMaps::Identity<1>>));
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                       domain::CoordinateMaps::Identity<2>>));
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                       domain::CoordinateMaps::Identity<3>>));

  // The test is designed to test the `ComputeTimeDerivative` action for DG.
  // This action does a lot:
  //
  // - compute partial derivatives as needed
  // - compute the time derivative from
  //   `System::compute_volume_time_derivative`. This includes fluxes, sources,
  //   and nonconservative products.
  // - adds moving mesh terms as needed.
  // - compute flux divergence and add to the time derivative.
  // - compute mortar data for internal boundaries.
  //
  // The action supports conservative systems, and nonconservative systems
  // (mixed conservative-nonconservative systems will be added in the future).
  //
  // To test the action thoroughly we need to test a lot of different
  // combinations:
  //
  // - system type (conservative/nonconservative), using the enum SystemType
  // - 1d, 2d, 3d
  // - whether the mesh is moving or not
  //
  // Note that because the test is quite expensive to build, we have split the
  // compilation across multiple translation units by having the test be defined
  // in ComputeTimeDerivativeImpl.tpp.

  test_wrapper<SystemType::Nonconservative, UseBoundaryCorrection::No>();
  test_wrapper<SystemType::Conservative, UseBoundaryCorrection::No>();

  test_wrapper<SystemType::Conservative, UseBoundaryCorrection::Yes>();
  test_wrapper<SystemType::Nonconservative, UseBoundaryCorrection::Yes>();
  test_wrapper<SystemType::Mixed, UseBoundaryCorrection::Yes>();

  for (const bool use_moving_mesh : {true, false}) {
    WithInverseSpatialMetricTag::test<1>(use_moving_mesh);
  }
}
}  // namespace
}  // namespace TestHelpers::evolution::dg::Actions
