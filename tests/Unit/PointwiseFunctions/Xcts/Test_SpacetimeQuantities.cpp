// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Xcts/SpacetimeQuantities.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Xcts {

namespace {
template <typename Computed, typename... Args>
void check_with_python(const Computed& computed,
                       const std::string& function_name, const Args&... args) {
  CAPTURE(function_name);
  const auto expected = pypp::call<Computed>(
      "PointwiseFunctions.Xcts.SpacetimeQuantities", function_name, args...);
  Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(computed, expected, custom_approx);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Xcts.SpacetimeQuantities",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  // Set up a mesh for numerical derivatives
  const Mesh<3> mesh{8, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const size_t num_points = mesh.number_of_grid_points();
  auto inv_jacobian = make_with_value<
      InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>>(
      num_points, 0.);
  get<0, 0>(inv_jacobian) = 1.;
  get<1, 1>(inv_jacobian) = 1.;
  get<2, 2>(inv_jacobian) = 1.;
  // Generate random input vars
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist_factor{0.5, 2.};
  std::uniform_real_distribution<double> dist_isotropic{-1., 1.};
  const auto conformal_factor = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), make_not_null(&dist_factor), num_points);
  const auto lapse_times_conformal_factor =
      make_with_random_values<Scalar<DataVector>>(
          make_not_null(&gen), make_not_null(&dist_factor), num_points);
  const auto shift_excess = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&gen), make_not_null(&dist_isotropic), num_points);
  const auto conformal_metric =
      make_with_random_values<tnsr::ii<DataVector, 3>>(
          make_not_null(&gen), make_not_null(&dist_isotropic), num_points);
  const auto inv_conformal_metric =
      make_with_random_values<tnsr::II<DataVector, 3>>(
          make_not_null(&gen), make_not_null(&dist_isotropic), num_points);
  const auto shift_background = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&gen), make_not_null(&dist_isotropic), num_points);
  // Check output vars
  const auto box = db::create<
      tmpl::remove_duplicates<typename Tags::SpacetimeQuantitiesCompute<
          typename SpacetimeQuantities::tags_list>::argument_tags>,
      db::AddComputeTags<Tags::SpacetimeQuantitiesCompute<
          typename SpacetimeQuantities::tags_list>>>(
      mesh, conformal_factor, lapse_times_conformal_factor, shift_excess,
      conformal_metric, inv_conformal_metric, shift_background,
      std::move(inv_jacobian));
  const auto& vars =
      get<::Tags::Variables<typename SpacetimeQuantities::tags_list>>(box);
  check_with_python(get<gr::Tags::SpatialMetric<3>>(vars), "spatial_metric",
                    conformal_factor, conformal_metric);
  check_with_python(get<gr::Tags::InverseSpatialMetric<3>>(vars),
                    "inv_spatial_metric", conformal_factor,
                    inv_conformal_metric);
  check_with_python(get<gr::Tags::Lapse<DataVector>>(vars), "lapse",
                    conformal_factor, lapse_times_conformal_factor);
  check_with_python(get<gr::Tags::Shift<3>>(vars), "shift", shift_excess,
                    shift_background);
  check_with_python(
      get<gr::Tags::ExtrinsicCurvature<3>>(vars), "extrinsic_curvature",
      get<gr::Tags::Lapse<DataVector>>(vars), get<gr::Tags::Shift<3>>(vars),
      get<::Tags::deriv<gr::Tags::Shift<3>, tmpl::size_t<3>, Frame::Inertial>>(
          vars),
      get<gr::Tags::SpatialMetric<3>>(vars),
      get<::Tags::dt<gr::Tags::SpatialMetric<3>>>(vars),
      get<::Tags::deriv<gr::Tags::SpatialMetric<3>, tmpl::size_t<3>,
                        Frame::Inertial>>(vars));
  check_with_python(get<gr::Tags::HamiltonianConstraint<DataVector>>(vars),
                    "hamiltonian_constraint",
                    get<gr::Tags::SpatialRicci<3>>(vars),
                    get<gr::Tags::ExtrinsicCurvature<3>>(vars),
                    get<gr::Tags::InverseSpatialMetric<3>>(vars));
  check_with_python(get<gr::Tags::MomentumConstraint<3>>(vars),
                    "momentum_constraint",
                    get<::Tags::deriv<gr::Tags::ExtrinsicCurvature<3>,
                                      tmpl::size_t<3>, Frame::Inertial>>(vars),
                    get<gr::Tags::InverseSpatialMetric<3>>(vars));
}

}  // namespace Xcts
