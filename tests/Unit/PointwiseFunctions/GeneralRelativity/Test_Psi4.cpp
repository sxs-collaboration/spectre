// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4Real.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylPropagating.hpp"

namespace {
template <typename RealDataType>
void test_compute_item_in_databox(const RealDataType& used_for_size_real) {
  TestHelpers::db::test_compute_tag<gr::Tags::Psi4RealCompute<Frame::Inertial>>(
      "Psi4Real");

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 3.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto spatial_ricci =
      make_with_random_values<tnsr::ii<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);
  const auto extrinsic_curvature =
      make_with_random_values<tnsr::ii<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);
  const auto cov_deriv_extrinsic_curvature =
      make_with_random_values<tnsr::ijj<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3, RealDataType, Frame::Inertial>(
          nn_generator, used_for_size_real);
  const auto inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inertial_coords =
      make_with_random_values<tnsr::I<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);

  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::SpatialRicci<RealDataType, 3>,
          gr::Tags::ExtrinsicCurvature<RealDataType, 3>,
          ::Tags::deriv<gr::Tags::ExtrinsicCurvature<RealDataType, 3>,
                        tmpl::size_t<3>, Frame::Inertial>,
          gr::Tags::SpatialMetric<RealDataType, 3>,
          gr::Tags::InverseSpatialMetric<RealDataType, 3>,
          domain::Tags::Coordinates<3, Frame::Inertial>>,
      db::AddComputeTags<gr::Tags::Psi4RealCompute<Frame::Inertial>>>(
      spatial_ricci, extrinsic_curvature, cov_deriv_extrinsic_curvature,
      spatial_metric, inv_spatial_metric, inertial_coords);
  const auto expected = gr::psi_4_real(
      spatial_ricci, extrinsic_curvature, cov_deriv_extrinsic_curvature,
      spatial_metric, inv_spatial_metric, inertial_coords);
  CHECK((db::get<gr::Tags::Psi4Real<RealDataType>>(box)) == expected);
}

template <typename RealDataType, typename ComplexDataType>
void test_psi_4(const RealDataType& used_for_size_real,
                const ComplexDataType& /*used_for_size_complex*/) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 3.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto spatial_ricci =
      make_with_random_values<tnsr::ii<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);
  const auto extrinsic_curvature =
      make_with_random_values<tnsr::ii<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);
  const auto cov_deriv_extrinsic_curvature =
      make_with_random_values<tnsr::ijj<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3, RealDataType, Frame::Inertial>(
          nn_generator, used_for_size_real);
  const auto inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  auto inertial_coords =
      make_with_random_values<tnsr::I<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);

  const auto python_psi_4 = pypp::call<Scalar<ComplexDataType>>(
      "GeneralRelativity.Psi4", "psi_4", spatial_ricci, extrinsic_curvature,
      cov_deriv_extrinsic_curvature, spatial_metric, inv_spatial_metric,
      inertial_coords);
  const auto expected = gr::psi_4(spatial_ricci, extrinsic_curvature,
                                  cov_deriv_extrinsic_curvature, spatial_metric,
                                  inv_spatial_metric, inertial_coords);
  CHECK_ITERABLE_APPROX(expected, python_psi_4);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Psi4.",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env("PointwiseFunctions/");

  const size_t size = 5;
  const DataVector used_for_size_real_dv =
      DataVector(size, std::numeric_limits<double>::signaling_NaN());
  const ComplexDataVector used_for_size_complex_dv =
      ComplexDataVector(size, std::numeric_limits<double>::signaling_NaN());
  test_psi_4(used_for_size_real_dv, used_for_size_complex_dv);
  test_compute_item_in_databox(used_for_size_real_dv);
}
