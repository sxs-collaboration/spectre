// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <pup.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/IO/VolumeData.hpp"
#include "Helpers/ParallelAlgorithms/Interpolation/InterpolationTargetTestHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/AngularOrdering.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/FillYlmLegendAndData.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/AddTemporalIdsToInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/CleanUpInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetReceiveVars.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceivePoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceiveVolumeData.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorRegisterElement.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/TryToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveSurfaceData.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/KerrHorizon.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace gr::surfaces::Tags {
template <typename Frame>
struct AreaElement;
template <typename IntegrandTag, typename Frame>
struct SurfaceIntegral;
}  // namespace gr::surfaces::Tags

namespace {

template <InterpTargetTestHelpers::ValidPoints ValidPoints>
domain::creators::Sphere make_sphere() {
  if constexpr (ValidPoints == InterpTargetTestHelpers::ValidPoints::All) {
    return {0.9, 4.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false};
  }
  if constexpr (ValidPoints == InterpTargetTestHelpers::ValidPoints::None) {
    return {4.9, 8.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false};
  }
  return {3.4, 4.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false};
}

void check_ylm_data(const std::string& h5_file_name) {
  // Parameters chosen to match SurfaceE choices below
  constexpr size_t l_max = 3;
  constexpr std::array<double, 3> expansion_center{{0.04, 0.05, 0.06}};
  constexpr double mass = 1.1;
  constexpr std::array<double, 3> dimensionless_spin{{1.0, 0.0, 0.0}};

  ylm::Strahlkorper<Frame::Inertial> expected_surface(
      l_max, l_max,
      get(gr::Solutions::kerr_horizon_radius(
          ::ylm::Spherepack(l_max, l_max).theta_phi_points(), mass,
          dimensionless_spin)),
      expansion_center);

  const std::vector<std::string> ylm_expected_legend{
      "Time",
      "InertialExpansionCenter_x",
      "InertialExpansionCenter_y",
      "InertialExpansionCenter_z",
      "Lmax",
      "coef(0,0)",
      "coef(1,-1)",
      "coef(1,0)",
      "coef(1,1)",
      "coef(2,-2)",
      "coef(2,-1)",
      "coef(2,0)",
      "coef(2,1)",
      "coef(2,2)",
      "coef(3,-3)",
      "coef(3,-2)",
      "coef(3,-1)",
      "coef(3,0)",
      "coef(3,1)",
      "coef(3,2)",
      "coef(3,3)"};
  const size_t expected_num_columns = ylm_expected_legend.size();

  // Check that the H5 file was written correctly.
  const auto file = h5::H5File<h5::AccessType::ReadOnly>(h5_file_name);
  const auto& ylm_dat_file = file.get<h5::Dat>("/SurfaceE_Ylm");
  const Matrix ylm_written_data = ylm_dat_file.get_data();
  const auto& ylm_written_legend = ylm_dat_file.get_legend();

  CHECK(ylm_written_legend.size() == expected_num_columns);
  CHECK(ylm_written_data.columns() == expected_num_columns);
  CHECK(ylm_written_legend == ylm_expected_legend);

  std::vector<double> ylm_expected_data{0.0, expansion_center[0],
                                        expansion_center[1],
                                        expansion_center[2], l_max};

  ylm::SpherepackIterator iter(l_max, l_max);
  for (size_t l = 0; l <= l_max; l++) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {
      iter.set(l, m);
      ylm_expected_data.push_back(expected_surface.coefficients()[iter()]);
    }
  }

  ASSERT(ylm_expected_data.size() == expected_num_columns,
         "The size of the constructed test Ylm legend ("
             << expected_num_columns
             << ") and the number of columns in the constructed test Ylm data ("
             << ylm_expected_data.size() << ") do not match.");

  for (size_t i = 0; i < expected_num_columns; i++) {
    CHECK(ylm_written_data(0, i) == ylm_expected_data[i]);
  }
}

// This tests the helper function for intrp::callbacks::ObserveSurfaceData that
// fills in the Ylm legend and data to write to disk but with a max_l value that
// is greater than the l_max of the surface to write. We test this separately
// since currently, ObserveSurfaceData will only ever pass in the surface's
// l_max as the max_l value since l_max for a surface does not change over the
// course of a simulation.
void check_ylm_data_with_greater_max_l() {
  const double time = 2.2;
  constexpr std::array<double, 3> expansion_center{{-1.0, -2.0, -3.0}};
  constexpr size_t l_max = 3;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  const auto radius = make_with_random_values<DataVector>(
      make_not_null(&generator), distribution,
      DataVector(ylm::Spherepack::physical_size(l_max, l_max),
                 std::numeric_limits<double>::signaling_NaN()));
  const ylm::Strahlkorper<Frame::Distorted> strahlkorper(l_max, l_max, radius,
                                                         expansion_center);

  const size_t max_l = 4;  // max_l > l_max
  const std::vector<std::string> ylm_expected_legend{
      "Time",
      "DistortedExpansionCenter_x",
      "DistortedExpansionCenter_y",
      "DistortedExpansionCenter_z",
      "Lmax",
      "coef(0,0)",
      "coef(1,-1)",
      "coef(1,0)",
      "coef(1,1)",
      "coef(2,-2)",
      "coef(2,-1)",
      "coef(2,0)",
      "coef(2,1)",
      "coef(2,2)",
      "coef(3,-3)",
      "coef(3,-2)",
      "coef(3,-1)",
      "coef(3,0)",
      "coef(3,1)",
      "coef(3,2)",
      "coef(3,3)",
      "coef(4,-4)",
      "coef(4,-3)",
      "coef(4,-2)",
      "coef(4,-1)",
      "coef(4,0)",
      "coef(4,1)",
      "coef(4,2)",
      "coef(4,3)",
      "coef(4,4)"};
  const size_t expected_num_columns = ylm_expected_legend.size();

  std::vector<double> ylm_expected_data{time, expansion_center[0],
                                        expansion_center[1],
                                        expansion_center[2], l_max};

  ylm::SpherepackIterator iter(l_max, l_max);
  for (size_t l = 0; l <= l_max; l++) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {
      iter.set(l, m);
      ylm_expected_data.push_back(strahlkorper.coefficients()[iter()]);
    }
  }
  for (size_t l = l_max + 1; l <= max_l; l++) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {
      ylm_expected_data.push_back(0.0);
    }
  }

  ASSERT(ylm_expected_data.size() == expected_num_columns,
         "The size of the constructed test Ylm legend ("
             << expected_num_columns
             << ") and the number of columns in the constructed test Ylm data ("
             << ylm_expected_data.size() << ") do not match.");

  std::vector<std::string> ylm_written_legend;
  std::vector<double> ylm_written_data;

  ylm::fill_ylm_legend_and_data(make_not_null(&ylm_written_legend),
                                make_not_null(&ylm_written_data), strahlkorper,
                                time, max_l);

  CHECK(ylm_written_legend.size() == expected_num_columns);
  CHECK(ylm_written_data.size() == expected_num_columns);
  CHECK(ylm_written_legend == ylm_expected_legend);

  for (size_t i = 0; i < expected_num_columns; i++) {
    CHECK(ylm_written_data[i] == ylm_expected_data[i]);
  }
}

template <InterpTargetTestHelpers::ValidPoints ValidPoints>
void check_surface_volume_data(const std::string& surfaces_file_prefix) {
  // Parameters chosen to match SurfaceD choices below
  constexpr size_t l_max = 10;
  constexpr size_t m_max = 10;
  constexpr double sphere_radius = 2.8;
  constexpr std::array<double, 3> center{{0.01, 0.02, 0.03}};
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{l_max, m_max,
                                                        sphere_radius, center};
  const ylm::Spherepack& ylm = strahlkorper.ylm_spherepack();
  const std::vector<size_t> extents{
      {ylm.physical_extents()[0], ylm.physical_extents()[1]}};
  const std::array<DataVector, 2> theta_phi = ylm.theta_phi_points();
  const DataVector theta = theta_phi[0];
  const DataVector phi = theta_phi[1];
  const DataVector sin_theta = sin(theta);
  const DataVector radius = ylm.spec_to_phys(strahlkorper.coefficients());
  const std::string grid_name{"SurfaceD"};

  const auto x{radius * sin_theta * cos(phi) + center[0]};
  const auto y{radius * sin_theta * sin(phi) + center[1]};
  const auto z{radius * cos(theta) + center[2]};
  const std::vector<DataVector> tensor_and_coord_data{
      x, y, z, square(2.0 * x + 3.0 * y + 5.0 * z)};
  const std::vector<TensorComponent> tensor_components{
      {grid_name + "/InertialCoordinates_x", tensor_and_coord_data[0]},
      {grid_name + "/InertialCoordinates_y", tensor_and_coord_data[1]},
      {grid_name + "/InertialCoordinates_z", tensor_and_coord_data[2]},
      {grid_name + "/Square", tensor_and_coord_data[3]}};

  const std::vector<Spectral::Basis> bases{2,
                                           Spectral::Basis::SphericalHarmonic};
  const std::vector<Spectral::Quadrature> quadratures{
      {Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}};
  const observers::ObservationId observation_id{0., "/SurfaceD.vol"};
  if constexpr (ValidPoints == InterpTargetTestHelpers::ValidPoints::None) {
    TestHelpers::io::VolumeData::check_volume_data(
        surfaces_file_prefix + ".h5"s, 0, grid_name, observation_id.hash(),
        observation_id.value(), std::nullopt, tensor_and_coord_data,
        {grid_name}, {bases}, {quadratures}, {extents},
        {"InertialCoordinates_x"s, "InertialCoordinates_y"s,
         "InertialCoordinates_z"s, "Square"s},
        {{0, 1, 2, 3}}, 1.e-14, 1.0, {"Square"s});
  } else {
    TestHelpers::io::VolumeData::check_volume_data(
        surfaces_file_prefix + ".h5"s, 0, grid_name, observation_id.hash(),
        observation_id.value(), std::nullopt, tensor_and_coord_data,
        {grid_name}, {bases}, {quadratures}, {extents},
        {"InertialCoordinates_x"s, "InertialCoordinates_y"s,
         "InertialCoordinates_z"s, "Square"s},
        {{0, 1, 2, 3}}, 1.e-2);  // loose tolerance because of low resolution
                                 // in the volume, which limits interpolation
                                 // accuracy
  }
}

// Simple DataBoxItems for test.
namespace Tags {
struct TestSolution : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Square : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct SquareCompute : Square, db::ComputeTag {
  static void function(gsl::not_null<Scalar<DataVector>*> result,
                       const Scalar<DataVector>& x) {
    get(*result) = square(get(x));
  }
  using argument_tags = tmpl::list<TestSolution>;
  using base = Square;
  using return_type = Scalar<DataVector>;
};
struct Negate : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct NegateCompute : Negate, db::ComputeTag {
  static void function(gsl::not_null<Scalar<DataVector>*> result,
                       const Scalar<DataVector>& x) {
    get(*result) = -get(x);
  }
  using argument_tags = tmpl::list<Square>;
  using base = Negate;
  using return_type = Scalar<DataVector>;
};
}  // namespace Tags

template <typename Metavariables>
struct MockObserverWriter {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::list<observers::Tags::ReductionFileName,
                                             observers::Tags::SurfaceFileName>;
  using simple_tags =
      typename observers::Actions::InitializeWriter<Metavariables>::simple_tags;
  using compute_tags = typename observers::Actions::InitializeWriter<
      Metavariables>::compute_tags;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<observers::Actions::InitializeWriter<Metavariables>>>,
      Parallel::PhaseActions<Parallel::Phase::Register, tmpl::list<>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;

  using component_being_mocked = observers::ObserverWriter<Metavariables>;
};

template <typename Metavariables, typename InterpolationTargetTag>
struct MockInterpolationTarget {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::flatten<tmpl::append<
      Parallel::get_const_global_cache_tags_from_actions<
          tmpl::flatten<tmpl::list<
              typename InterpolationTargetTag::compute_target_points,
              typename InterpolationTargetTag::post_interpolation_callbacks>>>,
      tmpl::list<domain::Tags::Domain<Metavariables::volume_dim>>>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolationTarget<
              Metavariables, InterpolationTargetTag>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
};

template <typename Metavariables>
struct MockInterpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<::intrp::Actions::InitializeInterpolator<
              tmpl::list<
                  intrp::Tags::VolumeVarsInfo<Metavariables,
                                              ::Tags::TimeStepId>,
                  intrp::Tags::VolumeVarsInfo<Metavariables, ::Tags::Time>>,
              intrp::Tags::InterpolatedVarsHolders<Metavariables>>>>,
      Parallel::PhaseActions<Parallel::Phase::Register, tmpl::list<>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
  using component_being_mocked = intrp::Interpolator<Metavariables>;
};

// This test was originally written with non-sequential targets, but an
// infrastructure change made the interpolator only work with sequential
// targets (horizon find). Rather than rewrite the whole test with horizon
// finds, we just make new targets from the originals that are now sequential
template <typename OriginalComputeTargetPoints>
struct MockComputeTargetPoints : public OriginalComputeTargetPoints {
  using is_sequential = std::true_type;
};

struct MockMetavariables {
  struct SurfaceA : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution, gr::Tags::SpatialMetric<DataVector, 3>>;
    using compute_items_on_target =
        tmpl::list<Tags::SquareCompute,
                   gr::surfaces::Tags::AreaElementCompute<Frame::Inertial>,
                   gr::surfaces::Tags::SurfaceIntegralCompute<
                       Tags::Square, ::Frame::Inertial>>;
    using compute_target_points = MockComputeTargetPoints<
        intrp::TargetPoints::KerrHorizon<SurfaceA, ::Frame::Inertial>>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<gr::surfaces::Tags::SurfaceIntegral<Tags::Square,
                                                           ::Frame::Inertial>>,
            SurfaceA>>;
  };
  struct SurfaceB : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution, gr::Tags::SpatialMetric<DataVector, 3>>;
    using compute_items_on_target =
        tmpl::list<Tags::SquareCompute, Tags::NegateCompute,
                   gr::surfaces::Tags::AreaElementCompute<Frame::Inertial>,
                   gr::surfaces::Tags::SurfaceIntegralCompute<Tags::Square,
                                                              Frame::Inertial>,
                   gr::surfaces::Tags::SurfaceIntegralCompute<Tags::Negate,
                                                              Frame::Inertial>>;
    using compute_target_points = MockComputeTargetPoints<
        intrp::TargetPoints::KerrHorizon<SurfaceB, ::Frame::Inertial>>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<gr::surfaces::Tags::SurfaceIntegral<Tags::Square,
                                                           ::Frame::Inertial>,
                       gr::surfaces::Tags::SurfaceIntegral<Tags::Negate,
                                                           ::Frame::Inertial>>,
            SurfaceB>>;
  };
  struct SurfaceC : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution, gr::Tags::SpatialMetric<DataVector, 3>>;
    using compute_items_on_target =
        tmpl::list<Tags::SquareCompute, Tags::NegateCompute,
                   gr::surfaces::Tags::AreaElementCompute<Frame::Inertial>,
                   gr::surfaces::Tags::SurfaceIntegralCompute<
                       Tags::Negate, ::Frame::Inertial>>;
    using compute_target_points = MockComputeTargetPoints<
        intrp::TargetPoints::KerrHorizon<SurfaceC, ::Frame::Inertial>>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<gr::surfaces::Tags::SurfaceIntegral<Tags::Negate,
                                                           ::Frame::Inertial>>,
            SurfaceC>>;
  };

  struct SurfaceD : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution, gr::Tags::SpatialMetric<DataVector, 3>>;
    using compute_items_on_target =
        tmpl::list<Tags::SquareCompute,
                   gr::surfaces::Tags::AreaElementCompute<Frame::Inertial>,
                   gr::surfaces::Tags::SurfaceIntegralCompute<
                       Tags::Square, ::Frame::Inertial>>;
    using compute_target_points = MockComputeTargetPoints<
        intrp::TargetPoints::KerrHorizon<SurfaceD, ::Frame::Inertial>>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveSurfaceData<
            tmpl::list<Tags::Square>, SurfaceD, ::Frame::Inertial>>;
  };

  struct SurfaceE : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution, gr::Tags::SpatialMetric<DataVector, 3>>;
    using compute_items_on_target =
        tmpl::list<Tags::SquareCompute,
                   gr::surfaces::Tags::AreaElementCompute<Frame::Inertial>,
                   gr::surfaces::Tags::SurfaceIntegralCompute<
                       Tags::Square, ::Frame::Inertial>>;
    using compute_target_points = MockComputeTargetPoints<
        intrp::TargetPoints::KerrHorizon<SurfaceE, ::Frame::Inertial>>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveSurfaceData<
            tmpl::list<Tags::Square>, SurfaceE, ::Frame::Inertial>>;
  };

  using observed_reduction_data_tags = tmpl::list<>;

  using interpolator_source_vars =
      tmpl::list<Tags::TestSolution, gr::Tags::SpatialMetric<DataVector, 3>>;
  using interpolation_target_tags =
      tmpl::list<SurfaceA, SurfaceB, SurfaceC, SurfaceD, SurfaceE>;
  static constexpr size_t volume_dim = 3;
  using component_list =
      tmpl::list<MockObserverWriter<MockMetavariables>,
                 MockInterpolationTarget<MockMetavariables, SurfaceA>,
                 MockInterpolationTarget<MockMetavariables, SurfaceB>,
                 MockInterpolationTarget<MockMetavariables, SurfaceC>,
                 MockInterpolationTarget<MockMetavariables, SurfaceD>,
                 MockInterpolationTarget<MockMetavariables, SurfaceE>,
                 MockInterpolator<MockMetavariables>>;
};

template <InterpTargetTestHelpers::ValidPoints ValidPoints>
void run_test() {
  // Check if either file generated by this test exists and remove them
  // if so. Check for both files existing before the test runs, since
  // both files get written when evaluating the list of post interpolation
  // callbacks below.
  const std::string h5_file_prefix = "Test_ObserveTimeSeriesOnSurface";
  const auto h5_file_name = h5_file_prefix + ".h5";
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  const auto surfaces_file_prefix = "Surfaces";
  if (file_system::check_if_file_exists(surfaces_file_prefix + ".h5"s)) {
    file_system::rm(surfaces_file_prefix + ".h5"s, true);
  }

  // Test That ObserveTimeSeriesOnSurface indeed does conform to its protocol
  using callback_A = tmpl::front<
      typename MockMetavariables::SurfaceA::post_interpolation_callbacks>;
  using callback_B = tmpl::front<
      typename MockMetavariables::SurfaceB::post_interpolation_callbacks>;
  using callback_C = tmpl::front<
      typename MockMetavariables::SurfaceC::post_interpolation_callbacks>;
  using callback_D = tmpl::front<
      typename MockMetavariables::SurfaceD::post_interpolation_callbacks>;
  using callback_E = tmpl::front<
      typename MockMetavariables::SurfaceE::post_interpolation_callbacks>;
  using protocol = intrp::protocols::PostInterpolationCallback;
  static_assert(tt::assert_conforms_to_v<callback_A, protocol>);
  static_assert(tt::assert_conforms_to_v<callback_B, protocol>);
  static_assert(tt::assert_conforms_to_v<callback_C, protocol>);
  static_assert(tt::assert_conforms_to_v<callback_D, protocol>);
  static_assert(tt::assert_conforms_to_v<callback_E, protocol>);

  using metavars = MockMetavariables;
  using interp_component = MockInterpolator<metavars>;
  using target_a_component =
      MockInterpolationTarget<metavars, metavars::SurfaceA>;
  using target_b_component =
      MockInterpolationTarget<metavars, metavars::SurfaceB>;
  using target_c_component =
      MockInterpolationTarget<metavars, metavars::SurfaceC>;
  using target_d_component =
      MockInterpolationTarget<metavars, metavars::SurfaceD>;
  using target_e_component =
      MockInterpolationTarget<metavars, metavars::SurfaceE>;
  using obs_writer = MockObserverWriter<metavars>;

  // Options for all InterpolationTargets.
  intrp::OptionHolders::KerrHorizon kerr_horizon_opts_A(
      10, {{0.0, 0.0, 0.0}}, 1.0, {{0.0, 0.0, 0.0}},
      ylm::AngularOrdering::Strahlkorper);
  intrp::OptionHolders::KerrHorizon kerr_horizon_opts_B(
      10, {{0.0, 0.0, 0.0}}, 2.0, {{0.0, 0.0, 0.0}},
      ylm::AngularOrdering::Strahlkorper);
  intrp::OptionHolders::KerrHorizon kerr_horizon_opts_C(
      10, {{0.0, 0.0, 0.0}}, 1.5, {{0.0, 0.0, 0.0}},
      ylm::AngularOrdering::Strahlkorper);
  intrp::OptionHolders::KerrHorizon kerr_horizon_opts_D(
      10, {{0.01, 0.02, 0.03}}, 1.4, {{0.0, 0.0, 0.0}},
      ylm::AngularOrdering::Strahlkorper);
  // Surface for testing Ylm coefficients are written correctly. Using a
  // non-zero spin because with zero spin, Y_{00} is the only term with a
  // non-zero coefficient
  intrp::OptionHolders::KerrHorizon kerr_horizon_opts_E(
      3, {{0.04, 0.05, 0.06}}, 1.1, {{1.0, 0.0, 0.0}},
      ylm::AngularOrdering::Strahlkorper);
  const auto domain_creator = make_sphere<ValidPoints>();
  tuples::TaggedTuple<
      observers::Tags::ReductionFileName, observers::Tags::SurfaceFileName,
      ::intrp::Tags::KerrHorizon<metavars::SurfaceA>, domain::Tags::Domain<3>,
      ::intrp::Tags::KerrHorizon<metavars::SurfaceB>,
      ::intrp::Tags::KerrHorizon<metavars::SurfaceC>,
      ::intrp::Tags::KerrHorizon<metavars::SurfaceD>,
      ::intrp::Tags::KerrHorizon<metavars::SurfaceE>>
      tuple_of_opts{h5_file_prefix,      surfaces_file_prefix,
                    kerr_horizon_opts_A, domain_creator.create_domain(),
                    kerr_horizon_opts_B, kerr_horizon_opts_C,
                    kerr_horizon_opts_D, kerr_horizon_opts_E};

  // Three mock nodes, with 2, 1, and 4 mock cores.
  ActionTesting::MockRuntimeSystem<metavars> runner{
      std::move(tuple_of_opts), {}, {2, 1, 4}};

  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_group_component<interp_component>(&runner);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t core = 0; core < 7; ++core) {
      ActionTesting::next_action<interp_component>(make_not_null(&runner),
                                                   core);
    }
  }
  ActionTesting::emplace_singleton_component<target_a_component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{1});
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_a_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_singleton_component<target_b_component>(
      &runner, ActionTesting::NodeId{2}, ActionTesting::LocalCoreId{2});
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_b_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_singleton_component<target_c_component>(
      &runner, ActionTesting::NodeId{2}, ActionTesting::LocalCoreId{1});
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_c_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_singleton_component<target_d_component>(
      &runner, ActionTesting::NodeId{2}, ActionTesting::LocalCoreId{3});
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_d_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_singleton_component<target_e_component>(
      &runner, ActionTesting::NodeId{2}, ActionTesting::LocalCoreId{0});
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_e_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_nodegroup_component<obs_writer>(&runner);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t node = 0; node < 2; ++node) {
      ActionTesting::next_action<obs_writer>(make_not_null(&runner), node);
    }
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Register);

  Slab slab(0.0, 1.0);
  TimeStepId temporal_id(true, 0, Time(slab, 0));
  const auto domain = domain_creator.create_domain();

  // Create element_ids.
  std::vector<ElementId<3>> element_ids{};
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs =
        domain_creator.initial_refinement_levels()[block.id()];
    auto elem_ids = initial_element_ids(block.id(), initial_ref_levs);
    element_ids.insert(element_ids.end(), elem_ids.begin(), elem_ids.end());
  }

  // Tell the interpolator how many elements there are by registering
  // each one. Normally intrp::Actions::RegisterElement is called by
  // RegisterElementWithInterpolator, and invoked on the ckLocalBranch
  // of the interpolator that is associated with each element
  // (i.e. the local core on each element).
  // Here we assign elements round-robin to the mock cores.
  // And for group components, the array_index is the global core index.
  const size_t num_cores = runner.num_global_cores();
  std::unordered_map<ElementId<3>, size_t> mock_core_for_each_element;
  size_t core_for_next_element = 0;
  for (const auto& element_id : element_ids) {
    mock_core_for_each_element.insert({element_id, core_for_next_element});
    ActionTesting::simple_action<interp_component,
                                 intrp::Actions::RegisterElement>(
        make_not_null(&runner), core_for_next_element);
    if (++core_for_next_element >= num_cores) {
      core_for_next_element = 0;
    }
  }

  // Tell the InterpolationTargets that we want to interpolate at
  // temporal_id.
  ActionTesting::simple_action<
      target_a_component,
      intrp::Actions::AddTemporalIdsToInterpolationTarget<metavars::SurfaceA>>(
      make_not_null(&runner), 0, temporal_id.substep_time());
  ActionTesting::simple_action<
      target_b_component,
      intrp::Actions::AddTemporalIdsToInterpolationTarget<metavars::SurfaceB>>(
      make_not_null(&runner), 0, temporal_id);
  ActionTesting::simple_action<
      target_c_component,
      intrp::Actions::AddTemporalIdsToInterpolationTarget<metavars::SurfaceC>>(
      make_not_null(&runner), 0, temporal_id);
  ActionTesting::simple_action<
      target_d_component,
      intrp::Actions::AddTemporalIdsToInterpolationTarget<metavars::SurfaceD>>(
      make_not_null(&runner), 0, temporal_id.substep_time());
  ActionTesting::simple_action<
      target_e_component,
      intrp::Actions::AddTemporalIdsToInterpolationTarget<metavars::SurfaceE>>(
      make_not_null(&runner), 0, temporal_id.substep_time());

  CHECK(ActionTesting::is_simple_action_queue_empty<obs_writer>(runner, 0));
  CHECK(ActionTesting::is_simple_action_queue_empty<obs_writer>(runner, 1));
  CHECK(ActionTesting::is_simple_action_queue_empty<obs_writer>(runner, 2));

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // Create volume data and send it to the interpolator.
  for (const auto& element_id : element_ids) {
    const auto& block = domain.blocks()[element_id.block_id()];
    ::Mesh<3> mesh{domain_creator.initial_extents()[element_id.block_id()],
                   Spectral::Basis::Legendre,
                   Spectral::Quadrature::GaussLobatto};
    if (block.is_time_dependent()) {
      ERROR("The block must be time-independent");
    }
    ElementMap<3, Frame::Inertial> map{element_id,
                                       block.stationary_map().get_clone()};
    const auto inertial_coords = map(logical_coordinates(mesh));
    ::Variables<typename metavars::interpolator_source_vars> output_vars(
        mesh.number_of_grid_points());
    auto& test_solution = get<Tags::TestSolution>(output_vars);

    // Fill test_solution with some analytic solution.
    get(test_solution) = 2.0 * get<0>(inertial_coords) +
                         3.0 * get<1>(inertial_coords) +
                         5.0 * get<2>(inertial_coords);

    // Fill the metric with Minkowski for simplicity.  The
    // InterpolationTarget is called "KerrHorizon" merely because the
    // surface corresponds to where the horizon *would be* in a Kerr
    // spacetime in Kerr-Schild coordinates; this in no way requires
    // that there is an actual horizon or that the metric is Kerr.
    gr::Solutions::Minkowski<3> solution;
    get<gr::Tags::SpatialMetric<DataVector, 3>>(output_vars) =
        get<gr::Tags::SpatialMetric<DataVector, 3>>(solution.variables(
            inertial_coords, 0.0,
            tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>>{}));

    // Call the InterpolatorReceiveVolumeData action on each element_id.
    ActionTesting::simple_action<interp_component,
                                 intrp::Actions::InterpolatorReceiveVolumeData<
                                     typename metavars::SurfaceA::temporal_id>>(
        make_not_null(&runner), mock_core_for_each_element.at(element_id),
        temporal_id.substep_time(), element_id, mesh, output_vars);
    ActionTesting::simple_action<interp_component,
                                 intrp::Actions::InterpolatorReceiveVolumeData<
                                     typename metavars::SurfaceB::temporal_id>>(
        make_not_null(&runner), mock_core_for_each_element.at(element_id),
        temporal_id, element_id, mesh, std::move(output_vars));
  }

  // Invoke remaining actions in random order.
  MAKE_GENERATOR(generator);
  auto array_indices_with_queued_simple_actions =
      ActionTesting::array_indices_with_queued_simple_actions<
          metavars::component_list>(make_not_null(&runner));
  while (ActionTesting::number_of_elements_with_queued_simple_actions<
             metavars::component_list>(
             array_indices_with_queued_simple_actions) > 0) {
    ActionTesting::invoke_random_queued_simple_action<metavars::component_list>(
        make_not_null(&runner), make_not_null(&generator),
        array_indices_with_queued_simple_actions);
    array_indices_with_queued_simple_actions =
        ActionTesting::array_indices_with_queued_simple_actions<
            metavars::component_list>(make_not_null(&runner));
  }

  // There should be seven more threaded actions, so invoke them and check
  // that there are no more.  They should all be on node zero.
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);
  CHECK(ActionTesting::is_threaded_action_queue_empty<obs_writer>(runner, 0));
  CHECK(ActionTesting::is_threaded_action_queue_empty<obs_writer>(runner, 1));
  CHECK(ActionTesting::is_threaded_action_queue_empty<obs_writer>(runner, 2));

  // By hand compute integral(r^2 d(cos theta) dphi (2x+3y+5z)^2)
  const std::vector<double> expected_integral_a{2432.0 * M_PI / 3.0};
  // SurfaceB has a larger radius by a factor of 2 than SurfaceA,
  // but the same function.  This results in a factor of 4 increase
  // (because the integrand scales like r^2), and an additional factor of
  // 4 (from the area element), for a net factor of 16.  There is also a
  // minus sign for "Negate".
  const std::vector<double> expected_integral_b{16.0 * 2432.0 * M_PI / 3.0,
                                                -16.0 * 2432.0 * M_PI / 3.0};
  // SurfaceC has a larger radius by a factor of 1.5 than SurfaceA,
  // but the same function.  This results in a factor of 2.25 increase
  // (because the integrand scales like r^2), and an additional factor of
  // 2.25 (from the area element), for a net factor of 5.0625.  There is also a
  // minus sign for "Negate".
  const std::vector<double> expected_integral_c{-5.0625 * 2432.0 * M_PI / 3.0};
  const std::vector<std::string> expected_legend_a{"Time",
                                                   "SurfaceIntegral(Square)"};
  const std::vector<std::string> expected_legend_b{
      "Time", "SurfaceIntegral(Square)", "SurfaceIntegral(Negate)"};
  const std::vector<std::string> expected_legend_c{"Time",
                                                   "SurfaceIntegral(Negate)"};

  // Check that the H5 file was written correctly.
  const auto file = h5::H5File<h5::AccessType::ReadOnly>(h5_file_name);
  auto check_file_contents =
      [&file](const std::vector<double>& expected_integral,
              const std::vector<std::string>& expected_legend,
              const std::string& group_name) {
        file.close_current_object();
        const auto& dat_file = file.get<h5::Dat>(group_name);
        const Matrix written_data = dat_file.get_data();
        const auto& written_legend = dat_file.get_legend();
        CHECK(written_legend == expected_legend);
        CHECK(0.0 == written_data(0, 0));
        // The interpolation is not perfect because I use too few grid points.
        Approx custom_approx = Approx::custom().epsilon(1.e-4).scale(1.0);
        for (size_t i = 0; i < expected_integral.size(); ++i) {
          if constexpr (ValidPoints ==
                        InterpTargetTestHelpers::ValidPoints::None) {
            CHECK_THAT(written_data(0, i + 1), Catch::Matchers::IsNaN());
          } else {
            CHECK(expected_integral[i] ==
                  custom_approx(written_data(0, i + 1)));
          }
        }
      };
  check_file_contents(expected_integral_a, expected_legend_a, "/SurfaceA");
  check_file_contents(expected_integral_b, expected_legend_b, "/SurfaceB");
  check_file_contents(expected_integral_c, expected_legend_c, "/SurfaceC");

  // Check that the Ylm data were written correctly
  // As this data depends only on the known target (a KerrHorizon) it
  // uses no interpolated data
  check_ylm_data(h5_file_name);

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  // Check that the Surfaces file contains the correct surface data
  check_surface_volume_data<ValidPoints>(surfaces_file_prefix);

  if (file_system::check_if_file_exists(surfaces_file_prefix + ".h5"s)) {
    file_system::rm(surfaces_file_prefix + ".h5"s, true);
  }

  // This check also uses no interpolated data
  check_ylm_data_with_greater_max_l();
}

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Interpolator.ObserveTimeSeriesAndSurfaceData",
    "[Unit]") {
  domain::creators::register_derived_with_charm();
  run_test<InterpTargetTestHelpers::ValidPoints::All>();
  run_test<InterpTargetTestHelpers::ValidPoints::None>();
  // ValidPoints::Some is not tested as that would vastly increase the
  // complexity of the test for limited gain.  In order to test
  // properly would require passing in the list of valid points.  The
  // only difference between the cases All and Some would be that
  // invalid points print nan, which is only tested by None.
}
}  // namespace
