// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/SphereTimeDependentMaps.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeVarsToInterpolate.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Frame {
struct ElementLogical;
struct Grid;
struct Distorted;
struct Inertial;
}  // namespace Frame

namespace {
static constexpr size_t num_grid_points = 5;

template <typename Frame>
struct FakeVars : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame>;
};

template <typename Frame>
struct MockComputeVarsToInterpolate
    : tt::ConformsTo<intrp::protocols::ComputeVarsToInterpolate> {
  // Single-frame case
  template <typename SrcTagList, typename DestTagList>
  static void apply(
      const gsl::not_null<Variables<DestTagList>*> /*target_vars*/,
      const Variables<SrcTagList>& /*src_vars*/, const Mesh<3>& /*mesh*/) {}

  // Dual-frame case
  template <typename SrcTagList, typename DestTagList, typename TargetFrame>
  static void apply(
      const gsl::not_null<Variables<DestTagList>*> /*target_vars*/,
      const Variables<SrcTagList>& /*src_vars*/, const Mesh<3>& /*mesh*/,
      const Jacobian<DataVector, 3, TargetFrame, ::Frame::Inertial>&
          jac_target_to_inertial,
      const InverseJacobian<DataVector, 3, TargetFrame, ::Frame::Inertial>&
          invjac_target_to_inertial,
      const Jacobian<DataVector, 3, ::Frame::ElementLogical, TargetFrame>&
          jac_logical_to_target,
      const InverseJacobian<DataVector, 3, ::Frame::ElementLogical,
                            TargetFrame>& invjac_logical_to_target,
      const tnsr::I<DataVector, 3, ::Frame::Inertial>& inertial_mesh_velocity,
      const tnsr::I<DataVector, 3, TargetFrame>&
          grid_to_target_frame_mesh_velocity) {
    // We only want to check that these aren't empty
    CHECK(get<0, 0>(jac_target_to_inertial).size() != 0);
    CHECK(get<0, 0>(invjac_target_to_inertial).size() != 0);
    CHECK(get<0, 0>(jac_logical_to_target).size() != 0);
    CHECK(get<0, 0>(invjac_logical_to_target).size() != 0);
    CHECK(get<0>(inertial_mesh_velocity).size() != 0);
    CHECK(get<0>(grid_to_target_frame_mesh_velocity).size() != 0);
  }

  using allowed_src_tags = tmpl::list<FakeVars<Frame>>;
  using required_src_tags = tmpl::list<FakeVars<Frame>>;
  template <typename TargetFrame>
  using allowed_dest_tags_target_frame = tmpl::list<FakeVars<TargetFrame>>;
  template <typename TargetFrame>
  using allowed_dest_tags = tmpl::list<FakeVars<TargetFrame>>;
  template <typename TargetFrame>
  using required_dest_tags = tmpl::list<FakeVars<TargetFrame>>;
};

static_assert(
    tt::assert_conforms_to_v<MockComputeVarsToInterpolate<::Frame::Grid>,
                             intrp::protocols::ComputeVarsToInterpolate>);

// Doesn't need to conform to a protocol because we only need these type alias.
template <typename Frame>
struct MockInterpolationTargetTag {
  using temporal_id = ::Tags::Time;
  using vars_to_interpolate_to_target = tmpl::list<FakeVars<Frame>>;
  using compute_vars_to_interpolate = MockComputeVarsToInterpolate<Frame>;
};

struct Metavars {
  static constexpr size_t volume_dim = 3;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;
  using component_list = tmpl::list<>;
};

template <typename Frame>
void test() {
  domain::FunctionsOfTime::register_derived_with_charm();

  using TDMO = domain::creators::sphere::TimeDependentMapOptions;
  TDMO time_dep_opts{
      0.0, TDMO::ShapeMapOptions{2, std::nullopt},
      TDMO::TranslationMapOptions{std::array<double, 3>{0.0, 0.0, 0.0},
                                  std::array<double, 3>{0.0, 0.0, 0.0}}};

  const auto domain_creator = domain::creators::Sphere(
      0.9, 4.9, domain::creators::Sphere::Excision{}, 1_st, 7_st, false, {},
      std::vector<double>{2.0},
      std::vector<domain::CoordinateMaps::Distribution>{
          {domain::CoordinateMaps::Distribution::Linear,
           domain::CoordinateMaps::Distribution::Linear}},
      {}, time_dep_opts);
  const Domain<3> domain = domain_creator.create_domain();

  Parallel::GlobalCache<Metavars> cache{{domain_creator.functions_of_time()}};

  Variables<tmpl::list<FakeVars<Frame>>> dest_vars{num_grid_points, 0.0};
  Variables<tmpl::list<FakeVars<::Frame::Inertial>>> source_vars{
      num_grid_points, 0.0};

  const ElementId<3> element_id{0};
  const double time = 0.0;
  const Mesh<3> mesh{num_grid_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

  using tag = MockInterpolationTargetTag<Frame>;

  intrp::InterpolationTarget_detail::compute_dest_vars_from_source_vars<tag>(
      make_not_null(&dest_vars), source_vars, domain, mesh, element_id, cache,
      time);
}

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Interpolator.ComputeDestVarsFromSourceVars",
    "[Unit]") {
  test<::Frame::Grid>();
  test<::Frame::Distorted>();
}
}  // namespace
