// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/NumericData.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "Utilities/FileSystem.hpp"

namespace ray_tracing {

namespace {

template <typename DataType>
using DerivLapse =
    ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>, Frame::Inertial>;
template <typename DataType>
using DerivShift = ::Tags::deriv<gr::Tags::Shift<DataType, 3>, tmpl::size_t<3>,
                                 Frame::Inertial>;
template <typename DataType>
using DerivInvSpatialMetric =
    ::Tags::deriv<gr::Tags::InverseSpatialMetric<DataType, 3>, tmpl::size_t<3>,
                  Frame::Inertial>;
template <typename DataType>
using DerivSpatialMetric = ::Tags::deriv<gr::Tags::SpatialMetric<DataType, 3>,
                                         tmpl::size_t<3>, Frame::Inertial>;
template <typename DataType>
using solution_vars_list =
    tmpl::list<gr::Tags::Lapse<DataType>, DerivLapse<DataType>,
               gr::Tags::Shift<DataType, 3>, DerivShift<DataType>,
               gr::Tags::SpatialMetric<DataType, 3>,
               gr::Tags::InverseSpatialMetric<DataType, 3>,
               DerivSpatialMetric<DataType>,
               gr::Tags::ExtrinsicCurvature<DataType, 3>>;

void make_test_volume_data_file(const std::string& volfile_name) {
  // Create a simple volume data file with Schwarzschild spacetime in
  // Kerr-Schild coordinates for testing.
  const gr::Solutions::KerrSchild solution{/* mass */ 1.,
                                           /* spin */ {{0., 0., 0.}},
                                           /* center */ {{0., 0., 0.}}};
  const size_t num_points_per_dim = 6;
  const Mesh<3> mesh{num_points_per_dim, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const domain::creators::Sphere domain_creator{
      3.0, 4.0, domain::creators::Sphere::Excision{}, 0_st, num_points_per_dim,
      true};
  const auto domain = domain_creator.create_domain();
  const auto element_ids =
      initial_element_ids(domain_creator.initial_refinement_levels());
  h5::H5File<h5::AccessType::ReadWrite> h5file(volfile_name);
  auto& volfile = h5file.insert<h5::VolumeData>("/VolumeData", 0);
  std::vector<ElementVolumeData> element_voldata;
  element_voldata.reserve(element_ids.size());
  for (const auto& element_id : element_ids) {
    const auto logical_coords = logical_coordinates(mesh);
    const ElementMap<3, Frame::Inertial> element_map{
        element_id, domain.blocks()[element_id.block_id()]};
    const auto inertial_coords = element_map(logical_coords);
    std::vector<TensorComponent> tensor_components;
    const auto solution_vars = solution.variables(
        inertial_coords, 0.0, solution_vars_list<DataVector>{});
    const auto vars = tuples::tagged_tuple_cat(
        solution_vars,
        tuples::TaggedTuple<DerivInvSpatialMetric<DataVector>>{
            gr::deriv_inverse_spatial_metric(
                get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                    solution_vars),
                get<DerivSpatialMetric<DataVector>>(solution_vars))});
    tmpl::for_each<typename decltype(vars)::tags_list>(
        [&tensor_components, &vars](auto tag_v) {
          using Tag = tmpl::type_from<decltype(tag_v)>;
          const auto& tensor = get<Tag>(vars);
          const std::string tensor_name = db::tag_name<Tag>();
          for (size_t i = 0; i < tensor.size(); ++i) {
            tensor_components.emplace_back(
                tensor_name + tensor.component_suffix(i), tensor[i]);
          }
        });
    element_voldata.emplace_back(element_id, std::move(tensor_components),
                                 mesh);
  }
  volfile.write_volume_data(0, 0.0, element_voldata, serialize(domain));
}

void test_numeric_data(const std::string& options_string) {
  // Test factory-creation
  auto created =
      TestHelpers::test_factory_creation<BackgroundSpacetime, NumericData>(
          options_string);
  REQUIRE(dynamic_cast<const NumericData*>(created.get()) != nullptr);
  auto& background_spacetime = dynamic_cast<NumericData&>(*created);
  {
    INFO("Semantics");
    test_serialization(background_spacetime);
    test_copy_semantics(background_spacetime);
    auto move_background_spacetime = background_spacetime;
    test_move_semantics(std::move(move_background_spacetime),
                        background_spacetime);
    const auto clone = background_spacetime.get_clone();
    REQUIRE(dynamic_cast<const NumericData*>(clone.get()) != nullptr);
    CHECK(dynamic_cast<const NumericData&>(*clone) == background_spacetime);
  }
  {
    INFO("Variables");
    const tnsr::I<double, 3> x{{1.0, 2.0, 3.0}};
    background_spacetime.initialize({0.0, 1.0});
    const auto vars = background_spacetime.variables(x, 0.0);
    const gr::Solutions::KerrSchild solution{/* mass */ 1.,
                                             /* spin */ {{0., 0., 0.}},
                                             /* center */ {{0., 0., 0.}}};
    const auto solution_vars =
        solution.variables(x, 0.0, solution_vars_list<double>{});
    const auto expected_vars = tuples::tagged_tuple_cat(
        solution_vars,
        tuples::TaggedTuple<DerivInvSpatialMetric<double>>{
            gr::deriv_inverse_spatial_metric(
                get<gr::Tags::InverseSpatialMetric<double, 3>>(solution_vars),
                get<DerivSpatialMetric<double>>(solution_vars))});
    const Approx custom_approx = Approx::custom().epsilon(1e-4).scale(1.0);
    tmpl::for_each<BackgroundSpacetime::tags>([&](auto tag_v) {
      using Tag = tmpl::type_from<decltype(tag_v)>;
      CHECK_ITERABLE_CUSTOM_APPROX(get<Tag>(vars), get<Tag>(expected_vars),
                                   custom_approx);
    });
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.RayTracer.BackgroundSpacetimes.NumericData",
                  "[Unit][ParallelAlgorithms]") {
  domain::creators::register_derived_with_charm();
  const std::string volfile_name =
      "Unit.RayTracer.BackgroundSpacetimes.NumericData.h5";
  if (file_system::check_if_file_exists(volfile_name)) {
    file_system::rm(volfile_name, true);
  }
  make_test_volume_data_file(volfile_name);
  test_numeric_data(
      "NumericData:\n"
      "  FileGlob: " +
      volfile_name +
      "\n"
      "  SubfileName: VolumeData\n"
      "  ObservationStep: -1\n"
      "  Verbosity: Debug\n");
  if (file_system::check_if_file_exists(volfile_name)) {
    file_system::rm(volfile_name, true);
  }
}

}  // namespace ray_tracing
