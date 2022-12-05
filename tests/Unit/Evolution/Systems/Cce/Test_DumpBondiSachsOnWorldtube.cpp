// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Callbacks/DumpBondiSachsOnWorldtube.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct test_metavariables {
  struct Target : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial>,
                   GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>,
                   GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>;
    using compute_target_points =
        intrp::TargetPoints::Sphere<Target, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::DumpBondiSachsOnWorldtube<Target>;
    using compute_items_on_target = tmpl::list<>;
  };

  using const_global_cache_tags =
      tmpl::list<intrp::Tags::Sphere<Target>, Cce::Tags::FilePrefix>;
  using component_list = tmpl::list<>;
};

std::string get_filename(const std::string& filename_prefix,
                         const double radius) {
  return MakeString{} << filename_prefix << "CceR" << std::setfill('0')
                      << std::setw(4) << std::lround(radius) << ".h5";
}

std::string replace_name(const std::string& db_tag_name) {
  if (db_tag_name == "BondiBeta") {
    return "Beta";
  } else if (db_tag_name == "Dr(J)") {
    return "DrJ";
  } else if (db_tag_name == "Du(R)") {
    return "DuR";
  } else {
    return db_tag_name;
  }
}

template <typename Tags>
auto make_spacetime_variables(const size_t size) {
  Variables<Tags> spacetime_variables{size, 0.0};
  auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial>>(spacetime_variables);
  get<0, 0>(spacetime_metric) = -1.0;
  for (size_t i = 1; i < 4; i++) {
    spacetime_metric.get(i, i) = 1.0;
  }
  return spacetime_variables;
}

void test(const std::string& filename_prefix,
          const std::vector<double>& radii) {
  using metavars = test_metavariables;
  using target = typename metavars::Target;
  using spacetime_tags = typename target::vars_to_interpolate_to_target;
  using cce_tags =
      typename target::post_interpolation_callback::cce_boundary_tags;
  using written_cce_tags =
      typename target::post_interpolation_callback::cce_tags_to_dump;

  // Choose only l_max = 2 for two reasons:
  //   1. Speed
  //   2. So we can easily check the legend by hand
  const size_t l_max = 2;
  const size_t num_points_single_sphere =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const std::vector<std::string> expected_all_legend{
      "Time",     "Re(0,0)",  "Im(0,0)",  "Re(1,-1)", "Im(1,-1)",
      "Re(1,0)",  "Im(1,0)",  "Re(1,1)",  "Im(1,1)",  "Re(2,-2)",
      "Im(2,-2)", "Re(2,-1)", "Im(2,-1)", "Re(2,0)",  "Im(2,0)",
      "Re(2,1)",  "Im(2,1)",  "Re(2,2)",  "Im(2,2)"};
  const std::vector<std::string> expected_real_legend{
      "Time",    "Re(0,0)", "Re(1,0)", "Re(1,1)", "Im(1,1)",
      "Re(2,0)", "Re(2,1)", "Im(2,1)", "Re(2,2)", "Im(2,2)"};

  // It doesn't really matter what the GH data is so long as we are able to
  // calculate bondi data, because this is just testing that we can write the
  // data. So choose Minkowski spacetime. This means pi, phi, and deriv phi are
  // trivially 0.
  Variables<spacetime_tags> spacetime_variables =
      make_spacetime_variables<spacetime_tags>(radii.size() *
                                               num_points_single_sphere);

  // Options for Sphere
  const intrp::AngularOrdering angular_ordering = intrp::AngularOrdering::Cce;
  const std::array<double, 3> center = {{0.05, 0.06, 0.07}};
  intrp::OptionHolders::Sphere sphere_opts(l_max, center, radii,
                                           angular_ordering);

  Parallel::MutableGlobalCache<metavars> mutable_cache{};
  Parallel::GlobalCache<metavars> cache{
      {std::move(sphere_opts), filename_prefix}, &mutable_cache};

  // Only need variables in the box for this test
  using db_tags = tmpl::list<::Tags::Variables<spacetime_tags>>;
  const auto box = db::create<db_tags>(spacetime_variables);

  // Check the error
  CHECK_THROWS_WITH(
      ([&box, &radii, &center, &filename_prefix, &mutable_cache]() {
        const intrp::AngularOrdering local_angular_ordering =
            intrp::AngularOrdering::Strahlkorper;
        intrp::OptionHolders::Sphere local_sphere_opts(l_max, center, radii,
                                                       local_angular_ordering);
        Parallel::GlobalCache<metavars> local_cache{
            {std::move(local_sphere_opts), filename_prefix}, &mutable_cache};

        target::post_interpolation_callback::apply(box, local_cache, 0.1);
      })(),
      Catch::Contains(
          "To use the DumpBondiSachsOnWorldtube post interpolation callback, "
          "the angular ordering of the Spheres must be Cce"));

  const std::vector<double> times{0.9, 1.3};

  for (const double time : times) {
    target::post_interpolation_callback::apply(box, cache, time);
  }

  Variables<spacetime_tags> single_spacetime_variables =
      make_spacetime_variables<spacetime_tags>(num_points_single_sphere);
  const auto& [spacetime_metric, pi, phi] = single_spacetime_variables;
  Variables<cce_tags> bondi_boundary_data{num_points_single_sphere};

  for (size_t i = 0; i < radii.size(); i++) {
    const double radius = radii[i];
    CAPTURE(radius);
    // Have to create the bondi data for every radius individually
    Cce::create_bondi_boundary_data(make_not_null(&bondi_boundary_data), phi,
                                    pi, spacetime_metric, radius, l_max);
    const auto file = h5::H5File<h5::AccessType::ReadOnly>(
        get_filename(filename_prefix, radius));

    tmpl::for_each<written_cce_tags>([&file, &bondi_boundary_data, &times,
                                      &l_max, &expected_all_legend,
                                      &expected_real_legend](auto tag_v) {
      using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
      const auto& bondi_data = get(get<tag>(bondi_boundary_data));
      constexpr int spin = tag::tag::type::type::spin;
      constexpr bool is_real = spin == 0;
      const auto& expected_legend =
          is_real ? expected_real_legend : expected_all_legend;

      SpinWeighted<ComplexModalVector, spin> expected_data{square(l_max + 1)};
      Spectral::Swsh::libsharp_to_goldberg_modes(
          make_not_null(&expected_data),
          Spectral::Swsh::swsh_transform(l_max, 1, bondi_data), l_max);

      const std::string tag_path =
          "/" + replace_name(db::tag_name<typename tag::tag>());
      CAPTURE(tag_path);
      const auto& dat_file = file.get<h5::Dat>(tag_path);
      const Matrix written_data = dat_file.get_data();

      CHECK(expected_legend == dat_file.get_legend());
      CHECK(times.size() == written_data.rows());
      const size_t expected_data_size = square(l_max + 1) * (is_real ? 1 : 2);
      CHECK(expected_data_size == written_data.columns() - 1);

      // Since the metric isn't changing it should be the same data on each
      // row just with a different time
      for (size_t j = 0; j < times.size(); j++) {
        const double time = times[j];
        CAPTURE(time);
        CHECK(time == written_data(j, 0));
        size_t counter = 1;
        for (size_t ell = 0; ell <= l_max; ell++) {
          for (size_t m = is_real ? 0 : -ell; m <= ell; m++) {
            const size_t goldberg_index =
                Spectral::Swsh::goldberg_mode_index(l_max, ell, m);
            CHECK(written_data(j, counter) ==
                  real(expected_data.data()[goldberg_index]));
            counter++;
            if (not is_real or m != 0) {
              CHECK(written_data(j, counter) ==
                    imag(expected_data.data()[goldberg_index]));
              counter++;
            }
          }
        }
      }

      file.close_current_object();
    });
  }
}

void delete_files(const std::string& filename_prefix,
                  const std::vector<double>& radii_to_delete) {
  for (const auto& radius : radii_to_delete) {
    const std::string filename = get_filename(filename_prefix, radius);
    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, true);
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.DumpBondiSachsOnWorldtube",
                  "[Unit][Cce]") {
  const std::string filename_prefix{"Shrek-the-Third"};
  std::vector<double> radii{100.0};

  delete_files(filename_prefix, radii);
  test(filename_prefix, radii);

  radii.push_back(150.0);
  radii.push_back(200.0 - std::numeric_limits<double>::epsilon());

  delete_files(filename_prefix, radii);
  test(filename_prefix, radii);
  delete_files(filename_prefix, radii);
}
}  // namespace
