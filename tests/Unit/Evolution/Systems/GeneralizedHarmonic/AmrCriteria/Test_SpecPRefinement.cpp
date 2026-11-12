// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/AmrCriteria/SpecPRefinement.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Type.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  static constexpr size_t volume_dim = 3;
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<amr::Criterion,
                             tmpl::list<gh::AmrCriteria::SpecPRefinement>>>;
  };
};

using FilteredTags = gh::System<3>::variables_tag::tags_list;
using Filter = Filters::Filter<3, FilteredTags>;

using DbTags =
    tmpl::list<domain::Tags::Mesh<3>,
               Filters::Tags::SpectralFilter<3, FilteredTags>,
               gr::Tags::SpacetimeMetric<DataVector, 3>,
               gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>;

// The Hypercube filter supports only one half power per executable. With this
// one the number of filtered modes depends on the number of modes.
constexpr unsigned half_power = 8;

Filters::Hypercube<3, FilteredTags> hypercube_filter(
    const bool enable, const bool volume_filter_on_substep,
    const std::optional<size_t> volume_filter_every_n_steps) {
  return {half_power,   enable,
          std::nullopt, volume_filter_on_substep,
          false,        volume_filter_every_n_steps,
          std::nullopt};
}

std::array<amr::Flag, 3> evaluate(
    const amr::Criterion& criterion, const Mesh<3>& mesh, const Filter& filter,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::aa<DataVector, 3>& pi, const tnsr::iaa<DataVector, 3>& phi) {
  Parallel::GlobalCache<Metavariables> empty_cache{};
  auto databox =
      db::create<DbTags>(mesh, filter.get_clone(), spacetime_metric, pi, phi);
  const ObservationBox<tmpl::list<>, db::DataBox<DbTags>> box{
      make_not_null(&databox)};
  return criterion.evaluate(box, empty_cache, ElementId<3>{0});
}

// Modal coefficients that fall off as a power law.
ModalVector power_law(const size_t number_of_modes, const double rate) {
  ModalVector result(number_of_modes);
  for (size_t k = 0; k < number_of_modes; ++k) {
    result[k] = pow(10.0, -rate * static_cast<double>(k));
  }
  return result;
}

// The sum of one function of each logical coordinate, given by its modal
// coefficients. The function of the other coordinates is constant along a
// line in direction d, so above the zeroth mode the directional power monitor
// in direction d is the magnitude of the coefficients given for direction d.
DataVector sum_of_profiles(const Mesh<3>& mesh,
                           const std::array<ModalVector, 3>& coefficients) {
  const auto& extents = mesh.extents();
  DataVector result(mesh.number_of_grid_points(), 0.0);
  for (size_t d = 0; d < 3; ++d) {
    const DataVector profile =
        to_nodal_coefficients(gsl::at(coefficients, d), mesh.slice_through(d));
    for (size_t i = 0; i < extents[0]; ++i) {
      for (size_t j = 0; j < extents[1]; ++j) {
        for (size_t l = 0; l < extents[2]; ++l) {
          const Index<3> point{{{i, j, l}}};
          result[collapsed_index(point, extents)] += profile[point[d]];
        }
      }
    }
  }
  return result;
}

// The truncation error and the truncation error with one less mode, in each
// direction.
using Errors = std::array<std::array<double, 2>, 3>;

Errors errors_of(const std::array<DataVector, 3>& power, const Mesh<3>& mesh,
                 const std::array<size_t, 3>& number_of_filtered_modes) {
  Errors result{};
  for (size_t d = 0; d < 3; ++d) {
    const size_t modes = mesh.extents(d) - gsl::at(number_of_filtered_modes, d);
    const double error =
        PowerMonitors::relative_truncation_error(gsl::at(power, d), modes);
    gsl::at(result, d) = {
        {error, std::max(error, PowerMonitors::relative_truncation_error(
                                    gsl::at(power, d), modes - 1))}};
  }
  return result;
}

// Checks that each direction is refined exactly when the target is below the
// truncation error, and coarsened exactly when the target is at least the
// truncation error with one less mode times SpEC's margin.
void check_thresholds(const Mesh<3>& mesh, const Filter& filter,
                      const tnsr::aa<DataVector, 3>& spacetime_metric,
                      const tnsr::aa<DataVector, 3>& pi,
                      const tnsr::iaa<DataVector, 3>& phi,
                      const Errors& expected_errors) {
  const double margin = pow(10.0, 0.1);
  for (size_t d = 0; d < 3; ++d) {
    CAPTURE(d);
    const auto flag = [&](const double target) {
      return gsl::at(evaluate(gh::AmrCriteria::SpecPRefinement{target}, mesh,
                              filter, spacetime_metric, pi, phi),
                     d);
    };
    const double error = gsl::at(expected_errors, d)[0];
    const double error_with_one_less_mode = gsl::at(expected_errors, d)[1];
    CHECK(flag(0.999 * error) == amr::Flag::IncreaseResolution);
    CHECK(flag(1.001 * error) == amr::Flag::DoNothing);
    CHECK(flag(0.999 * margin * error_with_one_less_mode) ==
          amr::Flag::DoNothing);
    CHECK(flag(1.001 * margin * error_with_one_less_mode) ==
          amr::Flag::DecreaseResolution);
  }
}

void test_creation_and_serialization() {
  register_factory_classes_with_charm<Metavariables>();
  const auto created =
      TestHelpers::test_factory_creation<amr::Criterion,
                                         gh::AmrCriteria::SpecPRefinement>(
          "SpecPRefinement:\n"
          "  RelativeTruncationTarget: 1.e-4\n");
  CHECK(created->type() == amr::Criteria::Type::p);
  CHECK(created->observation_name() == "SpecPRefinement");
  const auto copied = serialize_and_deserialize(created);

  const Mesh<3> mesh{{{7, 8, 9}},
                     Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  tnsr::aa<DataVector, 3> spacetime_metric{number_of_grid_points, 0.0};
  get<0, 0>(spacetime_metric) = sum_of_profiles(
      mesh, {{power_law(7, 0.3), power_law(8, 0.6), power_law(9, 1.0)}});
  const tnsr::aa<DataVector, 3> pi{number_of_grid_points, 0.0};
  const tnsr::iaa<DataVector, 3> phi{number_of_grid_points, 0.0};
  const Filters::None<3, FilteredTags> filter{std::nullopt};
  const auto expected = evaluate(gh::AmrCriteria::SpecPRefinement{1.e-4}, mesh,
                                 filter, spacetime_metric, pi, phi);
  // The slowly decaying direction is refined, the quickly decaying one
  // coarsened.
  CHECK(expected == std::array{amr::Flag::IncreaseResolution,
                               amr::Flag::DoNothing,
                               amr::Flag::DecreaseResolution});
  CHECK(evaluate(*created, mesh, filter, spacetime_metric, pi, phi) ==
        expected);
  CHECK(evaluate(*copied, mesh, filter, spacetime_metric, pi, phi) == expected);
}

void test_decisions() {
  const Mesh<3> mesh{{{8, 9, 10}},
                     Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const tnsr::aa<DataVector, 3> zero_aa{number_of_grid_points, 0.0};
  const tnsr::iaa<DataVector, 3> zero_iaa{number_of_grid_points, 0.0};
  const Filters::None<3, FilteredTags> no_filter{std::nullopt};
  const auto no_filtered_modes = make_array<3>(0_st);

  const std::array<ModalVector, 3> smooth_coefficients{
      {power_law(8, 0.3), power_law(9, 0.5), power_law(10, 0.7)}};
  const DataVector smooth = sum_of_profiles(mesh, smooth_coefficients);
  const DataVector rough = sum_of_profiles(
      mesh, {{power_law(8, 0.15), power_law(9, 0.2), power_law(10, 0.25)}});
  const auto smooth_power =
      PowerMonitors::directional_power_monitors(smooth, mesh);
  const auto rough_power =
      PowerMonitors::directional_power_monitors(rough, mesh);
  const auto smooth_errors = errors_of(smooth_power, mesh, no_filtered_modes);
  const auto rough_errors = errors_of(rough_power, mesh, no_filtered_modes);

  {
    INFO("One component of the metric");
    auto spacetime_metric = zero_aa;
    get<0, 0>(spacetime_metric) = smooth;
    check_thresholds(mesh, no_filter, spacetime_metric, zero_aa, zero_iaa,
                     smooth_errors);
  }
  {
    INFO("The group with the larger error decides");
    auto spacetime_metric = zero_aa;
    get<0, 0>(spacetime_metric) = smooth;
    auto phi = zero_iaa;
    get<1, 0, 2>(phi) = rough;
    Errors larger_errors{};
    for (size_t d = 0; d < 3; ++d) {
      for (size_t i = 0; i < 2; ++i) {
        // Make sure that this case differs from the previous one.
        CHECK(gsl::at(gsl::at(rough_errors, d), i) >
              2.0 * gsl::at(gsl::at(smooth_errors, d), i));
        gsl::at(gsl::at(larger_errors, d), i) =
            std::max(gsl::at(gsl::at(smooth_errors, d), i),
                     gsl::at(gsl::at(rough_errors, d), i));
      }
    }
    check_thresholds(mesh, no_filter, spacetime_metric, zero_aa, phi,
                     larger_errors);
  }
  {
    INFO("Pi and Phi are one group");
    auto pi = zero_aa;
    get<1, 2>(pi) = smooth;
    auto phi = zero_iaa;
    get<1, 0, 2>(phi) = rough;
    std::array<DataVector, 3> combined_power{};
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(combined_power, d) = sqrt(square(gsl::at(smooth_power, d)) +
                                        square(gsl::at(rough_power, d)));
    }
    check_thresholds(mesh, no_filter, zero_aa, pi, phi,
                     errors_of(combined_power, mesh, no_filtered_modes));
  }
  {
    INFO("Fields that vanish are left alone");
    for (const double target : {0.0, 1.e-4, 1.e10}) {
      CHECK(evaluate(gh::AmrCriteria::SpecPRefinement{target}, mesh, no_filter,
                     zero_aa, zero_aa,
                     zero_iaa) == make_array<3>(amr::Flag::DoNothing));
    }
  }
  {
    INFO("Small meshes");
    const Mesh<3> small_mesh{3, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
    tnsr::aa<DataVector, 3> spacetime_metric{small_mesh.number_of_grid_points(),
                                             0.0};
    get<0, 0>(spacetime_metric) =
        sum_of_profiles(small_mesh, make_array<3>(power_law(3, 1.0)));
    const tnsr::aa<DataVector, 3> pi{small_mesh.number_of_grid_points(), 0.0};
    const tnsr::iaa<DataVector, 3> phi{small_mesh.number_of_grid_points(), 0.0};
    // Three unfiltered modes can be coarsened to two.
    CHECK(evaluate(gh::AmrCriteria::SpecPRefinement{1.e10}, small_mesh,
                   no_filter, spacetime_metric, pi,
                   phi) == make_array<3>(amr::Flag::DecreaseResolution));
    // The filter removes the highest of the three modes, and two unfiltered
    // modes cannot be coarsened.
    const auto filter = hypercube_filter(true, true, std::nullopt);
    CHECK(evaluate(gh::AmrCriteria::SpecPRefinement{1.e10}, small_mesh, filter,
                   spacetime_metric, pi,
                   phi) == make_array<3>(amr::Flag::DoNothing));
    // One unfiltered mode is not enough.
    const Mesh<3> smallest_mesh{2, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
    CHECK_THROWS_WITH(
        evaluate(gh::AmrCriteria::SpecPRefinement{1.e10}, smallest_mesh, filter,
                 tnsr::aa<DataVector, 3>{8_st, 1.0},
                 tnsr::aa<DataVector, 3>{8_st, 0.0},
                 tnsr::iaa<DataVector, 3>{8_st, 0.0}),
        Catch::Matchers::ContainsSubstring(
            "at least two unfiltered modes in every direction"));
  }
}

// The number of modes that the exponential filter removes, from its modal
// attenuation.
size_t expected_number_of_filtered_modes(const size_t number_of_modes) {
  double removed = 0.0;
  for (size_t k = 0; k < number_of_modes; ++k) {
    removed +=
        1.0 - exp(-36.0 * pow(static_cast<double>(k) /
                                  static_cast<double>(number_of_modes - 1),
                              2.0 * half_power));
  }
  return static_cast<size_t>(std::round(removed));
}

void test_filtered_modes(const Spectral::Basis basis) {
  CAPTURE(basis);
  const Mesh<3> mesh{{{5, 8, 12}}, basis, Spectral::Quadrature::GaussLobatto};
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const tnsr::aa<DataVector, 3> zero_aa{number_of_grid_points, 0.0};
  const tnsr::iaa<DataVector, 3> zero_iaa{number_of_grid_points, 0.0};

  std::array<size_t, 3> filtered_modes{};
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(filtered_modes, d) =
        expected_number_of_filtered_modes(mesh.extents(d));
  }
  // Make sure that the number of filtered modes depends on the resolution.
  CHECK(filtered_modes == std::array{1_st, 2_st, 3_st});

  // Put large coefficients into the filtered modes of each direction.
  std::array<ModalVector, 3> coefficients{
      {power_law(5, 0.8), power_law(8, 0.5), power_law(12, 0.7)}};
  for (size_t d = 0; d < 3; ++d) {
    auto& coefficients_d = gsl::at(coefficients, d);
    for (size_t k = coefficients_d.size() - gsl::at(filtered_modes, d);
         k < coefficients_d.size(); ++k) {
      coefficients_d[k] = 1.0;
    }
  }
  auto spacetime_metric = zero_aa;
  get<0, 0>(spacetime_metric) = sum_of_profiles(mesh, coefficients);
  const auto power = PowerMonitors::directional_power_monitors(
      get<0, 0>(spacetime_metric), mesh);
  const auto errors_without_filtered_modes =
      errors_of(power, mesh, filtered_modes);
  const auto errors_with_filtered_modes =
      errors_of(power, mesh, make_array<3>(0_st));
  for (size_t d = 0; d < 3; ++d) {
    // Make sure that the filtered modes matter.
    CHECK(gsl::at(errors_with_filtered_modes, d)[0] >
          2.0 * gsl::at(errors_without_filtered_modes, d)[0]);
  }

  {
    INFO("Filter applied on every substep");
    check_thresholds(mesh, hypercube_filter(true, true, std::nullopt),
                     spacetime_metric, zero_aa, zero_iaa,
                     errors_without_filtered_modes);
  }
  {
    INFO("Filter applied on some steps");
    check_thresholds(mesh, hypercube_filter(true, false, 3_st),
                     spacetime_metric, zero_aa, zero_iaa,
                     errors_without_filtered_modes);
  }
  {
    INFO("Disabled filter");
    check_thresholds(mesh, hypercube_filter(false, true, 3_st),
                     spacetime_metric, zero_aa, zero_iaa,
                     errors_with_filtered_modes);
  }
  {
    INFO("Filter never applied in the volume");
    check_thresholds(mesh, hypercube_filter(true, false, std::nullopt),
                     spacetime_metric, zero_aa, zero_iaa,
                     errors_with_filtered_modes);
  }
  {
    INFO("No filter");
    check_thresholds(mesh, Filters::None<3, FilteredTags>{std::nullopt},
                     spacetime_metric, zero_aa, zero_iaa,
                     errors_with_filtered_modes);
  }
  {
    INFO("Filter that needs Jacobians");
    const Filters::SphericalShell<FilteredTags> filter{
        4,    std::nullopt, 64_st,        true,        std::nullopt,
        true, false,        std::nullopt, std::nullopt};
    CHECK_THROWS_WITH(evaluate(gh::AmrCriteria::SpecPRefinement{1.e-4}, mesh,
                               filter, spacetime_metric, zero_aa, zero_iaa),
                      Catch::Matchers::ContainsSubstring(
                          "because the filter needs Jacobians"));
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.AmrCriteria.SpecPRefinement",
    "[Unit][Evolution]") {
  test_creation_and_serialization();
  test_decisions();
  test_filtered_modes(Spectral::Basis::Legendre);
  test_filtered_modes(Spectral::Basis::Chebyshev);
}
