// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>
#include <string>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/SwshTransformJob.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare ComplexModalVector
// IWYU pragma: no_forward_declare SpinWeighted

namespace Spectral {
namespace Swsh {
namespace {

// for storing a computed spin-weighted value during the transform test
template <int Spin>
struct TestTag : db::SimpleTag {
  static std::string name() noexcept { return "TestTag"; }
  using type = Scalar<SpinWeighted<ComplexDataVector, Spin>>;
};

// for storing an expected spin-weighted value during the transform test
template <int Spin>
struct ExpectedTestTag : db::SimpleTag {
  static std::string name() noexcept { return "ExpectedTestTag"; }
  using type = Scalar<SpinWeighted<ComplexDataVector, Spin>>;
};

using TestDerivativeTagList =
    tmpl::list<Tags::Derivative<TestTag<-1>, Tags::Eth>,
               Tags::Derivative<TestTag<-1>, Tags::EthEthbar>,
               Tags::Derivative<ExpectedTestTag<-1>, Tags::EthEthbar>,
               Tags::Derivative<TestTag<2>, Tags::EthbarEthbar>>;

/// [make_swsh_transform_job_list]
using ExpectedInverseJobs = tmpl::list<
    TransformJob<
        -1, ComplexRepresentation::RealsThenImags,
        tmpl::list<Tags::Derivative<TestTag<-1>, Tags::EthEthbar>,
                   Tags::Derivative<ExpectedTestTag<-1>, Tags::EthEthbar>>>,
    TransformJob<0, ComplexRepresentation::RealsThenImags,
                 tmpl::list<Tags::Derivative<TestTag<-1>, Tags::Eth>,
                            Tags::Derivative<TestTag<2>, Tags::EthbarEthbar>>>>;

static_assert(
    cpp17::is_same_v<
        make_swsh_transform_job_list<ComplexRepresentation::RealsThenImags,
                                     TestDerivativeTagList>,
        ExpectedInverseJobs>,
    "failed testing make_swsh_transform_job_list");
/// [make_swsh_transform_job_list]

/// [make_swsh_transform_from_derivative_tags]
using ExpectedJobs =
    tmpl::list<TransformJob<-1, ComplexRepresentation::Interleaved,
                            tmpl::list<TestTag<-1>, ExpectedTestTag<-1>>>,
               TransformJob<2, ComplexRepresentation::Interleaved,
                            tmpl::list<TestTag<2>>>>;

static_assert(
    cpp17::is_same_v<
        make_swsh_transform_job_list_from_derivative_tags<
            ComplexRepresentation::Interleaved, TestDerivativeTagList>,
        ExpectedJobs>,
    "failed testing make_swsh_transform_job_list_from_derivative_tags");
/// [make_swsh_transform_from_derivative_tags]

static_assert(
    cpp17::is_same_v<
        typename tmpl::front<ExpectedInverseJobs>::CoefficientTagList,
        tmpl::list<
            Tags::SwshTransform<Tags::Derivative<TestTag<-1>, Tags::EthEthbar>>,
            Tags::SwshTransform<
                Tags::Derivative<ExpectedTestTag<-1>, Tags::EthEthbar>>>>,
    "failed testing TransformJob");

template <ComplexRepresentation Representation, int S>
void test_transform_and_inverse_transform() noexcept {
  // generate parameters for the points to transform
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{2, 7};
  const size_t l_max = sdist(gen);
  const size_t number_of_radial_points = 2;
  UniformCustomDistribution<double> coefficient_distribution{-10.0, 10.0};

  // create the variables for the transformations
  // The four data structures will be used as:
  // - `generated_modes`: randomly generated
  // - `test_collocation`: computed from analytic expressions using the
  //                       randomly generated coefficients in `generated_modes`
  // - `transformed_modes`: computed from test_collocation points from forward
  //                        transform
  // next, we test the equivalence of the coefficients in test and expected
  // for the inverse transform test, we use:
  // - `expected_collocation`: copied from `test_collocation`
  // - `test_collocation`: computed from an inverse transform of the
  //                       `transformed_modes`, overwriting the previous data
  // Finally, we test the equivalence of the `expected_collocation` and
  // `test_collocation`
  Variables<tmpl::list<ExpectedTestTag<S>, TestTag<S>>> collocation_data{
      number_of_radial_points * number_of_swsh_collocation_points(l_max), 0.0};
  Variables<tmpl::list<Tags::SwshTransform<ExpectedTestTag<S>>,
                       Tags::SwshTransform<TestTag<S>>>>
      coefficient_data{
          number_of_swsh_coefficients(l_max) * 2 * number_of_radial_points,
          0.0};

  // randomly generate the mode coefficients
  ComplexModalVector& generated_modes =
      get(get<Tags::SwshTransform<ExpectedTestTag<S>>>(coefficient_data))
          .data();
  TestHelpers::generate_swsh_modes<S>(
      make_not_null(&generated_modes), make_not_null(&gen),
      make_not_null(&coefficient_distribution), number_of_radial_points, l_max);

  // fill the expected collocation point data by evaluating the analytic
  // functions. This is very slow and rough (due to factorial division), but
  // comparatively simple to formulate.
  ComplexDataVector& test_collocation =
      get(get<TestTag<S>>(collocation_data)).data();
  TestHelpers::swsh_collocation_from_coefficients_and_basis_func<
      S, Representation>(&test_collocation, &generated_modes, l_max,
                         number_of_radial_points,
                         TestHelpers::spin_weighted_spherical_harmonic);
  // create the forward transformation job and execute
  using JobTags = tmpl::list<TestTag<S>>;
  const TransformJob<S, Representation, JobTags> job{l_max,
                                                     number_of_radial_points};
  const auto test_collocation_copy = test_collocation;
  job.execute_transform(make_not_null(&coefficient_data),
                        make_not_null(&collocation_data));

  // approximation needs to be a little loose to consistently accommodate the
  // ratios of factorials in the analytic form
  Approx transform_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);

  // verify that the collocation points haven't been altered by the
  // transformation
  CHECK(test_collocation_copy == test_collocation_copy);

  ComplexDataVector& expected_collocation =
      get(get<ExpectedTestTag<S>>(collocation_data)).data();
  expected_collocation = test_collocation;

  const ComplexModalVector& transformed_modes =
      get(get<Tags::SwshTransform<TestTag<S>>>(coefficient_data)).data();
  CHECK_ITERABLE_CUSTOM_APPROX(transformed_modes, generated_modes,
                               transform_approx);

  // create the inverse transformation job and execute
  using InverseJobTags = tmpl::list<TestTag<S>>;
  const TransformJob<S, Representation, InverseJobTags> inverse_job{
      l_max, number_of_radial_points};
  inverse_job.execute_inverse_transform(make_not_null(&collocation_data),
                                        make_not_null(&coefficient_data));

  CHECK_ITERABLE_CUSTOM_APPROX(test_collocation, expected_collocation,
                               transform_approx);
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.Transform",
                  "[Unit][NumericalAlgorithms]") {
  {
    INFO("Testing with ComplexRepresentation::Interleaved");
    test_transform_and_inverse_transform<ComplexRepresentation::Interleaved,
                                         -2>();
    test_transform_and_inverse_transform<ComplexRepresentation::Interleaved,
                                         0>();
    test_transform_and_inverse_transform<ComplexRepresentation::Interleaved,
                                         1>();
  }
  {
    INFO("Testing with ComplexRepresentation::RealsThenImags");
    test_transform_and_inverse_transform<ComplexRepresentation::RealsThenImags,
                                         -1>();
    test_transform_and_inverse_transform<ComplexRepresentation::RealsThenImags,
                                         0>();
    test_transform_and_inverse_transform<ComplexRepresentation::RealsThenImags,
                                         2>();
  }
}
}  // namespace
}  // namespace Swsh
}  // namespace Spectral
