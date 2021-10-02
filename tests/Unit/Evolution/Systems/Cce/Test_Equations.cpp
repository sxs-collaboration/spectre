// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/Cce/Equations.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Cce::Tags::BondiBeta
// IWYU pragma: no_forward_declare Cce::Tags::H
// IWYU pragma: no_forward_declare Cce::Tags::Q
// IWYU pragma: no_forward_declare Cce::Tags::U
// IWYU pragma: no_forward_declare Cce::Tags::W
// IWYU pragma: no_forward_declare Cce::Tags::Integrand
// IWYU pragma: no_forward_declare Cce::Tags::LinearFactor
// IWYU pragma: no_forward_declare Cce::Tags::LinearFactorForConjugate
// IWYU pragma: no_forward_declare Cce::Tags::PoleOfIntegrand
// IWYU pragma: no_forward_declare Cce::Tags::RegularIntegrand
// IWYU pragma: no_forward_declare Cce::TagsCategory::IndependentOfBondiIntegration
// IWYU pragma: no_forward_declare Cce::TagsCategory::SwshDerivatives
// IWYU pragma: no_forward_declare Cce::TagsCategory::TagsToSwshDifferentiate
// IWYU pragma: no_forward_declare Cce::TagsCategory::Temporary
// IWYU pragma: no_forward_declare Cce::ComputeBondiIntegrand
// IWYU pragma: no_forward_declare Variables

namespace Cce {

namespace {

template <typename Type, size_t N>
using ForwardIgnoreIndex = Type;

// This is a wrapper struct such that the evaluate function can be called with
// the full set of arguments that would ordinarily be packed into the
// DataBox. This is necessary for constructing a function that can be compared
// to the pypp versions.
template <typename Mutator, typename OutputTag, typename ArgumentTagList,
          size_t NumberOfGridPoints, typename... Args>
struct MutationFromArguments;

template <typename Mutator, typename OutputTag, typename... ArgumentTags,
          size_t NumberOfGridPoints, typename... Args>
struct MutationFromArguments<Mutator, OutputTag, tmpl::list<ArgumentTags...>,
                             NumberOfGridPoints, Args...> {
  static ComplexDataVector evaluate(const Args&... args) {
    auto box = db::create<db::AddSimpleTags<OutputTag, ArgumentTags...>>(
        typename OutputTag::type{typename OutputTag::type::type{
            ComplexDataVector{NumberOfGridPoints}}},
        typename ArgumentTags::type{
            typename ArgumentTags::type::type{args}}...);
    db::mutate_apply<Mutator>(make_not_null(&box));
    return ComplexDataVector{get(db::get<OutputTag>(box)).data()};
  }
};

// Creates a proxy struct which acts like the function being tested, but with
// arguments not packed into a DataBox, then compares the result of that
// function to the result of a python function
template <typename Mutator, typename ArgumentTagList, typename OutputTag,
          size_t NumberOfGridPoints, typename DataType, size_t... ScalarIndices>
void forward_to_pypp_with(std::index_sequence<ScalarIndices...> /*meta*/,
                          const std::string& python_function) {
  using Evaluatable =
      MutationFromArguments<Mutator, OutputTag, ArgumentTagList,
                            NumberOfGridPoints,
                            ForwardIgnoreIndex<DataType, ScalarIndices>...>;

  pypp::check_with_random_values<1>(
      &(Evaluatable::evaluate), "Equations", python_function, {{{-2.0, 2.0}}},
      DataType{NumberOfGridPoints});
}

template <typename Tag>
using all_arguments_for_integrand =
    tmpl::append<typename ComputeBondiIntegrand<Tag>::temporary_tags,
                 typename ComputeBondiIntegrand<Tag>::argument_tags>;

template <typename bondi_integrand_tag>
struct python_function_for_bondi_integrand;

template <>
struct python_function_for_bondi_integrand<Tags::Integrand<Tags::BondiBeta>> {
  static std::string name() { return "integrand_for_beta"; }
};
template <>
struct python_function_for_bondi_integrand<
    Tags::PoleOfIntegrand<Tags::BondiQ>> {
  static std::string name() { return "integrand_for_q_pole_part"; }
};
template <>
struct python_function_for_bondi_integrand<
    Tags::RegularIntegrand<Tags::BondiQ>> {
  static std::string name() { return "integrand_for_q_regular_part"; }
};
template <>
struct python_function_for_bondi_integrand<Tags::Integrand<Tags::BondiU>> {
  static std::string name() { return "integrand_for_u"; }
};
template <>
struct python_function_for_bondi_integrand<
    Tags::PoleOfIntegrand<Tags::BondiW>> {
  static std::string name() { return "integrand_for_w_pole_part"; }
};
template <>
struct python_function_for_bondi_integrand<
    Tags::RegularIntegrand<Tags::BondiW>> {
  static std::string name() { return "integrand_for_w_regular_part"; }
};
template <>
struct python_function_for_bondi_integrand<
    Tags::PoleOfIntegrand<Tags::BondiH>> {
  static std::string name() { return "integrand_for_h_pole_part"; }
};
template <>
struct python_function_for_bondi_integrand<
    Tags::RegularIntegrand<Tags::BondiH>> {
  static std::string name() { return "integrand_for_h_regular_part"; }
};
template <>
struct python_function_for_bondi_integrand<Tags::LinearFactor<Tags::BondiH>> {
  static std::string name() { return "linear_factor_for_h"; }
};
template <>
struct python_function_for_bondi_integrand<
    Tags::LinearFactorForConjugate<Tags::BondiH>> {
  static std::string name() { return "linear_factor_for_conjugate_h"; }
};

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Equations", "[Unit][Cce]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Evolution/Systems/Cce/"};

  using all_bondi_tags = tmpl::list<Tags::BondiBeta, Tags::BondiQ, Tags::BondiU,
                                    Tags::BondiW, Tags::BondiH>;
  tmpl::for_each<all_bondi_tags>([](auto x) {
    using bondi_tag = typename decltype(x)::type;
    tmpl::for_each<integrand_terms_to_compute_for_bondi_variable<bondi_tag>>(
        [](auto y) {
          using bondi_integrand_tag = typename decltype(y)::type;
          using bondi_integrand_argument_list =
              all_arguments_for_integrand<bondi_integrand_tag>;
          // We check the equations with the DataBox interface, as it is easy
          // to generate the box with the appropriate type lists,
          // which are then used to create wrappers that can be compared to Pypp
          // equations without explicitly enumerating the long argument lists in
          // this test.
          forward_to_pypp_with<ComputeBondiIntegrand<bondi_integrand_tag>,
                               bondi_integrand_argument_list,
                               bondi_integrand_tag, 5, ComplexDataVector>(
              std::make_index_sequence<
                  tmpl::size<bondi_integrand_argument_list>::value>{},
              python_function_for_bondi_integrand<bondi_integrand_tag>::name());
        });
  });
}
}  // namespace
}  // namespace Cce
