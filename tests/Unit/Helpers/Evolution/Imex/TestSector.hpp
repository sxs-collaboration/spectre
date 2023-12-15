// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/ContractFirstNIndices.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Evolution/Imex/Protocols/ImplicitSector.hpp"
#include "Evolution/Imex/Tags/Jacobian.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace TestHelpers::imex {
namespace test_sector_detail {
template <typename Tag>
struct ReadOnlySource : ::db::SimpleTag {
  using type = typename Tag::type;
};

template <typename Tag>
struct ReadOnly : Tag, ::db::ComputeTag {
  using base = Tag;
  using argument_tags = tmpl::list<ReadOnlySource<Tag>>;
  static void function(const gsl::not_null<typename ReadOnly::type*> dest,
                       const typename ReadOnly::type& source) {
    *dest = source;
  }
};

template <typename T>
struct OnlyEntry;

template <typename T>
struct OnlyEntry<tmpl::list<T>> {
  using type = T;
};
}  // namespace test_sector_detail

/// Check that an implicit sector conforms to
/// ::imex::protocols::ImplicitSector and that its `source` and
/// `jacobian` are consistent with each other.
template <typename Sector, typename SolveAttempt = typename test_sector_detail::
                               OnlyEntry<typename Sector::solve_attempts>::type>
void test_sector(const double stencil_size, const double tolerance,
                 const Variables<typename Sector::tensors>& explicit_values,
                 tuples::tagged_tuple_from_typelist<
                     typename SolveAttempt::tags_from_evolution>
                     evolution_data) {
  static_assert(
      tt::assert_conforms_to_v<Sector, ::imex::protocols::ImplicitSector>);
  static_assert(
      tmpl::list_contains_v<typename Sector::solve_attempts, SolveAttempt>);

  using sector_variables_tag = ::Tags::Variables<typename Sector::tensors>;
  using SectorVariables = typename sector_variables_tag::type;
  using sector_source_tag =
      ::db::add_tag_prefix<::Tags::Source, sector_variables_tag>;

  using sector_jacobian_tag = ::Tags::Variables<::imex::jacobian_tags<
      typename Sector::tensors, typename sector_source_tag::type::tags_list>>;

  // Make tags that are immutable in real use immutable here too.
  using read_only_tags_from_evolution_source =
      tmpl::transform<typename SolveAttempt::tags_from_evolution,
                      tmpl::bind<test_sector_detail::ReadOnlySource, tmpl::_1>>;
  using read_only_tags_from_evolution =
      tmpl::transform<typename SolveAttempt::tags_from_evolution,
                      tmpl::bind<test_sector_detail::ReadOnly, tmpl::_1>>;

  using simple_tags = tmpl::append<
      read_only_tags_from_evolution_source, typename SolveAttempt::simple_tags,
      tmpl::list<sector_variables_tag, sector_source_tag, sector_jacobian_tag>>;
  using compute_tags = tmpl::append<read_only_tags_from_evolution,
                                    typename SolveAttempt::compute_tags>;
  auto box = ::db::create<simple_tags, compute_tags>();
  tmpl::for_each<typename SolveAttempt::tags_from_evolution>([&](auto tag) {
    using Tag = tmpl::type_from<decltype(tag)>;
    ::db::mutate<test_sector_detail::ReadOnlySource<Tag>>(
        [&](const gsl::not_null<typename Tag::type*> box_value) {
          *box_value = std::move(get<Tag>(evolution_data));
        },
        make_not_null(&box));
  });
  using variables_tags = tmpl::push_front<
      tmpl::filter<typename SolveAttempt::simple_tags,
                   tt::is_a<Variables, tmpl::bind<tmpl::type_from, tmpl::_1>>>,
      sector_source_tag, sector_jacobian_tag>;
  ::db::mutate_apply<variables_tags, tmpl::list<>>(
      [](const auto... vars) { expand_pack((vars->initialize(1, 0.0), 0)...); },
      make_not_null(&box));
  ::db::mutate<sector_variables_tag>(
      [&](const gsl::not_null<SectorVariables*> vars) {
        *vars = explicit_values;
      },
      make_not_null(&box));

  // Arbitrary values.  These would depend on the choice of time
  // stepper and step size and such.
  const SectorVariables inhomogeneous_terms = 1.1 * explicit_values;
  const double implicit_weight = 0.4;

  const std::vector<::imex::GuessResult> guess_type =
      ::db::mutate_apply<typename Sector::initial_guess>(
          make_not_null(&box), inhomogeneous_terms, implicit_weight);
  CHECK(guess_type.size() <= 1);
  const SectorVariables initial_guess = ::db::get<sector_variables_tag>(box);
  if (guess_type == std::vector{::imex::GuessResult::ExactSolution}) {
    // If the sector can always be solved analytically, the jacobian
    // need not be coded.  Instead, we check that the supplied
    // solution is correct.
    tmpl::for_each<typename SolveAttempt::source_prep>([&](auto mutator) {
      using Mutator = tmpl::type_from<decltype(mutator)>;
      ::db::mutate_apply<Mutator>(make_not_null(&box));
    });
    ::db::mutate_apply<typename SolveAttempt::source>(make_not_null(&box));
    const auto& source = ::db::get<sector_source_tag>(box);
    const SectorVariables step_result =
        inhomogeneous_terms + implicit_weight * source;
    CHECK_VARIABLES_CUSTOM_APPROX(initial_guess, step_result,
                                  Approx::custom().epsilon(tolerance));
  } else {
    // Check the jacobian numerically.
    tmpl::for_each<typename SolveAttempt::jacobian_prep>([&](auto mutator) {
      using Mutator = tmpl::type_from<decltype(mutator)>;
      ::db::mutate_apply<Mutator>(make_not_null(&box));
    });
    ::db::mutate_apply<typename SolveAttempt::jacobian>(make_not_null(&box));
    const auto jacobian = ::db::get<sector_jacobian_tag>(box);
    tmpl::for_each<typename Sector::tensors>([&](auto varying_tensor_v) {
      using varying_tensor = tmpl::type_from<decltype(varying_tensor_v)>;
      CAPTURE(pretty_type::get_name<varying_tensor>());
      for (size_t varying_component = 0;
           varying_component < varying_tensor::type::size();
           ++varying_component) {
        ::db::mutate<sector_variables_tag>(
            [&](const gsl::not_null<SectorVariables*> vars) {
              *vars = initial_guess;
            },
            make_not_null(&box));
        const auto finite_difference_derivative = numerical_derivative(
            [&](const std::array<double, 1>& component_value) {
              ::db::mutate<varying_tensor>(
                  [&](const gsl::not_null<typename varying_tensor::type*> var) {
                    (*var)[varying_component] = component_value[0];
                  },
                  make_not_null(&box));
              tmpl::for_each<typename SolveAttempt::source_prep>(
                  [&](auto mutator) {
                    using Mutator = tmpl::type_from<decltype(mutator)>;
                    ::db::mutate_apply<Mutator>(make_not_null(&box));
                  });
              ::db::mutate_apply<typename SolveAttempt::source>(
                  make_not_null(&box));
              return ::db::get<sector_source_tag>(box);
            },
            std::array{
                get<varying_tensor>(initial_guess)[varying_component][0]},
            0, stencil_size);
        typename varying_tensor::type variation(1_st, 0.0);
        variation[varying_component] = 1.0;
        CAPTURE(variation);
        tmpl::for_each<typename sector_source_tag::type::tags_list>(
            [&](auto source_tensor_v) {
              using source_tensor = tmpl::type_from<decltype(source_tensor_v)>;
              CAPTURE(pretty_type::get_name<source_tensor>());
              auto analytic_derivative = contract_first_n_indices<
                  varying_tensor::type::num_tensor_indices>(
                  variation,
                  get<::imex::Tags::Jacobian<varying_tensor, source_tensor>>(
                      jacobian));
              // We want the derivative with respect to a single
              // component, but with multiplicities all the equivalent
              // components were actually set, so the result is larger
              // than it should be.
              for (auto& component : analytic_derivative) {
                component /= variation.multiplicity(varying_component);
              }
              static_assert(
                  std::is_same_v<std::decay_t<decltype(analytic_derivative)>,
                                 typename source_tensor::type>);
              CHECK_ITERABLE_CUSTOM_APPROX(
                  analytic_derivative,
                  get<source_tensor>(finite_difference_derivative),
                  Approx::custom().epsilon(tolerance));
            });
      }
    });
  }
}
}  // namespace TestHelpers::imex
