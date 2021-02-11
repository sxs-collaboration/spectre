// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Helper functions to test elliptic first-order systems

#pragma once

#include <cstddef>
#include <limits>
#include <random>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Framework/TestHelpers.hpp"
#include "Framework/TestingFramework.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace TestHelpers {
/// \ingroup TestingFrameworkGroup
/// Helper functions to test elliptic first-order systems
namespace elliptic {
namespace detail {

template <typename FluxesComputer, typename... PrimalFields,
          typename... AuxiliaryFields, typename... PrimalFluxes,
          typename... AuxiliaryFluxes, typename... FluxesArgsTags>
void test_first_order_fluxes_computer_impl(
    const DataVector& used_for_size, tmpl::list<PrimalFields...> /*meta*/,
    tmpl::list<AuxiliaryFields...> /*meta*/,
    tmpl::list<PrimalFluxes...> /*meta*/,
    tmpl::list<AuxiliaryFluxes...> /*meta*/,
    tmpl::list<FluxesArgsTags...> /*meta*/) {
  using vars_tag =
      ::Tags::Variables<tmpl::list<PrimalFields..., AuxiliaryFields...>>;
  using VarsType = typename vars_tag::type;
  using fluxes_tag =
      ::Tags::Variables<tmpl::list<PrimalFluxes..., AuxiliaryFluxes...>>;
  using FluxesType = typename fluxes_tag::type;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(0.5, 2.);

  // Generate random variables
  const auto vars = make_with_random_values<VarsType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);

  // Generate fluxes from the variables with random arguments
  tuples::TaggedTuple<FluxesArgsTags...> fluxes_args{
      make_with_random_values<typename FluxesArgsTags::type>(
          make_not_null(&generator), make_not_null(&dist), used_for_size)...};
  // Silence unused variable warning when fluxes args is empty
  (void)fluxes_args;
  FluxesType expected_fluxes{used_for_size.size()};
  FluxesComputer::apply(make_not_null(&get<PrimalFluxes>(expected_fluxes))...,
                        get<FluxesArgsTags>(fluxes_args)...,
                        get<AuxiliaryFields>(vars)...);
  FluxesComputer::apply(
      make_not_null(&get<AuxiliaryFluxes>(expected_fluxes))...,
      get<FluxesArgsTags>(fluxes_args)..., get<PrimalFields>(vars)...);

  // Create a DataBox
  auto box =
      db::create<db::AddSimpleTags<vars_tag, fluxes_tag, FluxesArgsTags...>>(
          vars,
          make_with_value<typename fluxes_tag::type>(
              used_for_size, std::numeric_limits<double>::signaling_NaN()),
          get<FluxesArgsTags>(fluxes_args)...);

  // Apply the fluxes computer to the DataBox
  db::mutate_apply<tmpl::list<PrimalFluxes...>,
                   tmpl::list<FluxesArgsTags..., AuxiliaryFields...>>(
      FluxesComputer{}, make_not_null(&box));
  db::mutate_apply<tmpl::list<AuxiliaryFluxes...>,
                   tmpl::list<FluxesArgsTags..., PrimalFields...>>(
      FluxesComputer{}, make_not_null(&box));
  CHECK(expected_fluxes == get<fluxes_tag>(box));
}

}  // namespace detail

/*!
 * \brief Test the `System::fluxes_computer` is functional
 *
 * This function tests the following properties of the
 * `System::fluxes_computer`:
 *
 * - It works with the fields and fluxes specified in the `System`.
 * - It can be applied to a DataBox, i.e. its argument tags are consistent with
 *   its apply function.
 */
template <typename System>
void test_first_order_fluxes_computer(const DataVector& used_for_size) {
  detail::test_first_order_fluxes_computer_impl<
      typename System::fluxes_computer>(
      used_for_size, typename System::primal_fields{},
      typename System::auxiliary_fields{}, typename System::primal_fluxes{},
      typename System::auxiliary_fluxes{},
      typename System::fluxes_computer::argument_tags{});
}

namespace detail {

template <typename SourcesComputer, typename... PrimalFields,
          typename... AuxiliaryFields, typename... PrimalFluxes,
          typename... SourcesArgsTags>
void test_first_order_sources_computer_impl(
    const DataVector& used_for_size, tmpl::list<PrimalFields...> /*meta*/,
    tmpl::list<AuxiliaryFields...> /*meta*/,
    tmpl::list<PrimalFluxes...> /*meta*/,
    tmpl::list<SourcesArgsTags...> /*meta*/) {
  using vars_tag =
      ::Tags::Variables<tmpl::list<PrimalFields..., AuxiliaryFields...>>;
  using VarsType = typename vars_tag::type;
  using fluxes_tag = ::Tags::Variables<tmpl::list<PrimalFluxes...>>;
  using FluxesType = typename fluxes_tag::type;
  using sources_tag = db::add_tag_prefix<::Tags::Source, vars_tag>;
  using SourcesType = typename sources_tag::type;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(0.5, 2.);

  // Generate random variables and fluxes. Note that the sources make use of the
  // pre-computed fluxes as an optimization (see elliptic::first_order_sources),
  // but for the purpose of this random-value test the fluxes don't have to be
  // computed from the variables.
  const auto vars = make_with_random_values<VarsType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto fluxes = make_with_random_values<FluxesType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);

  // Generate sources from the variables with random arguments
  tuples::TaggedTuple<SourcesArgsTags...> sources_args{
      make_with_random_values<typename SourcesArgsTags::type>(
          make_not_null(&generator), make_not_null(&dist), used_for_size)...};
  // Silence unused variable warning when sources args is empty
  (void)sources_args;
  SourcesType expected_sources{used_for_size.size(), 0.};
  SourcesComputer::apply(
      make_not_null(&get<::Tags::Source<AuxiliaryFields>>(expected_sources))...,
      get<SourcesArgsTags>(sources_args)..., get<PrimalFields>(vars)...);
  SourcesComputer::apply(
      make_not_null(&get<::Tags::Source<PrimalFields>>(expected_sources))...,
      get<SourcesArgsTags>(sources_args)..., get<PrimalFields>(vars)...,
      get<PrimalFluxes>(fluxes)...);

  // Create a DataBox
  auto box = db::create<
      db::AddSimpleTags<vars_tag, fluxes_tag, sources_tag, SourcesArgsTags...>>(
      vars, fluxes,
      make_with_value<typename sources_tag::type>(used_for_size, 0.),
      get<SourcesArgsTags>(sources_args)...);

  // Apply the sources computer to the DataBox
  db::mutate_apply<tmpl::list<::Tags::Source<AuxiliaryFields>...>,
                   tmpl::list<SourcesArgsTags..., PrimalFields...>>(
      SourcesComputer{}, make_not_null(&box));
  db::mutate_apply<
      tmpl::list<::Tags::Source<PrimalFields>...>,
      tmpl::list<SourcesArgsTags..., PrimalFields..., PrimalFluxes...>>(
      SourcesComputer{}, make_not_null(&box));
  CHECK(expected_sources == get<sources_tag>(box));
}

}  // namespace detail

/*!
 * \brief Test the `System::sources_computer` is functional
 *
 * This function tests the following properties of the
 * `System::sources_computer`:
 *
 * - It works with the fields and fluxes specified in the `System`.
 * - It can be applied to a DataBox, i.e. its argument tags are consistent with
 *   its apply function.
 */
template <typename System>
void test_first_order_sources_computer(const DataVector& used_for_size) {
  using sources_computer = typename System::sources_computer;
  detail::test_first_order_sources_computer_impl<sources_computer>(
      used_for_size, typename System::primal_fields{},
      typename System::auxiliary_fields{}, typename System::primal_fluxes{},
      typename sources_computer::argument_tags{});
}

}  // namespace elliptic
}  // namespace TestHelpers
