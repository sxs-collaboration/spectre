// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "ImplicitSource.hpp"
#include "ImplicitSourceJacobian.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
namespace Tags {
template <typename Tag>
struct Source;
}  // namespace Tags
/// \endcond

namespace imex::protocols {
/// Protocol for an implicit sector of an IMEX system.
///
/// An implicit sector describes the sources for one implicit solve
/// performed during IMEX evolution.  A system may have multiple
/// implicit sectors, but they must be independent, i.e., their
/// sources must not depend on any tensors in other sectors.
///
/// Classes implementing this protocol must define:
///
/// * a `tensors` type alias of tags for the variables to be solved for
///
/// * an `initial_guess` type to be passed to `db::mutate_apply`,
///   taking additional arguments for the inhomogeneous terms \f$X\f$
///   and implicit weight \f$w\f$ in the equation to be solved:
///   \f$u = X + w S(u)\f$.  (See example below.)  It must return a
///   `std::vector<GuessResult>` indicating, for each point, whether
///   the implicit equation has been solved analytically or whether
///   the numerical solve should continue.  An empty return is
///   equivalent to `imex::GuessResult::InitialGuess` for every point,
///   so numerical solves will be performed for each.  When this is
///   called, the sector variables will have the value from the
///   explicit part of the time step.  This mutator will not be called
///   if the implicit weight is zero, as the solution is trivial in
///   that case.  If using the value of the explicit step as an
///   initial guess is acceptable, this can be the type
///   `imex::GuessExplicitResult`.
///
/// * a `solve_attempts` list of sources that will be attempted to be
///   solved, in order.  The first one that succeeds at each point
///   will be used.  Pieces of code that need "the" source for the
///   sector will use the source from the first entry.  Each attempt
///   must be a struct with the following types:
///
///   * lists used to construct a DataBox during the pointwise
///     implicit solve:
///
///     * `tags_from_evolution` for tags in addition to the sector
///        tensors to be made available from the evolution DataBox.
///        Volume quantities will be reduced to have one grid point,
///        with the appropriate value for the point being solved for.
///
///     * `simple_tags` for temporaries (e.g., primitives)
///
///     * `compute_tags`
///
///   * a `source` type to be passed to `db::mutate_apply` to compute
///     the sources. It must conform to imex::protocols::ImplicitSource.
///
///   * a `jacobian` type to be passed to `db::mutate_apply` to
///     compute the source jacobian.  It must conform to
///     imex::protocols::ImplicitSourceJacobian. If the implicit equation can
///     always be solved analytically for the sector, the jacobian is not
///     required and this may be the type
///     `imex::NoJacobianBecauseSolutionIsAnalytic`.
///
///   * lists `source_prep` and `jacobian_prep` that will be called
///     before the corresponding main mutator, e.g., for computing
///     primitives.  Mutators appearing in multiple lists, as well as
///     the `source` and `jacobian` mutators, will be skipped if they
///     have already been applied for the current point.  Note that
///     the `source_prep` mutators are only used during the implicit
///     solve, and any preparation needed before the `source` call in
///     the main action loop to record the history is the
///     responsibility of the user.
///
/// All `Variables` in the DataBox, including the sources and source
/// jacobian, will be initialized to zero with a single grid point.
///
/// \snippet DoImplicitStepSector.hpp simple_sector
///
/// Examples of definitions of a complicated implicit source and
/// jacobian:
///
/// \snippet Test_SolveImplicitSector.cpp source
/// \snippet Test_SolveImplicitSector.cpp Jacobian
struct ImplicitSector {
  template <typename ConformingType>
  struct test {
    using tensors = typename ConformingType::tensors;
    using initial_guess = typename ConformingType::initial_guess;
    using solve_attempts = typename ConformingType::solve_attempts;

    static_assert(tt::is_a_v<tmpl::list, tensors>);
    static_assert(tt::is_a_v<tmpl::list, solve_attempts>);
    static_assert(tmpl::size<solve_attempts>::value >= 1);

    static_assert(
        tmpl::all<
            tensors,
            tt::is_a<Tensor, tmpl::bind<tmpl::type_from, tmpl::_1>>>::value);

    using source_tensors =
        tmpl::transform<tensors, tmpl::bind<::Tags::Source, tmpl::_1>>;

    static_assert(
        tmpl::size<tmpl::list_difference<typename initial_guess::return_tags,
                                         tensors>>::value == 0,
        "initial_guess can only modify sector variables.");

    template <typename SolveAttempt>
    struct test_solve_attempt {
      using source = typename SolveAttempt::source;
      using jacobian = typename SolveAttempt::jacobian;

      using tags_from_evolution = typename SolveAttempt::tags_from_evolution;
      using simple_tags = typename SolveAttempt::simple_tags;
      using compute_tags = typename SolveAttempt::compute_tags;

      using source_prep = typename SolveAttempt::source_prep;
      using jacobian_prep = typename SolveAttempt::jacobian_prep;

      // check protocol conformity
      static_assert(
          tt::assert_conforms_to_v<source, imex::protocols::ImplicitSource>);
      static_assert(tt::assert_conforms_to_v<
                    jacobian, imex::protocols::ImplicitSourceJacobian>);

      static_assert(tt::is_a_v<tmpl::list, tags_from_evolution>);
      static_assert(tt::is_a_v<tmpl::list, simple_tags>);
      static_assert(tt::is_a_v<tmpl::list, compute_tags>);
      static_assert(tt::is_a_v<tmpl::list, source_prep>);
      static_assert(tt::is_a_v<tmpl::list, jacobian_prep>);

      static_assert(
          std::is_same_v<tmpl::list_difference<tags_from_evolution, tensors>,
                         tags_from_evolution>,
          "tags_from_evolution cannot include the sector tensors.");

      static_assert(
          std::is_same_v<tmpl::list_difference<source_tensors,
                                               typename source::return_tags>,
                         tmpl::list<>> and
              std::is_same_v<tmpl::list_difference<typename source::return_tags,
                                                   source_tensors>,
                             tmpl::list<>>,
          "Implicit source must provide sources for the entire sector.");

      template <typename T>
      struct is_a_tensor_of_data_vector : std::false_type {};

      template <typename Symm, typename IndexList>
      struct is_a_tensor_of_data_vector<Tensor<DataVector, Symm, IndexList>>
          : std::true_type {};

      static_assert(
          tmpl::none<simple_tags, is_a_tensor_of_data_vector<tmpl::bind<
                                      tmpl::type_from, tmpl::_1>>>::value,
          "Do not include tags for Tensor<DataVector> in simple_tags, because "
          "they trigger many memory allocations.  Add the tensors as part of "
          "a Variables instead.");

      using type = std::true_type;
    };

    static_assert(
        tmpl::all<solve_attempts, test_solve_attempt<tmpl::_1>>::value);
  };
};
}  // namespace imex::protocols
