// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Imex/SolveImplicitSector.hpp"

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ExtractPoint.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Evolution/Imex/Mode.hpp"
#include "Evolution/Imex/Protocols/ImplicitSector.hpp"
#include "Evolution/Imex/Tags/Jacobian.hpp"
#include "NumericalAlgorithms/LinearSolver/Lapack.hpp"
#include "NumericalAlgorithms/RootFinding/GslMultiRoot.hpp"
#include "Time/History.hpp"
#include "Time/TimeSteppers/ImexTimeStepper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/SplitTuple.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
class TimeDelta;
/// \endcond

namespace imex {
namespace solve_implicit_sector_detail {
// Calculates the inhomogeneous terms and the implicit weight for the
// equation to be solved (X and w in the sector documentation), and
// writes the initial guess into system_variables (which generally
// points into the evolution DateBox).
template <typename SectorVariables>
class ImplicitEquation {
 public:
  template <typename ImplicitSector, typename SystemVariables>
  ImplicitEquation(
      const gsl::not_null<SystemVariables*> system_variables,
      const ImexTimeStepper& time_stepper, const TimeDelta& time_step,
      const TimeSteppers::History<SectorVariables>& implicit_history,
      const ForwardTuple<typename ImplicitSector::initial_guess::argument_tags>&
          initial_guess_arguments,
      tmpl::type_<ImplicitSector> /*meta*/)
      : implicit_weight_(
            time_stepper.implicit_weight(implicit_history, time_step)) {
    static_assert(std::is_same_v<typename ImplicitSector::tensors,
                                 typename SectorVariables::tags_list>);

    if (implicit_weight_ == 0.0) {
      // Explicit substep.  No solves.  Just write the correct answer
      // into the evolution box.
      auto sector_subset = system_variables->template reference_subset<
          typename SectorVariables::tags_list>();
      time_stepper.add_inhomogeneous_implicit_terms(
          make_not_null(&sector_subset), implicit_history, time_step);
      return;
    }

    inhomogeneous_terms_ =
        system_variables
            ->template extract_subset<typename SectorVariables::tags_list>();
    time_stepper.add_inhomogeneous_implicit_terms(
        make_not_null(&inhomogeneous_terms_), implicit_history, time_step);

    {
      const auto initial_guess_return =
          tmpl::as_pack<typename ImplicitSector::initial_guess::return_tags>(
              [&](auto... tags) {
                return std::make_tuple(
                    make_not_null(&get<tmpl::type_from<decltype(tags)>>(
                        *system_variables))...);
              });
      initial_guess_types_ = std::apply(
          [&](const auto&... args) {
            return ImplicitSector::initial_guess::apply(
                args..., inhomogeneous_terms_, implicit_weight_);
          },
          std::tuple_cat(initial_guess_return, initial_guess_arguments));
    }

    ASSERT(initial_guess_types_.size() ==
                   system_variables->number_of_grid_points() or
               initial_guess_types_.empty(),
           "initial_guess must return one GuessResult per point or an empty "
           "vector for GuessResult::InitialGuess everywhere.");
  }

  double implicit_weight() const { return implicit_weight_; }
  const SectorVariables& inhomogeneous_terms() const {
    return inhomogeneous_terms_;
  }

  bool solve_needed() const { return implicit_weight_ != 0.0; }
  GuessResult initial_guess_result(const size_t point_index) const {
    return initial_guess_types_.empty() ? GuessResult::InitialGuess
                                        : initial_guess_types_[point_index];
  }

 private:
  double implicit_weight_;
  SectorVariables inhomogeneous_terms_{};
  std::vector<GuessResult> initial_guess_types_{};
};

// Calculates the residual and jacobian for the ImplicitEquation
// pointwise, using the source from the SolveAttempt.  This involved
// setting up a local DataBox for the tags specified in the
// SolveAttempt and pulling the requested values from the evolution
// DataBox.
//
// This object performs no memory allocations, unless the SolveAttempt
// adds a tag that allocates or allocates in one of its provided
// functions.  Tags containing Variables or Tensors are optimized to
// not allocate.
template <typename ImplicitSector, typename SolveAttempt>
class ImplicitSolver {
  static_assert(
      tt::assert_conforms_to_v<ImplicitSector, protocols::ImplicitSector>);

  using sector_variables_tag =
      ::Tags::Variables<typename ImplicitSector::tensors>;
  using SectorVariables = typename sector_variables_tag::type;
  static constexpr size_t solve_dimension =
      SectorVariables::number_of_independent_components;

  using tags_from_evolution = typename SolveAttempt::tags_from_evolution;
  using EvolutionData = ForwardTuple<tags_from_evolution>;

  struct EvolutionDataTag : db::SimpleTag {
    using type = const EvolutionData*;
  };

  struct SolverPointIndex : db::SimpleTag {
    using type = size_t;
  };

  template <typename Tag, typename = typename Tag::type>
  struct FromEvolution : Tag, db::ReferenceTag {
    using base = Tag;
    using parent_tag = EvolutionDataTag;
    using argument_tags = tmpl::list<parent_tag>;
    static const typename base::type& get(
        const EvolutionData* const evolution_data) {
      return std::get<tmpl::index_of<tags_from_evolution, Tag>::value>(
          *evolution_data);
    }
  };

  template <typename Tag, typename VariablesTags>
  struct FromEvolution<Tag, Variables<VariablesTags>> : Tag, db::ComputeTag {
    using base = Tag;
    using argument_tags = tmpl::list<EvolutionDataTag, SolverPointIndex>;
    static constexpr auto function(
        const gsl::not_null<typename base::type*> result,
        const EvolutionData* const evolution_data, const size_t index) {
      result->initialize(1);
      extract_point(
          result,
          get<tmpl::index_of<tags_from_evolution, Tag>::value>(*evolution_data),
          index);
    }
  };

  // Tensor<DataVector> always allocates, so instead of creating one
  // we create a one-tensor Variables, and the DataBox will allow
  // access to the tensor transparently.  We could instead manually
  // set all the DataVectors as non-owning, pointing at individual
  // doubles, but that's more work and it's not clear it gains us
  // anything.
  template <typename Tag, typename Symm, typename IndexList>
  struct FromEvolution<Tag, Tensor<DataVector, Symm, IndexList>>
      : ::Tags::Variables<tmpl::list<Tag>>, db::ComputeTag {
    using base = ::Tags::Variables<tmpl::list<Tag>>;
    using argument_tags = tmpl::list<EvolutionDataTag, SolverPointIndex>;
    static constexpr auto function(
        const gsl::not_null<typename base::type*> result,
        const EvolutionData* const evolution_data, const size_t index) {
      result->initialize(1);
      extract_point(
          make_not_null(&get<Tag>(*result)),
          get<tmpl::index_of<tags_from_evolution, Tag>::value>(*evolution_data),
          index);
    }
  };

  using all_mutators = tmpl::remove_duplicates<
      tmpl::append<tmpl::list<typename SolveAttempt::source,
                              typename SolveAttempt::jacobian>,
                   typename SolveAttempt::source_prep,
                   typename SolveAttempt::jacobian_prep>>;

  using source_tag = db::add_tag_prefix<::Tags::Source, sector_variables_tag>;
  using jacobian_tag =
      ::Tags::Variables<jacobian_tags<typename ImplicitSector::tensors,
                                      typename source_tag::type::tags_list>>;

  using internal_simple_tags =
      tmpl::list<EvolutionDataTag, SolverPointIndex, sector_variables_tag,
                 source_tag, jacobian_tag>;
  using wrapped_tags_from_evolution =
      tmpl::transform<tags_from_evolution, tmpl::bind<FromEvolution, tmpl::_1>>;

  using simple_tags =
      tmpl::append<internal_simple_tags, typename SolveAttempt::simple_tags>;
  using compute_tags = tmpl::append<wrapped_tags_from_evolution,
                                    typename SolveAttempt::compute_tags>;

  using SolveBox =
      db::compute_databox_type<tmpl::append<simple_tags, compute_tags>>;

 public:
  template <typename PassedEvolutionData>
  ImplicitSolver(const ImplicitEquation<SectorVariables>& implicit_equation,
                 PassedEvolutionData&& data_from_evolution)
      : solve_box_(db::create<simple_tags, compute_tags>()),
        implicit_equation_(&implicit_equation) {
    static_assert(std::is_same_v<PassedEvolutionData, const EvolutionData&>,
                  "ImplicitSolver was passed a temporary.  "
                  "This will lead to a dangling pointer.");
    db::mutate_apply<
        tmpl::push_front<
            tmpl::filter<
                simple_tags,
                tt::is_a<Variables, tmpl::bind<tmpl::type_from, tmpl::_1>>>,
            EvolutionDataTag>,
        tmpl::list<>>(
        [&data_from_evolution](
            const gsl::not_null<const EvolutionData**> evolution_data_pointer,
            const auto... vars) {
          *evolution_data_pointer = &data_from_evolution;
          expand_pack((vars->initialize(1, 0.0), 0)...);
        },
        make_not_null(&solve_box_));
  }

  void set_index(const size_t index) {
    db::mutate<SolverPointIndex>(
        [&index](const gsl::not_null<size_t*> box_index) {
          *box_index = index;
        },
        make_not_null(&solve_box_));

    extract_point(make_not_null(&inhomogeneous_terms_),
                  implicit_equation_->inhomogeneous_terms(), index);

    completed_mutators_ = decltype(completed_mutators_){};
  }

  std::array<double, solve_dimension> operator()(
      const std::array<double, solve_dimension>& sector_variables_array) const {
    ASSERT(implicit_equation_->implicit_weight() != 0.0,
           "Should not be performing solves on explicit substeps");
    set_sector_variables(sector_variables_array);
    run_mutators<tmpl::push_back<typename SolveAttempt::source_prep,
                                 typename SolveAttempt::source>>();
    std::array<double, solve_dimension> residual_array{};
    SectorVariables residual(residual_array.data(), residual_array.size());
    residual =
        inhomogeneous_terms_ - db::get<sector_variables_tag>(solve_box_) +
        implicit_equation_->implicit_weight() *
            db::get<db::add_tag_prefix<::Tags::Source, sector_variables_tag>>(
                solve_box_);
    return residual_array;
  }

  std::array<std::array<double, solve_dimension>, solve_dimension> jacobian(
      const std::array<double, solve_dimension>& sector_variables_array) const {
    ASSERT(implicit_equation_->implicit_weight() != 0.0,
           "Should not be performing solves on explicit substeps");
    set_sector_variables(sector_variables_array);
    run_mutators<tmpl::push_back<typename SolveAttempt::jacobian_prep,
                                 typename SolveAttempt::jacobian>>();

    std::array<std::array<double, solve_dimension>, solve_dimension>
        jacobian_array{};
    // The storage order for the tensors does not match the required
    // order for the returned array, so we have to copy components
    // individually.
    //
    // Despite repeated references to then, the result of this is
    // independent of the *_for_offsets variables.  They are only used
    // for calculating offsets into the returned array.
    const auto& variables_for_offsets =
        db::get<sector_variables_tag>(solve_box_);
    tmpl::for_each<typename SectorVariables::tags_list>(
        [&](auto dependent_tag_v) {
          using dependent_tag = tmpl::type_from<decltype(dependent_tag_v)>;
          const auto& dependent_for_offsets =
              get<dependent_tag>(variables_for_offsets);
          for (size_t dependent_component = 0;
               dependent_component < dependent_for_offsets.size();
               ++dependent_component) {
            const auto dependent_index =
                dependent_for_offsets.get_tensor_index(dependent_component);
            auto& result_row = jacobian_array[static_cast<size_t>(
                dependent_for_offsets[dependent_component].data() -
                variables_for_offsets.data())];
            tmpl::for_each<typename SectorVariables::tags_list>(
                [&](auto independent_tag_v) {
                  using independent_tag =
                      tmpl::type_from<decltype(independent_tag_v)>;
                  using jacobian_component_tag =
                      imex::Tags::Jacobian<independent_tag,
                                           ::Tags::Source<dependent_tag>>;
                  const auto& independent_for_offsets =
                      get<independent_tag>(variables_for_offsets);

                  for (size_t independent_component = 0;
                       independent_component < independent_for_offsets.size();
                       ++independent_component) {
                    const auto independent_index =
                        independent_for_offsets.get_tensor_index(
                            independent_component);
                    result_row[static_cast<size_t>(
                        independent_for_offsets[independent_component].data() -
                        variables_for_offsets.data())] =
                        get<jacobian_component_tag>(solve_box_)
                            .get(concatenate(independent_index,
                                             dependent_index))[0];
                  }
                });
          }
        });

    jacobian_array *= implicit_equation_->implicit_weight();

    for (size_t i = 0; i < solve_dimension; ++i) {
      jacobian_array[i][i] -= 1.0;
    }
    return jacobian_array;
  }

 private:
  void set_sector_variables(
      std::array<double, solve_dimension> sector_variables_array) const {
    const SectorVariables sector_variables(sector_variables_array.data(),
                                           sector_variables_array.size());
    set_sector_variables(sector_variables);
  }

  void set_sector_variables(const SectorVariables& sector_variables) const {
    if (sector_variables == most_recent_sector_variables_) {
      return;
    }
    most_recent_sector_variables_ = sector_variables;
    db::mutate<sector_variables_tag>(
        [&sector_variables](const gsl::not_null<SectorVariables*> vars) {
          *vars = sector_variables;
        },
        make_not_null(&solve_box_));
    completed_mutators_ = decltype(completed_mutators_){};
  }

  template <typename Mutators>
  void run_mutators() const {
    tmpl::for_each<Mutators>([this](auto mutator_v) {
      using mutator = tmpl::type_from<decltype(mutator_v)>;
      if (not get<RanMutator<mutator>>(completed_mutators_)) {
        db::mutate_apply<mutator>(make_not_null(&solve_box_));
        get<RanMutator<mutator>>(completed_mutators_) = true;
      }
    });
  }

  template <typename Mutator>
  struct RanMutator {
    using type = bool;
  };

  // Re mutables: This struct is only used locally in serial
  // single-threaded implicit solves.  The gsl_multiroot interface
  // takes a const solver object, but we want to be able to share
  // calculations between the source and jacobian calculations.
  // NOLINTNEXTLINE(spectre-mutable)
  mutable SolveBox solve_box_;
  SectorVariables inhomogeneous_terms_{1};
  gsl::not_null<const ImplicitEquation<SectorVariables>*> implicit_equation_;
  // NOLINTNEXTLINE(spectre-mutable)
  mutable SectorVariables most_recent_sector_variables_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable tuples::tagged_tuple_from_typelist<
      tmpl::transform<all_mutators, tmpl::bind<RanMutator, tmpl::_1>>>
      completed_mutators_{};
};
}  // namespace solve_implicit_sector_detail

template <typename SystemVariablesTag, typename ImplicitSector>
void SolveImplicitSector<SystemVariablesTag, ImplicitSector>::apply_impl(
    const gsl::not_null<SystemVariables*> system_variables,
    const gsl::not_null<Scalar<DataVector>*> solve_failures,
    const ImexTimeStepper& time_stepper, const TimeDelta& time_step,
    const TimeSteppers::History<SectorVariables>& implicit_history,
    const Mode implicit_solve_mode, const double implicit_solve_tolerance,
    const EvolutionDataTuple& joined_evolution_data) {
  get(*solve_failures) = 0.0;

  const auto evolution_data = split_tuple<
      tmpl::transform<evolution_data_tags, tmpl::bind<tmpl::size, tmpl::_1>>>(
      joined_evolution_data);
  const auto& initial_guess_arguments = std::get<0>(evolution_data);

  const solve_implicit_sector_detail::ImplicitEquation<SectorVariables>
      equation(system_variables, time_stepper, time_step, implicit_history,
               initial_guess_arguments, tmpl::type_<ImplicitSector>{});
  if (not equation.solve_needed()) {
    return;
  }

  const size_t number_of_grid_points = get(*solve_failures).size();
  // Only allocated if used.
  Matrix semi_implicit_jacobian_matrix{};
  // Only allocated if used.
  std::vector<int> lapack_scratch{};

  bool solve_succeeded = false;
  tmpl::for_each<
      typename ImplicitSector::solve_attempts>([&](auto solve_attempt_v) {
    using solve_attempt = tmpl::type_from<decltype(solve_attempt_v)>;
    if (solve_succeeded) {
      return;
    }
    solve_succeeded = true;
    constexpr bool have_fallback =
        not std::is_same_v<solve_attempt,
                           tmpl::back<typename ImplicitSector::solve_attempts>>;
    constexpr auto attempt_number =
        tmpl::index_of<typename ImplicitSector::solve_attempts,
                       solve_attempt>::value;
    // Entry 0 is for the initial guess.
    const auto& attempt_evolution_data =
        std::get<attempt_number + 1>(evolution_data);
    solve_implicit_sector_detail::ImplicitSolver<ImplicitSector, solve_attempt>
        solver(equation, attempt_evolution_data);

    for (size_t point = 0; point < number_of_grid_points; ++point) {
      if (get(*solve_failures)[point] < attempt_number) {
        continue;
      }
      if (equation.initial_guess_result(point) == GuessResult::ExactSolution) {
        // Initial guess was written into the evolution DataBox
        // when it was computed.
        continue;
      }

      // Dimension of the vector space the (non)linear solve is performed in.
      constexpr size_t solve_dimension =
          SectorVariables::number_of_independent_components;
      std::array<double, solve_dimension> pointwise_vars_array;
      SectorVariables pointwise_vars(pointwise_vars_array.data(),
                                     pointwise_vars_array.size());
      solver.set_index(point);
      std::array<double, solve_dimension> initial_guess;
      {
        SectorVariables guess_vars(initial_guess.data(), initial_guess.size());
        extract_point(make_not_null(&guess_vars),
                      system_variables->template reference_subset<
                          typename SectorVariables::tags_list>(),
                      point);
      }
      switch (implicit_solve_mode) {
        case Mode::Implicit: {
          const size_t max_iterations = 100;
          try {
            pointwise_vars_array = RootFinder::gsl_multiroot(
                solver, initial_guess,
                RootFinder::StoppingConditions::Residual(
                    implicit_solve_tolerance),
                max_iterations);
          } catch (const convergence_error&) {
            if constexpr (have_fallback) {
              ++get(*solve_failures)[point];
              solve_succeeded = false;
              continue;
            } else {
              throw;
            }
          }
          break;
        }
        case Mode::SemiImplicit: {
          std::array<double, solve_dimension> correction_array =
              solver(initial_guess);
          DataVector correction(correction_array.data(),
                                correction_array.size());
          correction *= -1.0;
          const std::array<std::array<double, solve_dimension>, solve_dimension>
              semi_implicit_jacobian = solver.jacobian(initial_guess);
          // Copy into the dynamically allocated Matrix required by
          // the LAPACK wrapper.
          semi_implicit_jacobian_matrix = semi_implicit_jacobian;
          // Allocate scratch buffer (storing pivots from the
          // decomposition).  This does nothing after the first point.
          lapack_scratch.resize(solve_dimension);
          const int lapack_info = lapack::general_matrix_linear_solve(
              &correction, &lapack_scratch, &semi_implicit_jacobian_matrix);
          if (lapack_info != 0) {
            if (lapack_info < 0) {
              ERROR("LAPACK invalid argument: " << -lapack_info);
            } else {
              if constexpr (have_fallback) {
                ++get(*solve_failures)[point];
                solve_succeeded = false;
                continue;
              } else {
                ERROR("Semi-implicit inversion was singular at\n"
                      << pointwise_vars);
              }
            }
          }
          pointwise_vars_array = initial_guess + correction_array;
          break;
        }
        default:
          ERROR("Invalid implicit mode");
      }

      // Write the result into the evolution variables.
      auto sector_reference = system_variables->template reference_subset<
          typename SectorVariables::tags_list>();
      overwrite_point(make_not_null(&sector_reference), pointwise_vars, point);
    }
  });
}
}  // namespace imex
