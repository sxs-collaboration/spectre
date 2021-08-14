// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <random>
#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
/// \endcond

namespace elliptic::Actions {

/*!
 * \brief Optionally add random noise to the initial guess.
 *
 * Add this action to the action list just after
 * `elliptic::Actions::InitializeFields` to add random noise to the initial
 * guess. The random noise can be toggled and is configurable with input-file
 * options. A random initial guess can be useful to test the convergence of the
 * elliptic solver.
 */
template <typename System>
struct RandomizeInitialGuess {
 private:
  using fields_tag = ::Tags::Variables<typename System::primal_fields>;

 public:
  // Bundle the options so they can be placed in Options::Auto
  struct RandomParameters {
    struct Amplitude {
      using type = double;
      static constexpr Options::String help = "Amplitude of the uniform noise.";
    };
    struct Seed {
      using type = Options::Auto<size_t>;
      static constexpr Options::String help =
          "Random seed for the noise generator.";
    };
    using options = tmpl::list<Amplitude, Seed>;
    static constexpr Options::String help = "Parameters for the uniform noise.";
    void pup(PUP::er& p) noexcept {
      p | amplitude;
      p | seed;
    }
    double amplitude;
    std::optional<size_t> seed;
  };

  struct RandomParametersOptionTag {
    static std::string name() noexcept { return "RandomizeInitialGuess"; }
    using type = Options::Auto<RandomParameters, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "Add uniform random noise to the initial guess.";
  };

  struct RandomParametersTag : db::SimpleTag {
    using type = std::optional<RandomParameters>;
    using option_tags = tmpl::list<RandomParametersOptionTag>;
    static constexpr bool pass_metavariables = false;
    static type create_from_options(const type& value) noexcept {
      return value;
    }
  };

  using const_global_cache_tags = tmpl::list<RandomParametersTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Retrieve the options
    const std::optional<RandomParameters>& params =
        db::get<RandomParametersTag>(box);
    if (not params.has_value()) {
      return {std::move(box)};
    }
    const auto& [amplitude, seed] = params.value();
    // Seed a random generator. Include the element ID in the seed so the data
    // is different on each element.
    std::seed_seq seeds{std::hash<ElementId<Dim>>{}(element_id),
                        seed.value_or(std::random_device{}())};
    std::mt19937 generator{seeds};
    // Set up the random distribution
    std::uniform_real_distribution<> dist(-amplitude, amplitude);
    // Add noise to the fields
    db::mutate<fields_tag>(make_not_null(&box),
                           [&generator, &dist](const auto fields) noexcept {
                             for (size_t i = 0; i < fields->size(); ++i) {
                               fields->data()[i] += dist(generator);
                             }
                           });
    return {std::move(box)};
  }
};

}  // namespace elliptic::Actions
