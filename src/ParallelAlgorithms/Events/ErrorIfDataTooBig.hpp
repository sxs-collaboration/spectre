// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataBox/ValidateSelection.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ExtractPoint.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace Events {
/*!
 * \brief ERROR if tensors get too big.
 *
 * The magnitudes of the components of the specified tensors are
 * checked, and if any exceed the specified threshold an ERROR is
 * thrown, terminating the evolution.
 *
 * Any `Tensor` in the `db::DataBox` can be checked but must be listed
 * in the `Tensors` template parameter. Any additional compute tags
 * that hold a `Tensor` can also be added to the `Tensors` template
 * parameter. Finally, `Variables` and other non-tensor compute tags
 * used to calculate tensors can be listed in the
 * `NonTensorComputeTags`.
 *
 * \note The `NonTensorComputeTags` are intended to be used for `Variables`
 * compute tags like `Tags::DerivCompute`
 */
template <size_t Dim, typename Tensors, typename NonTensorComputeTags>
class ErrorIfDataTooBig : public Event {
 public:
  /// \cond
  explicit ErrorIfDataTooBig(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ErrorIfDataTooBig);  // NOLINT
  /// \endcond

  struct VariablesToCheck {
    static constexpr Options::String help = "Subset of variables to check";
    using type = std::vector<std::string>;
    static size_t lower_bound_on_size() { return 1; }
  };

  struct Threshold {
    static constexpr Options::String help = "Threshold at which to ERROR";
    using type = double;
    static type lower_bound() { return 0.0; }
  };

  using options = tmpl::list<VariablesToCheck, Threshold>;

  static constexpr Options::String help = "ERROR if tensors get too big";

  ErrorIfDataTooBig() = default;

  ErrorIfDataTooBig(const std::vector<std::string>& variables_to_check,
                    const double threshold, const Options::Context& context)
      : variables_to_check_(variables_to_check.begin(),
                            variables_to_check.end()),
        threshold_(threshold) {
    db::validate_selection<Tensors>(variables_to_check, context);
  }

  using compute_tags_for_observation_box =
      tmpl::append<Tensors, NonTensorComputeTags>;

  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::ObservationBox,
                 ::domain::Tags::Coordinates<Dim, Frame::Inertial>>;

  template <typename ComputeTagsList, typename DataBoxType,
            typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const ObservationBox<ComputeTagsList, DataBoxType>& box,
                  const tnsr::I<DataVector, Dim>& coordinates,
                  Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ArrayIndex& /*array_index*/,
                  const ParallelComponent* const /*component*/,
                  const ObservationValue& /*observation_value*/) const {
    tmpl::for_each<Tensors>([&](const auto tensor_tag_v) {
      using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
      const std::string tag_name = db::tag_name<tensor_tag>();
      if (variables_to_check_.count(tag_name) != 0) {
        const auto check_components = [&](const auto& tensor) {
          for (const auto& component : tensor) {
            for (size_t point = 0; point < component.size(); ++point) {
              if (std::abs(component[point]) > threshold_) {
                ERROR_NO_TRACE(
                    tag_name
                    << " too big with value " << extract_point(tensor, point)
                    << " at position\n"
                    << extract_point(coordinates, point) << "\nwith ElementId: "
                    << get<::domain::Tags::Element<Dim>>(box).id());
              }
            }
          }
        };
        const auto& tensor = get<tensor_tag>(box);
        if constexpr (tt::is_a_v<std::optional,
                                 std::decay_t<decltype(tensor)>>) {
          if (tensor.has_value()) {
            check_components(*tensor);
          }
        } else {
          check_components(tensor);
        }
      }
    });
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event::pup(p);
    p | variables_to_check_;
    p | threshold_;
  }

 private:
  std::unordered_set<std::string> variables_to_check_{};
  double threshold_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <size_t Dim, typename Tensors, typename NonTensorComputeTags>
PUP::able::PUP_ID
    ErrorIfDataTooBig<Dim, Tensors,
                      NonTensorComputeTags>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
