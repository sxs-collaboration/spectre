// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/EqualRateLts/FixedLtsRatio.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Time/CollectStepperErrorTolerances.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace evolution::dg::StepChoosers {
template <size_t Dim>
FixedLtsRatio<Dim>::FixedLtsRatio(
    std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>
        step_choosers,
    std::optional<std::vector<std::string>> regions)
    : step_choosers_(std::move(step_choosers)), regions_(std::move(regions)) {}

template <size_t Dim>
bool FixedLtsRatio<Dim>::uses_local_data() const {
  return true;
}

template <size_t Dim>
bool FixedLtsRatio<Dim>::can_be_delayed() const {
  return true;
}

template <size_t Dim>
bool FixedLtsRatio<Dim>::must_set_step_size() const {
  return false;
}

template <size_t Dim>
std::unordered_map<std::type_index, StepperErrorTolerances>
FixedLtsRatio<Dim>::tolerances() const {
  std::unordered_map<std::type_index, StepperErrorTolerances> tolerances{};
  for (const auto& step_chooser : step_choosers_) {
    collect_stepper_error_tolerances(&tolerances, *step_chooser);
  }
  return tolerances;
}

template <size_t Dim>
void FixedLtsRatio<Dim>::pup(PUP::er& p) {
  StepChooser<StepChooserUse::Slab>::pup(p);
  p | step_choosers_;
  p | regions_;
}

template <size_t Dim>
PUP::able::PUP_ID FixedLtsRatio<Dim>::my_PUP_ID = 0;  // NOLINT

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class FixedLtsRatio<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace evolution::dg::StepChoosers
