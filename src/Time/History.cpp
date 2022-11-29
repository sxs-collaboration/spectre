// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/History.hpp"

#include "Utilities/GenerateInstantiations.hpp"

namespace TimeSteppers {
template <typename T>
bool operator==(const UntypedStepRecord<T>& a, const UntypedStepRecord<T>& b) {
  return a.time_step_id == b.time_step_id and a.value == b.value and
         a.derivative == b.derivative;
}

template <typename T>
bool operator!=(const UntypedStepRecord<T>& a, const UntypedStepRecord<T>& b) {
  return not(a == b);
}

template <typename T>
size_t ConstUntypedHistory<T>::UntypedSubsteps::size() const {
  return history_->substep_values().size();
}

template <typename T>
size_t ConstUntypedHistory<T>::UntypedSubsteps::max_size() const {
  return history_->substep_values().max_size();
}

template <typename T>
auto ConstUntypedHistory<T>::UntypedSubsteps::operator[](
    const size_t index) const -> const UntypedStepRecord<T>& {
  ASSERT(index < size(),
         "Requested substep " << index << " but only have " << size());
  return history_->substep_values()[index];
}

template <typename T>
ConstUntypedHistory<T>::UntypedSubsteps::UntypedSubsteps(
    const ConstUntypedHistory& history)
    : history_(&history) {}

template <typename T>
auto ConstUntypedHistory<T>::substeps() const -> UntypedSubsteps {
  return UntypedSubsteps(*this);
}

namespace History_detail {
template <typename UntypedBase>
size_t UntypedAccessCommon<UntypedBase>::integration_order() const {
  return integration_order_;
}

template <typename UntypedBase>
size_t UntypedAccessCommon<UntypedBase>::size() const {
  return step_values_.size();
}

template <typename UntypedBase>
size_t UntypedAccessCommon<UntypedBase>::max_size() const {
  return history_max_past_steps + 2;
}

template <typename UntypedBase>
auto UntypedAccessCommon<UntypedBase>::operator[](const size_t index) const
    -> const UntypedStepRecord<WrapperType>& {
  ASSERT(index < size(),
         "Requested step " << index << " but only have " << size());
  return step_values_[index];
}

template <typename UntypedBase>
auto UntypedAccessCommon<UntypedBase>::operator[](const TimeStepId& id) const
    -> const UntypedStepRecord<WrapperType>& {
  return find_record(*this, id);
}

template <typename UntypedBase>
bool UntypedAccessCommon<UntypedBase>::at_step_start() const {
  return substep_values_.empty() or
         substep_values_.back().time_step_id < this->back().time_step_id;
}

template <typename UntypedBase>
auto UntypedAccessCommon<UntypedBase>::substep_values() const
    -> const boost::container::static_vector<UntypedStepRecord<WrapperType>,
                                             history_max_substeps>& {
  return substep_values_;
}
}  // namespace History_detail

#define MATH_WRAPPER_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                   \
  template bool operator==(                                    \
      const UntypedStepRecord<MATH_WRAPPER_TYPE(data)>& a,     \
      const UntypedStepRecord<MATH_WRAPPER_TYPE(data)>& b);    \
  template bool operator!=(                                    \
      const UntypedStepRecord<MATH_WRAPPER_TYPE(data)>& a,     \
      const UntypedStepRecord<MATH_WRAPPER_TYPE(data)>& b);    \
  template class ConstUntypedHistory<MATH_WRAPPER_TYPE(data)>; \
  template class History_detail::UntypedAccessCommon<          \
      ConstUntypedHistory<MATH_WRAPPER_TYPE(data)>>;           \
  template class History_detail::UntypedAccessCommon<          \
      MutableUntypedHistory<MATH_WRAPPER_TYPE(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (MATH_WRAPPER_TYPES))

#undef INSTANTIATE
#undef MATH_WRAPPER_TYPE
}  // namespace TimeSteppers
