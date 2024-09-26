// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/ByBlock.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>
#include <pup_stl.h>
#include <utility>
#include <vector>

#include "Domain/Structure/Element.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace StepChoosers {
template <size_t Dim>
ByBlock<Dim>::ByBlock(std::vector<double> sizes) : sizes_(std::move(sizes)) {}

template <size_t Dim>
std::pair<TimeStepRequest, bool> ByBlock<Dim>::operator()(
    const Element<Dim>& element, const double last_step) const {
  const size_t block = element.id().block_id();
  if (block >= sizes_.size()) {
    ERROR("Step size not specified for block " << block);
  }
  return {{.size_goal = std::copysign(sizes_[block], last_step)}, true};
}

template <size_t Dim>
bool ByBlock<Dim>::uses_local_data() const {
  return true;
}

template <size_t Dim>
bool ByBlock<Dim>::can_be_delayed() const {
  return true;
}

template <size_t Dim>
void ByBlock<Dim>::pup(PUP::er& p) {
  StepChooser<StepChooserUse::Slab>::pup(p);
  StepChooser<StepChooserUse::LtsStep>::pup(p);
  p | sizes_;
}

template <size_t Dim>
PUP::able::PUP_ID ByBlock<Dim>::my_PUP_ID = 0;  // NOLINT

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class ByBlock<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace StepChoosers
