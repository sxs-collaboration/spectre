// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/Tensor/IndexType.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Options/Context.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace ah {
template <typename Fr>
HorizonOptions<Fr>::HorizonOptions(
    ylm::Strahlkorper<Fr> initial_guess_in, ::FastFlow fast_flow_in,
    ::Verbosity verbosity_in, const size_t max_compute_coords_retries_in,
    std::optional<std::vector<std::string>> blocks_for_horizon_find_in)
    : initial_guess(std::move(initial_guess_in)),
      fast_flow(std::move(fast_flow_in)),  // NOLINT
      verbosity(std::move(verbosity_in)),  // NOLINT
      max_compute_coords_retries(max_compute_coords_retries_in),
      blocks_for_horizon_find(std::move(blocks_for_horizon_find_in)) {}

template <typename Fr>
void HorizonOptions<Fr>::pup(PUP::er& p) {
  p | initial_guess;
  p | fast_flow;
  p | verbosity;
  p | max_compute_coords_retries;
  p | blocks_for_horizon_find;
}

template <typename Fr>
bool operator==(const HorizonOptions<Fr>& lhs, const HorizonOptions<Fr>& rhs) {
  return lhs.initial_guess == rhs.initial_guess and
         lhs.fast_flow == rhs.fast_flow and lhs.verbosity == rhs.verbosity and
         lhs.max_compute_coords_retries == rhs.max_compute_coords_retries and
         lhs.blocks_for_horizon_find == rhs.blocks_for_horizon_find;
}

template <typename Fr>
bool operator!=(const HorizonOptions<Fr>& lhs, const HorizonOptions<Fr>& rhs) {
  return not(lhs == rhs);
}

// Explicit instantiations
#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                        \
  template struct HorizonOptions<FRAME(data)>;                      \
  template bool operator==(const HorizonOptions<FRAME(data)>& lhs,  \
                           const HorizonOptions<FRAME(data)>& rhs); \
  template bool operator!=(const HorizonOptions<FRAME(data)>& lhs,  \
                           const HorizonOptions<FRAME(data)>& rhs);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (::Frame::Grid, ::Frame::Distorted, ::Frame::Inertial))

#undef FRAME
#undef INSTANTIATE
}  // namespace ah
