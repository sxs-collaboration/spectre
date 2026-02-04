// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/DummyReconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"

namespace Ccz4::fd {
std::unique_ptr<Reconstructor> DummyReconstructor::get_clone() const {
  return std::make_unique<DummyReconstructor>(*this);
}

void DummyReconstructor::pup(PUP::er& p) { Reconstructor::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID DummyReconstructor::my_PUP_ID = 0;
}  // namespace Ccz4::fd
