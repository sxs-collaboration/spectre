// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/String.hpp"
#include "ParallelAlgorithms/Amr/Policies/Isotropy.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace amr {
/// \brief A set of runtime policies controlling adaptive mesh refinement
class Policies {
 public:
  /// The isotropy of AMR
  struct Isotropy {
    using type = amr::Isotropy;
    static constexpr Options::String help = {
        "Isotropy of adaptive mesh refinement (whether or not each dimension "
        "can be refined independently)."};
  };

  using options = tmpl::list<Isotropy>;

  static constexpr Options::String help = {
      "Policies controlling adaptive mesh refinement."};

  Policies() = default;

  explicit Policies(amr::Isotropy isotropy);

  amr::Isotropy isotropy() const { return isotropy_; }

  void pup(PUP::er& p);

 private:
  amr::Isotropy isotropy_{amr::Isotropy::Anisotropic};
};

bool operator==(const Policies& lhs, const Policies& rhs);

bool operator!=(const Policies& lhs, const Policies& rhs);
}  // namespace amr
