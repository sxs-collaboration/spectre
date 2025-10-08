// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmSphereToCart.hpp"

#include <blaze/math/CompressedMatrix.h>
#include <complex>
#include <optional>

#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/SparseMatrixFiller.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/WignerThreeJ.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Numeric.hpp"

namespace ylm::TensorYlm {

namespace {
// Inner loops of the rank-1 calculation.  The purpose of this
// function is so that there are not so many nested loops inside of
// the main function, making the main function and this function more
// readable.
void inner_loops_one(SparseMatrixFiller& filler, SpherepackIterator& iter_src,
                     SpherepackIterator& iter_dest, const size_t src_comp_index,
                     const size_t dest_comp_index, const size_t ell_max,
                     const size_t l_dest, const int m_dest, const int scheck,
                     const int mj, const std::complex<double> kj,
                     WignerThreeJ& threej_m, WignerThreeJ& threej_s,
                     const int sign_m_dest, const int sign_delta_sb,
                     const int sign_delta_sb_conj) {
  const auto add_element = [&filler, &iter_src, &iter_dest, dest_comp_index,
                            src_comp_index](const double element) {
    const size_t indx_dest =
        iter_dest() + dest_comp_index * iter_dest.spherepack_array_size();
    const size_t indx_src =
        iter_src() + src_comp_index * iter_src.spherepack_array_size();
    filler.add(element, indx_dest, indx_src);
  };
  const int m_src = m_dest - mj;
  for (size_t l_src =
           std::max(static_cast<size_t>(abs(m_src)), threej_s.l1_min());
       (l_src <= threej_s.l1_max() and l_src <= ell_max); ++l_src) {
    const double sqterm =
        sqrt(0.5 * static_cast<double>((2 * l_dest + 1) * (2 * l_src + 1)));
    const double coef_without_kj = sqterm * threej_s(l_src) * threej_m(l_src);
    const int sign_conjA =
        ((static_cast<int>(l_src + l_dest) + 1 - scheck + m_src) % 2 == 0 ? 1
                                                                          : -1);
    const int sign_plus_msrc_term = sign_m_dest * sign_delta_sb;
    const int sign_min_msrc_term = sign_delta_sb_conj * sign_conjA;
    if (m_src >= 0) {
      // Main term.
      if (kj.imag() == 0) {
        // Main term.
        if (iter_dest.coefficient_array() ==
            SpherepackIterator::CoefficientArray::a) {
          // ReRe
          iter_src.set(l_src, static_cast<size_t>(m_src),
                       SpherepackIterator::CoefficientArray::a);
          add_element(kj.real() * coef_without_kj * sign_plus_msrc_term);
        } else if (iter_dest.coefficient_array() ==
                   SpherepackIterator::CoefficientArray::b) {
          // ImIm
          iter_src.set(l_src, static_cast<size_t>(m_src),
                       SpherepackIterator::CoefficientArray::b);
          add_element(kj.real() * coef_without_kj * sign_plus_msrc_term);
        }
      } else {
        if (iter_dest.coefficient_array() ==
            SpherepackIterator::CoefficientArray::a) {
          // ReIm
          iter_src.set(l_src, static_cast<size_t>(m_src),
                       SpherepackIterator::CoefficientArray::b);
          add_element(-kj.imag() * coef_without_kj * sign_plus_msrc_term);
        } else if (iter_dest.coefficient_array() ==
                   SpherepackIterator::CoefficientArray::b) {
          // ImRe
          iter_src.set(l_src, static_cast<size_t>(m_src),
                       SpherepackIterator::CoefficientArray::a);
          add_element(kj.imag() * coef_without_kj * sign_plus_msrc_term);
        }
      }
    } else {
      if (kj.imag() == 0) {
        // Other term, for negative m_src.
        if (iter_dest.coefficient_array() ==
            SpherepackIterator::CoefficientArray::a) {
          // ReRe
          iter_src.set(l_src, static_cast<size_t>(-m_src),
                       SpherepackIterator::CoefficientArray::a);
          add_element(kj.real() * coef_without_kj * sign_min_msrc_term);
        } else if (iter_dest.coefficient_array() ==
                   SpherepackIterator::CoefficientArray::b) {
          // ImIm
          iter_src.set(l_src, static_cast<size_t>(-m_src),
                       SpherepackIterator::CoefficientArray::b);
          add_element(-kj.real() * coef_without_kj * sign_min_msrc_term);
        }
      } else {
        if (iter_dest.coefficient_array() ==
            SpherepackIterator::CoefficientArray::a) {
          // ReIm
          iter_src.set(l_src, static_cast<size_t>(-m_src),
                       SpherepackIterator::CoefficientArray::b);
          add_element(kj.imag() * coef_without_kj * sign_min_msrc_term);
        } else if (iter_dest.coefficient_array() ==
                   SpherepackIterator::CoefficientArray::b) {
          // ImRe
          iter_src.set(l_src, static_cast<size_t>(-m_src),
                       SpherepackIterator::CoefficientArray::a);
          add_element(kj.imag() * coef_without_kj * sign_min_msrc_term);
        }
      }
    }
  }
}

// Inner loops of the rank-2 calculation.  The purpose of this
// function is so that there are not so many nested loops inside of
// the main function, making the main function and this function more
// readable.
void inner_loops_two(SparseMatrixFiller& filler, SpherepackIterator& iter_src,
                     SpherepackIterator& iter_dest, const size_t src_comp_index,
                     const size_t dest_comp_index, const size_t ell_max,
                     const size_t l_dest, const int m_dest, const size_t lbar,
                     const std::vector<int>& mbars,
                     std::vector<std::optional<WignerThreeJ>>& threej_mbars,
                     WignerThreeJ& threej_ells_sbar, const double symm_factor,
                     const std::array<helpers::BasisVector, 2>& dest_bvs,
                     std::vector<WignerThreeJ>& threej_pqs,
                     const double threej_ones_sbar_val, const int sign_sbar,
                     const int sign_delta_sb, const int sign_delta_sb_conj) {
  const auto add_element = [&filler, &iter_src, &iter_dest, src_comp_index,
                            dest_comp_index](const double element) {
    const size_t indx_dest =
        iter_dest() + dest_comp_index * iter_dest.spherepack_array_size();
    const size_t indx_src =
        iter_src() + src_comp_index * iter_src.spherepack_array_size();
    filler.add(element, indx_dest, indx_src);
  };
  size_t mbar_indx = 0;
  for (int p = -1; p <= 1; p += 2) {
    for (int q = -1; q <= 1; q += 2, ++mbar_indx) {
      if (static_cast<size_t>(std::abs(mbars[mbar_indx])) <= lbar) {
        const int m_src = m_dest - mbars[mbar_indx];
        const int sign_m_src = (m_src % 2 == 0 ? 1 : -1);
        const std::complex<double> kj =
            helpers::bv_to_k(dest_bvs[0], p) * helpers::bv_to_k(dest_bvs[1], q);
        for (size_t l_src = std::max(static_cast<size_t>(abs(m_src)),
                                     threej_ells_sbar.l1_min());
             (l_src <= threej_ells_sbar.l1_max() and l_src <= ell_max);
             ++l_src) {
          const double sqterm =
              0.5 *
              sqrt(static_cast<double>((2 * l_dest + 1) * (2 * l_src + 1)));
          const int sign_toprow = ((l_src + l_dest) % 2 == 0 ? 1 : -1);
          const int sign_plus_msrc_term =
              sign_m_src * sign_delta_sb * sign_sbar;
          const int sign_min_msrc_term = sign_delta_sb_conj * sign_toprow;
          const double coef_without_kj =
              sqterm * threej_ells_sbar(l_src) * threej_ones_sbar_val *
              threej_mbars[mbar_indx].value()(l_src) *
              threej_pqs[mbar_indx](lbar) * symm_factor *
              static_cast<double>(2 * lbar + 1);

          if (m_src >= 0) {
            if (kj.imag() == 0) {
              // Main term.
              if (iter_dest.coefficient_array() ==
                  SpherepackIterator::CoefficientArray::a) {
                // ReRe
                iter_src.set(l_src, static_cast<size_t>(m_src),
                             SpherepackIterator::CoefficientArray::a);
                add_element(kj.real() * coef_without_kj * sign_plus_msrc_term);
              } else if (iter_dest.coefficient_array() ==
                         SpherepackIterator::CoefficientArray::b) {
                // ImIm
                iter_src.set(l_src, static_cast<size_t>(m_src),
                             SpherepackIterator::CoefficientArray::b);
                add_element(kj.real() * coef_without_kj * sign_plus_msrc_term);
              }
            } else {
              if (iter_dest.coefficient_array() ==
                  SpherepackIterator::CoefficientArray::a) {
                // ReIm
                iter_src.set(l_src, static_cast<size_t>(m_src),
                             SpherepackIterator::CoefficientArray::b);
                add_element(-kj.imag() * coef_without_kj * sign_plus_msrc_term);
              } else if (iter_dest.coefficient_array() ==
                         SpherepackIterator::CoefficientArray::b) {
                // ImRe
                iter_src.set(l_src, static_cast<size_t>(m_src),
                             SpherepackIterator::CoefficientArray::a);
                add_element(kj.imag() * coef_without_kj * sign_plus_msrc_term);
              }
            }
          } else {
            if (kj.imag() == 0) {
              // Other term, for negative m_src.
              if (iter_dest.coefficient_array() ==
                  SpherepackIterator::CoefficientArray::a) {
                // ReRe
                iter_src.set(l_src, static_cast<size_t>(-m_src),
                             SpherepackIterator::CoefficientArray::a);
                add_element(kj.real() * coef_without_kj * sign_min_msrc_term);
              } else if (iter_dest.coefficient_array() ==
                         SpherepackIterator::CoefficientArray::b) {
                // ImIm
                iter_src.set(l_src, static_cast<size_t>(-m_src),
                             SpherepackIterator::CoefficientArray::b);
                add_element(-kj.real() * coef_without_kj * sign_min_msrc_term);
              }
            } else {
              if (iter_dest.coefficient_array() ==
                  SpherepackIterator::CoefficientArray::a) {
                // ReIm
                iter_src.set(l_src, static_cast<size_t>(-m_src),
                             SpherepackIterator::CoefficientArray::b);
                add_element(kj.imag() * coef_without_kj * sign_min_msrc_term);
              } else if (iter_dest.coefficient_array() ==
                         SpherepackIterator::CoefficientArray::b) {
                // ImRe
                iter_src.set(l_src, static_cast<size_t>(-m_src),
                             SpherepackIterator::CoefficientArray::a);
                add_element(kj.imag() * coef_without_kj * sign_min_msrc_term);
              }
            }
          }
        }
      }
    }
  }
}

// Inner loops of the rank-3 calculation.  The purpose of this
// function is so that there are not so many nested loops inside of
// the main function, making the main function and this function more
// readable.
template <typename Symm>
void inner_loops_three(SparseMatrixFiller& filler, SpherepackIterator& iter_src,
                       SpherepackIterator& iter_dest,
                       const size_t src_comp_index,
                       const size_t dest_comp_index, const size_t ell_max,
                       const size_t l_dest, const int m_dest,
                       const size_t lcheck, const int mcheck, const int sbar,
                       const int scheck,
                       std::vector<WignerThreeJ>& threej_ones_sbars,
                       WignerThreeJ& threej_ssum, WignerThreeJ& threej_msdc,
                       std::vector<std::optional<WignerThreeJ>>& threej_sb0s,
                       std::vector<WignerThreeJ>& threej_pqs,
                       std::vector<std::optional<WignerThreeJ>>& threej_rs,
                       const std::vector<int>& mbars,
                       const std::array<helpers::BasisVector, 3>& src_bvs,
                       const std::array<helpers::BasisVector, 3>& dest_bvs,
                       const size_t src_multiplicity, const int sign_all_s,
                       const int sign_delta_sb, const int sign_delta_sb_conj) {
  const auto add_element = [&filler, &iter_src, &iter_dest, src_comp_index,
                            dest_comp_index](const double element) {
    const size_t indx_dest =
        iter_dest() + dest_comp_index * iter_dest.spherepack_array_size();
    const size_t indx_src =
        iter_src() + src_comp_index * iter_src.spherepack_array_size();
    filler.add(element, indx_dest, indx_src);
  };
  size_t mbar_indx = 0;
  for (int p = -1; p <= 1; p += 2) {
    for (int q = -1; q <= 1; q += 2, ++mbar_indx) {
      const int sign_mbar =
          ((sbar + scheck - mbars[mbar_indx]) % 2 == 0 ? 1 : -1);
      const std::complex<double> kj12 =
          helpers::bv_to_k(dest_bvs[1], p) * helpers::bv_to_k(dest_bvs[2], q);
      for (size_t lbar = static_cast<size_t>(
               std::max(std::abs(mbars[mbar_indx]), std::max(0, abs(sbar))));
           lbar <= 2; ++lbar) {
        const double threej_ones_sbar_val = threej_ones_sbars
            [static_cast<size_t>(helpers::bv_to_s(src_bvs[2]) + 1) +
             3 * static_cast<size_t>(helpers::bv_to_s(src_bvs[1]) + 1)](lbar);
        const double threej_pq_val = threej_pqs[mbar_indx](lbar);
        const int lbar_term =
            static_cast<int>((2 * lbar + 1) * (2 * lcheck + 1));
        const double threej_sb0_val = threej_sb0s[lbar].value()(lcheck);
        const double symm_factor =
            helpers::get_symm_factor<Symm>(src_multiplicity, lbar);
        if (symm_factor != 0.0 and threej_pq_val != 0.0 and
            threej_sb0_val != 0.0 and threej_ones_sbar_val != 0.0) {
          for (int r = -1; r <= 1; r += 2) {
            const int mr = helpers::bv_to_m(dest_bvs[0], r);
            if (mcheck == mbars[mbar_indx] + mr) {
              const int m_src = m_dest - mcheck;
              const int sign_m_src = (m_src % 2 == 0 ? 1 : -1);
              const std::complex<double> kj =
                  helpers::bv_to_k(dest_bvs[0], r) * kj12;
              const double threej_r_val =
                  threej_rs[lbar + 3 * static_cast<size_t>((r + 1) / 2) +
                            6 * mbar_indx]
                      .value()(lcheck);
              for (size_t l_src =
                       std::max(threej_ssum.l1_min(),
                                std::max(static_cast<size_t>(std::abs(m_src)),
                                         threej_msdc.l1_min()));
                   (l_src <=
                        std::min(threej_msdc.l1_max(), threej_ssum.l1_max()) and
                    l_src <= ell_max);
                   ++l_src) {
                const double threej_msdc_val = threej_msdc(l_src);
                const double threej_ssum_val = threej_ssum(l_src);
                const double sqterm = sqrt(
                    static_cast<double>((2 * l_dest + 1) * (2 * l_src + 1)) *
                    0.125);
                const int sign_toprow =
                    ((l_src + l_dest + 1) % 2 == 0 ? 1 : -1);
                const double coef_without_kj =
                    sqterm * threej_ones_sbar_val * threej_ssum_val *
                    threej_pq_val * threej_r_val * threej_sb0_val *
                    threej_msdc_val * symm_factor * lbar_term;
                const int sign_plus_msrc_term =
                    sign_m_src * sign_delta_sb * sign_mbar;
                const int sign_min_msrc_term =
                    sign_delta_sb_conj * sign_mbar * sign_toprow * sign_all_s;
                if (m_src >= 0) {
                  if (kj.imag() == 0) {
                    // Main term.
                    if (iter_dest.coefficient_array() ==
                        SpherepackIterator::CoefficientArray::a) {
                      // ReRe
                      iter_src.set(l_src, static_cast<size_t>(m_src),
                                   SpherepackIterator::CoefficientArray::a);
                      add_element(kj.real() * coef_without_kj *
                                  sign_plus_msrc_term);
                    } else if (iter_dest.coefficient_array() ==
                               SpherepackIterator::CoefficientArray::b) {
                      // ImIm
                      iter_src.set(l_src, static_cast<size_t>(m_src),
                                   SpherepackIterator::CoefficientArray::b);
                      add_element(kj.real() * coef_without_kj *
                                  sign_plus_msrc_term);
                    }
                  } else {
                    if (iter_dest.coefficient_array() ==
                        SpherepackIterator::CoefficientArray::a) {
                      // ReIm
                      iter_src.set(l_src, static_cast<size_t>(m_src),
                                   SpherepackIterator::CoefficientArray::b);
                      add_element(-kj.imag() * coef_without_kj *
                                  sign_plus_msrc_term);
                    } else if (iter_dest.coefficient_array() ==
                               SpherepackIterator::CoefficientArray::b) {
                      // ImRe
                      iter_src.set(l_src, static_cast<size_t>(m_src),
                                   SpherepackIterator::CoefficientArray::a);
                      add_element(kj.imag() * coef_without_kj *
                                  sign_plus_msrc_term);
                    }
                  }
                } else {
                  if (kj.imag() == 0) {
                    // Other term, for negative m_src.
                    if (iter_dest.coefficient_array() ==
                        SpherepackIterator::CoefficientArray::a) {
                      // ReRe
                      iter_src.set(l_src, static_cast<size_t>(-m_src),
                                   SpherepackIterator::CoefficientArray::a);
                      add_element(kj.real() * coef_without_kj *
                                  sign_min_msrc_term);
                    } else if (iter_dest.coefficient_array() ==
                               SpherepackIterator::CoefficientArray::b) {
                      // ImIm
                      iter_src.set(l_src, static_cast<size_t>(-m_src),
                                   SpherepackIterator::CoefficientArray::b);
                      add_element(-kj.real() * coef_without_kj *
                                  sign_min_msrc_term);
                    }
                  } else {
                    if (iter_dest.coefficient_array() ==
                        SpherepackIterator::CoefficientArray::a) {
                      // ReIm
                      iter_src.set(l_src, static_cast<size_t>(-m_src),
                                   SpherepackIterator::CoefficientArray::b);
                      add_element(kj.imag() * coef_without_kj *
                                  sign_min_msrc_term);
                    } else if (iter_dest.coefficient_array() ==
                               SpherepackIterator::CoefficientArray::b) {
                      // ImRe
                      iter_src.set(l_src, static_cast<size_t>(-m_src),
                                   SpherepackIterator::CoefficientArray::a);
                      add_element(kj.imag() * coef_without_kj *
                                  sign_min_msrc_term);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
}  // namespace

template <typename TensorStructure, typename SparseMatrixType>
void fill_sphere_to_cart(const gsl::not_null<SparseMatrixType*> matrix,
                      const size_t ell_max) {
  static constexpr size_t num_independent_components = TensorStructure::size();
  static constexpr size_t rank = TensorStructure::rank();
  static constexpr auto tensor_index_list =
      TensorStructure::storage_to_tensor_index();

  static_assert(rank > 0 and rank < 4, "Implemented only for ranks 1,2,3");

  SpherepackIterator iter_src(ell_max, ell_max, 1, false);
  SpherepackIterator iter_dest(ell_max, ell_max, 1, false);
  SparseMatrixFiller filler(square(num_independent_components) *
                                iter_src.spherepack_array_size() *
                                iter_dest.spherepack_array_size(),
                            true, 1.0);

  // We allocate space for some (not all) of the 3J symbols here, so
  // that they can be used and re-used later inside the inner loops.
  std::vector<WignerThreeJ> threej_pqs;  // NOLINT(misc-const-correctness)
  std::vector<int> mbars;                // NOLINT(misc-const-correctness)
  if constexpr (rank > 1) {
    threej_pqs.reserve(4);
    mbars.reserve(4);
  } else {
    // For rank 1 we don't need threej_pqs or mbars but we
    // need to declare them anyway for scoping; they will just be unused. We
    // will compute the rank-1 3J symbols below, and below we will also compute
    // additional 3J symbols for higher ranks.
    (void)threej_pqs;
    (void)mbars;
  }

  // threej_rs contains std::optionals because some of the Wigner 3Js
  // are not well-defined (because |m| > ell).  The ones without
  // values are never used or referenced.  It is possible to compute
  // fewer than 24 threej_rs, by pushing_back only those that have
  // values, but then it is difficult to index the threej_rs.  So
  // keeping 24 of them and making them std::optionals is easier.
  //
  // Same thing is true for threej_sb0s.
  // NOLINTNEXTLINE(misc-const-correctness)
  std::vector<std::optional<WignerThreeJ>> threej_rs;
  // NOLINTNEXTLINE(misc-const-correctness)
  std::vector<std::optional<WignerThreeJ>> threej_sb0s;
  if constexpr (rank > 2) {
    threej_rs.reserve(24);
    threej_sb0s.reserve(3);
  } else {
    // Unneeded except for rank 3.
    (void)threej_rs;
    (void)threej_sb0s;
  }

  // Same thing for threej_mbars, but for rank 2.
  // NOLINTNEXTLINE(misc-const-correctness)
  std::vector<std::optional<WignerThreeJ>> threej_mbars;
  if constexpr (rank == 2) {
    threej_mbars.reserve(4);
  } else {
    // Unneeded except for rank 2.
    (void)threej_mbars;
  }

  // threej_ones_sbars is the last 3J symbol in Eq. (32)
  // and the fourth 3J symbol in Eq. (34)
  // NOLINTNEXTLINE(misc-const-correctness)
  std::vector<WignerThreeJ> threej_ones_sbars;
  if constexpr (rank > 1) {
    threej_ones_sbars.reserve(9);
    for (int sb0 = -1; sb0 <= 1; ++sb0) {
      for (int sb1 = -1; sb1 <= 1; ++sb1) {
        threej_ones_sbars.emplace_back(1, sb0, 1, sb1);
      }
    }
  } else {
    // Unneeded for rank 1.
    (void)threej_ones_sbars;
  }

  for (size_t dest_comp_index = 0; dest_comp_index < num_independent_components;
       dest_comp_index++) {
    const auto dest_indices = tensor_index_list[dest_comp_index];
    const auto dest_bvs = helpers::to_cart_basis_vector(dest_indices);

    if constexpr (rank > 1) {
      // threej_pqs is (as a function of p and q) the second 3J symbol
      // in Eq. (22) for rank 2, and the first 3J symbol in Eq. (24) for
      // rank 3.  It is not used for rank 1.
      //
      // mbars are the values of mbar for rank 2 and 3, which are uniquely
      // determined by p and q because of the symmetries of threej_pqs.
      threej_pqs.clear();
      mbars.clear();
      for (int p = -1; p <= 1; p += 2) {
        for (int q = -1; q <= 1; q += 2) {
          mbars.push_back(helpers::bv_to_m(dest_bvs[rank - 2], p) +
                          helpers::bv_to_m(dest_bvs[rank - 1], q));
          threej_pqs.emplace_back(1, helpers::bv_to_m(dest_bvs[rank - 2], p), 1,
                                  helpers::bv_to_m(dest_bvs[rank - 1], q));
        }
      }
    }

    for (size_t src_comp_index = 0; src_comp_index < num_independent_components;
         src_comp_index++) {
      const auto src_indices = tensor_index_list[src_comp_index];
      const auto src_bvs = helpers::to_sphere_basis_vector(src_indices);
      const int sign_delta_sb =
          alg::accumulate(src_bvs, 1, [](const double acc, const auto bv) {
            return acc * (helpers::bv_to_s(bv) == -1 ? -1 : 1);
          });
      const int sign_delta_sb_conj =
          alg::accumulate(src_bvs, 1, [](const double acc, const auto bv) {
            return acc * (helpers::bv_to_s(bv) == 1 ? -1 : 1);
          });
      const size_t src_multiplicity =
          TensorStructure::multiplicity(src_comp_index);

      // scheck is used only in rank 1 and 3 but is easy to compute in general.
      const int scheck =
          -alg::accumulate(src_bvs, 0, [](const double acc, const auto bv) {
            return acc + helpers::bv_to_s(bv);
          });

      // sbar is used only for rank 2 and 3.
      // NOLINTNEXTLINE(misc-const-correctness)
      int sbar = std::numeric_limits<int>::min();
      if constexpr (rank > 1) {
        sbar = -(helpers::bv_to_s(src_bvs[rank - 1]) +
                 helpers::bv_to_s(src_bvs[rank - 2]));
      } else {
        (void)sbar;
      }

      if constexpr (rank == 3) {
        // threej_rs is (as a function of r, lbar, mbar) the penultimate
        // 3J symbol in Eq. (34) for rank 3.  It is not used for
        // ranks 1 and 2.
        threej_rs.clear();
        for (const int mbar : mbars) {
          for (int r = -1; r <= 1; r += 2) {
            const int mr = helpers::bv_to_m(dest_bvs[0], r);
            for (size_t lbar = 0; lbar <= 2; ++lbar) {
              if (abs(mbar) <= static_cast<int>(lbar)) {
                threej_rs.emplace_back(WignerThreeJ(1, mr, lbar, mbar));
              } else {
                threej_rs.emplace_back(std::nullopt);
              }
            }
          }
        }
        // threej_sb0s is the last 3J symbol in Eq. (34)
        threej_sb0s.clear();
        for (size_t lbar = 0; lbar <= 2; ++lbar) {
          if (static_cast<size_t>(abs(sbar)) <= lbar) {
            threej_sb0s.emplace_back(
                WignerThreeJ(1, helpers::bv_to_s(src_bvs[0]), lbar, -sbar));
          } else {
            threej_sb0s.emplace_back(std::nullopt);
          }
        }
      }

      for (iter_dest.reset(); iter_dest; ++iter_dest) {
        // In the docs, l_dest and m_dest are just called l and m.
        // l_src and m_src are called l' and m'.
        const auto l_dest = static_cast<size_t>(iter_dest.l());
        const auto m_dest = static_cast<int>(iter_dest.m());
        if constexpr (rank == 1) {
          (void)src_multiplicity;  // Unused variable for rank 1
          const int sign_m_dest = (m_dest % 2 == 0 ? 1 : -1);
          // threej_s is the first 3j symbol in Eq. (31)
          WignerThreeJ threej_s(l_dest, 0, 1, -scheck);
          for (int j = -1; j <= 1; j += 2) {
            const int mj = helpers::bv_to_m(dest_bvs[0], j);
            const std::complex<double> kj = helpers::bv_to_k(dest_bvs[0], j);
            // threej_m is the second 3j symbol in Eq. (31)
            WignerThreeJ threej_m(l_dest, -m_dest, 1, mj);
            inner_loops_one(filler, iter_src, iter_dest, src_comp_index,
                            dest_comp_index, ell_max, l_dest, m_dest, scheck,
                            mj, kj, threej_m, threej_s, sign_m_dest,
                            sign_delta_sb, sign_delta_sb_conj);
          }
        } else if constexpr (rank == 2) {
          (void)scheck;  // unused for rank 2
          const int sign_sbar = (sbar % 2 == 0 ? 1 : -1);
          for (size_t lbar = static_cast<size_t>(std::max(0, std::abs(sbar)));
               lbar <= 2; ++lbar) {
            const double symm_factor =
                helpers::get_symm_factor<typename TensorStructure::symmetry>(
                    src_multiplicity, lbar);
            if (symm_factor != 0.0) {
              const double threej_ones_sbar_val = threej_ones_sbars
                  [static_cast<size_t>(helpers::bv_to_s(src_bvs[1]) + 1) +
                   3 * static_cast<size_t>(helpers::bv_to_s(src_bvs[0]) + 1)](
                      lbar);
              // threej_mbars is the second 3J symbol in Eq. (32)
              threej_mbars.clear();
              for (const int mbar : mbars) {
                if (static_cast<size_t>(std::abs(mbar)) <= lbar) {
                  threej_mbars.emplace_back(
                      WignerThreeJ(l_dest, -m_dest, lbar, mbar));
                } else {
                  threej_mbars.emplace_back(std::nullopt);
                }
              }
              // threej_ells_sbar is the first 3J symbol in Eq. (32)
              WignerThreeJ threej_ells_sbar(l_dest, 0, lbar, -sbar);
              inner_loops_two(filler, iter_src, iter_dest, src_comp_index,
                              dest_comp_index, ell_max, l_dest, m_dest, lbar,
                              mbars, threej_mbars, threej_ells_sbar,
                              symm_factor, dest_bvs, threej_pqs,
                              threej_ones_sbar_val, sign_sbar, sign_delta_sb,
                              sign_delta_sb_conj);
            }
          }
        } else if constexpr (rank == 3) {
          const int sign_all_s = (-scheck % 2 == 0 ? 1 : -1);
          for (size_t lcheck =
                   static_cast<size_t>(std::max(0, std::abs(scheck)));
               lcheck <= 3; ++lcheck) {
            // threej_ssum is the second 3J symbol in Eq. (34)
            WignerThreeJ threej_ssum(lcheck, -scheck, l_dest, 0);
            for (int mcheck = -static_cast<int>(lcheck);
                 mcheck <= static_cast<int>(lcheck); ++mcheck) {
              // threej_msdc is the first 3J symbol in Eq. (34)
              WignerThreeJ threej_msdc(lcheck, mcheck, l_dest, -m_dest);
              inner_loops_three<typename TensorStructure::symmetry>(
                  filler, iter_src, iter_dest, src_comp_index, dest_comp_index,
                  ell_max, l_dest, m_dest, lcheck, mcheck, sbar, scheck,
                  threej_ones_sbars, threej_ssum, threej_msdc, threej_sb0s,
                  threej_pqs, threej_rs, mbars, src_bvs, dest_bvs,
                  src_multiplicity, sign_all_s, sign_delta_sb,
                  sign_delta_sb_conj);
            }
          }
        }
      }
    }
  }
  filler.fill(matrix);
}

// Explicit instantiations
#define TSTRUCT(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                             \
  template void fill_sphere_to_cart<typename TSTRUCT(data) < DataVector, \
                                    3>::structure >                      \
      (gsl::not_null<SimpleSparseMatrix*> matrix, size_t ell_max);       \
  template void fill_sphere_to_cart<typename TSTRUCT(data) < DataVector, \
                                    3>::structure >                      \
      (gsl::not_null<blaze::CompressedMatrix<double, blaze::rowMajor>*>  \
           matrix,                                                       \
       size_t ell_max);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (tnsr::i, tnsr::ii, tnsr::ij, tnsr::ijk, tnsr::ijj))

#undef INSTANTIATE
#undef TSTRUCT

}  // namespace ylm::TensorYlm
