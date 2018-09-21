// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Common code for classes DataVector and ModalVector

#pragma once


/**
 * \ingroup DataStructuresGroup
 * \brief Here be code common to container classes DataVector and ModalVector.
 *
 * \details DataVector is intended to contain function values on the
 * computational domain. It holds an array of contiguous data and supports a
 * variety of mathematical operations that are applicable to nodal coefficients.
 * ModalVector, on the other hand, is intended to contain values of spectral
 * coefficients for any quantity expanded in its respective bases. It also
 * holds an array of contiguous data, but allows only limited math operations,
 * i.e. only those that are applicable to spectral coefficients, such as
 * elementwise addition, subtraction, multiplication, division and a few more.
 *
 * Therefore, both classes have significantly common structure and properties.
 * This file contains code (in the form of macros) that is used to build both
 * {Data,Modal}Vector classes.
 *
 * A VECTYPE class (below) holds an array of contiguous data. VECTYPE can be
 * owning, meaning the array is deleted when the VECTYPE goes out of scope, or
 * non-owning, meaning it just has a pointer to an array.
 */
#define MAKE_EXPRESSION_DATA_MODAL_VECTOR_CLASSES(VECTYPE)                     \
class VECTYPE /* NOLINT */                                                     \
    : public PointerVector<double, blaze::unaligned, blaze::unpadded,          \
                           blaze::defaultTransposeFlag, VECTYPE> { /* NOLINT */\
  /** \cond HIDDEN_SYMBOLS */                                                  \
  static constexpr void private_asserts() noexcept { /* NOLINTNEXTLINE */      \
    static_assert(std::is_nothrow_move_constructible<VECTYPE>::value,          \
                  "Missing move semantics");                                   \
  }                                                                            \
  /** \endcond */                                                              \
 public:                                                                       \
  using value_type = double;                                                   \
  using allocator_type = std::allocator<value_type>;                           \
  using size_type = size_t;                                                    \
  using difference_type = std::ptrdiff_t;                                      \
  using BaseType = PointerVector<double, blaze::unaligned, blaze::unpadded,    \
                                 blaze::defaultTransposeFlag,                  \
                                 VECTYPE>; /* NOLINT */                        \
  static constexpr bool transpose_flag = blaze::defaultTransposeFlag;          \
                                                                               \
  using BaseType::ElementType;                                                 \
  using TransposeType = VECTYPE; /* NOLINT */                                  \
  using CompositeType = const VECTYPE&; /* NOLINT */                           \
                                                                               \
  using BaseType::operator[];                                                  \
  using BaseType::begin;                                                       \
  using BaseType::cbegin;                                                      \
  using BaseType::cend;                                                        \
  using BaseType::data;                                                        \
  using BaseType::end;                                                         \
  using BaseType::size;                                                        \
                                                                               \
  /** @{ */                                                                    \
  /* Upcast to `BaseType` */                                                   \
  const BaseType& operator~() const noexcept {                                 \
    return static_cast<const BaseType&>(*this);                                \
  }                                                                            \
  BaseType& operator~() noexcept { return static_cast<BaseType&>(*this); }     \
  /** @} */                                                                    \
                                                                               \
  /** Create with the given size and value. */                                 \
  /** */                                                                       \
  /** \param size number of values */                                          \
  /** \param value the value to initialize each element. */                    \
  explicit VECTYPE( /* NOLINT */                                               \
      size_t size,                                                             \
      double value = std::numeric_limits<double>::signaling_NaN()) noexcept;   \
                                                                               \
  /** Create a non-owning VECTYPE that points to `start` */                    \
  VECTYPE(double* start, size_t size) noexcept; /* NOLINT */                   \
                                                                               \
  /** Create from an initializer list of doubles. All elements in the */       \
  /** `std::initializer_list` must have decimal points */                      \
  template <class T, Requires<cpp17::is_same_v<T, double>> = nullptr>          \
  VECTYPE(std::initializer_list<T> list) noexcept; /* NOLINT */                \
                                                                               \
  /** Empty VECTYPE */                                                         \
  VECTYPE() noexcept = default; /* NOLINT */                                   \
  /** \cond HIDDEN_SYMBOLS */                                                  \
  ~VECTYPE() = default; /* NOLINT */                                           \
                                                                               \
  VECTYPE(const VECTYPE& rhs); /* NOLINT */                                    \
  VECTYPE(VECTYPE&& rhs) noexcept; /* NOLINT */                                \
  VECTYPE& operator=(const VECTYPE& rhs); /* NOLINT */                         \
  VECTYPE& operator=(VECTYPE&& rhs) noexcept; /* NOLINT */                     \
                                                                               \
  /* This is a converting constructor. clang-tidy complains that it's not */   \
  /* explicit, but we want it to allow conversion.                        */   \
  /* clang-tidy: mark as explicit (we want conversion to VECTYPE)      */      \
  template <typename VT, bool VF>                                              \
  VECTYPE(const blaze::DenseVector<VT, VF>& expression) noexcept; /* NOLINT */ \
                                                                               \
  template <typename VT, bool VF>  /* NOLINTNEXTLINE */                        \
  VECTYPE& operator=(const blaze::DenseVector<VT, VF>& expression) noexcept;   \
  /** \endcond */                                                              \
                                                                               \
  MAKE_EXPRESSION_MATH_ASSIGN_PV(+=, VECTYPE) /* NOLINT */                     \
  MAKE_EXPRESSION_MATH_ASSIGN_PV(-=, VECTYPE) /* NOLINT */                     \
  MAKE_EXPRESSION_MATH_ASSIGN_PV(*=, VECTYPE) /* NOLINT */                     \
  MAKE_EXPRESSION_MATH_ASSIGN_PV(/=, VECTYPE) /* NOLINT */                     \
                                                                               \
  VECTYPE& operator=(const double& rhs) noexcept { /* NOLINT */                \
    ~*this = rhs;                                                              \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  /** @{ */                                                                    \
  /** Set the VECTYPE to be a reference to another VECTYPE object */           \
  void set_data_ref(gsl::not_null<VECTYPE*> rhs) noexcept { /* NOLINT */       \
    set_data_ref(rhs->data(), rhs->size());                                    \
  }                                                                            \
  void set_data_ref(double* start, size_t size) noexcept {                     \
    owned_data_ = decltype(owned_data_){};                                     \
    (~*this).reset(start, size);                                               \
    owning_ = false;                                                           \
  }                                                                            \
  /** @} */                                                                    \
                                                                               \
  /** Returns true if the class owns the data */                               \
  bool is_owning() const noexcept { return owning_; }                          \
                                                                               \
  /** Serialization for Charm++ */                                             \
  /* clang-tidy: google-runtime-references */                                  \
  void pup(PUP::er& p) noexcept; /* NOLINT */                                  \
                                                                               \
 private:                                                                      \
  SPECTRE_ALWAYS_INLINE void reset_pointer_vector() noexcept {                 \
    reset(owned_data_.data(), owned_data_.size());                             \
  }                                                                            \
                                                                               \
  /** \cond HIDDEN_SYMBOLS */                                                  \
  std::vector<double, allocator_type> owned_data_;                             \
  bool owning_{true};                                                          \
  /** \endcond */                                                              \
};


/**
 * Declare left-shift, equivalence, and inequivalence operations for VECTYPE
 * with itself
 */
#define MAKE_EXPRESSION_VECMATH_OP_COMP_SELF(VECTYPE)                          \
/**Output operator for VECTYPE */                                              \
std::ostream& operator<<(std::ostream& os, const VECTYPE& d); /* NOLINT */     \
                                                                               \
/** Equivalence operator for VECTYPE */                                        \
bool operator==(const VECTYPE& lhs, const VECTYPE& rhs) noexcept; /* NOLINT */ \
                                                                               \
/** Inequivalence operator for VECTYPE */                                      \
bool operator!=(const VECTYPE& lhs, const VECTYPE& rhs) noexcept; /* NOLINT */


/**
 * Define equivalence, and inequivalence operations for VECTYPE
 * with blaze::DenseVector<VT, VF>
 */
/// \cond
#define MAKE_EXPRESSION_VECMATH_OP_COMP_DV(VECTYPE)               \
/* Used for comparing VECTYPE to an expression */                 \
template <typename VT, bool VF>                                   \
bool operator==(const VECTYPE& lhs, /* NOLINT */                  \
                const blaze::DenseVector<VT, VF>& rhs) noexcept { \
  return lhs == VECTYPE(rhs); /* NOLINT */                        \
}                                                                 \
                                                                  \
template <typename VT, bool VF>                                   \
bool operator!=(const VECTYPE& lhs, /* NOLINT */                  \
                const blaze::DenseVector<VT, VF>& rhs) noexcept { \
  return not(lhs == rhs);                                         \
}                                                                 \
                                                                  \
template <typename VT, bool VF>                                   \
bool operator==(const blaze::DenseVector<VT, VF>& lhs,            \
                const VECTYPE& rhs) noexcept { /* NOLINT */       \
  return VECTYPE(lhs) == rhs; /* NOLINT */                        \
}                                                                 \
                                                                  \
template <typename VT, bool VF>                                   \
bool operator!=(const blaze::DenseVector<VT, VF>& lhs,            \
                const VECTYPE& rhs) noexcept { /* NOLINT */       \
  return not(lhs == rhs);                                         \
}
/// \endcond

/**
 * Specialize the Blaze type traits (Add,Sub,Mult,Div) to handle VECTYPE
 * correctly.
 */
#define MAKE_EXPRESSION_VECMATH_SPECIALIZE_BLAZE_ARITHMETIC_TRAITS(VECTYPE) \
namespace blaze {                                                           \
template <>                                                                 \
struct IsVector<VECTYPE> : std::true_type {}; /* NOLINT */                  \
                                                                            \
template <>                                                                 \
struct TransposeFlag<VECTYPE> : BoolConstant< /* NOLINT */                  \
                VECTYPE::transpose_flag> {}; /* NOLINT */                   \
                                                                            \
template <>                                                                 \
struct AddTrait<VECTYPE, VECTYPE> { /* NOLINT */                            \
  using Type = VECTYPE; /* NOLINT */                                        \
};                                                                          \
                                                                            \
template <>                                                                 \
struct AddTrait<VECTYPE, double> { /* NOLINT */                             \
  using Type = VECTYPE; /* NOLINT */                                        \
};                                                                          \
                                                                            \
template <>                                                                 \
struct AddTrait<double, VECTYPE> { /* NOLINT */                             \
  using Type = VECTYPE; /* NOLINT */                                        \
};                                                                          \
                                                                            \
template <>                                                                 \
struct SubTrait<VECTYPE, VECTYPE> { /* NOLINT */                            \
  using Type = VECTYPE; /* NOLINT */                                        \
};                                                                          \
                                                                            \
template <>                                                                 \
struct SubTrait<VECTYPE, double> { /* NOLINT */                             \
  using Type = VECTYPE; /* NOLINT */                                        \
};                                                                          \
                                                                            \
template <>                                                                 \
struct SubTrait<double, VECTYPE> { /* NOLINT */                             \
  using Type = VECTYPE; /* NOLINT */                                        \
};                                                                          \
                                                                            \
template <>                                                                 \
struct MultTrait<VECTYPE, VECTYPE> { /* NOLINT */                           \
  using Type = VECTYPE; /* NOLINT */                                        \
};                                                                          \
                                                                            \
template <>                                                                 \
struct MultTrait<VECTYPE, double> { /* NOLINT */                            \
  using Type = VECTYPE; /* NOLINT */                                        \
};                                                                          \
                                                                            \
template <>                                                                 \
struct MultTrait<double, VECTYPE> { /* NOLINT */                            \
  using Type = VECTYPE; /* NOLINT */                                        \
};                                                                          \
                                                                            \
template <>                                                                 \
struct DivTrait<VECTYPE, VECTYPE> { /* NOLINT */                            \
  using Type = VECTYPE; /* NOLINT */                                        \
};                                                                          \
                                                                            \
template <>                                                                 \
struct DivTrait<VECTYPE, double> { /* NOLINT */                             \
  using Type = VECTYPE; /* NOLINT */                                        \
};                                                                          \
} /* namespace blaze*/


/**
 * Specialize the Blaze Map traits to correctly handle VECTYPE
 */
#define MAKE_EXPRESSION_VECMATH_SPECIALIZE_BLAZE_MAP_TRAITS(VECTYPE) \
namespace blaze {                                                    \
template <typename Operator>                                         \
struct UnaryMapTrait<VECTYPE, Operator> { /* NOLINT */               \
  using Type = VECTYPE; /* NOLINT */                                 \
};                                                                   \
                                                                     \
template <typename Operator>                                         \
struct BinaryMapTrait<VECTYPE, VECTYPE, Operator> { /* NOLINT */     \
  using Type = VECTYPE; /* NOLINT */                                 \
};                                                                   \
}  /* namespace blaze */


/**
 * Define + and += operations for std::arrays of VECTYPE's
 */
#define MAKE_EXPRESSION_VECMATH_OP_ADD_ARRAYS_OF_VEC(VECTYPE)                \
template <typename T, size_t Dim>                                            \
std::array<VECTYPE, Dim> operator+( /* NOLINT */                             \
    const std::array<T, Dim>& lhs,                                           \
    const std::array<VECTYPE, Dim>& rhs) noexcept { /* NOLINT */             \
  std::array<VECTYPE, Dim> result; /* NOLINT */                              \
  for (size_t i = 0; i < Dim; i++) {                                         \
    gsl::at(result, i) = gsl::at(lhs, i) + gsl::at(rhs, i);                  \
  }                                                                          \
  return result;                                                             \
}                                                                            \
template <typename U, size_t Dim>                                            \
std::array<VECTYPE, Dim> operator+( /* NOLINT */                             \
    const std::array<VECTYPE, Dim>& lhs, /* NOLINT */                        \
    const std::array<U, Dim>& rhs) noexcept {                                \
  return rhs + lhs;                                                          \
}                                                                            \
template <size_t Dim>                                                        \
std::array<VECTYPE, Dim> operator+( /* NOLINT */                             \
    const std::array<VECTYPE, Dim>& lhs, /* NOLINT */                        \
    const std::array<VECTYPE, Dim>& rhs) noexcept { /* NOLINT */             \
  std::array<VECTYPE, Dim> result; /* NOLINT */                              \
  for (size_t i = 0; i < Dim; i++) {                                         \
    gsl::at(result, i) = gsl::at(lhs, i) + gsl::at(rhs, i);                  \
  }                                                                          \
  return result;                                                             \
}                                                                            \
template <size_t Dim>                                                        \
std::array<VECTYPE, Dim>& operator+=( /* NOLINT */                           \
    std::array<VECTYPE, Dim>& lhs, /* NOLINT */                              \
    const std::array<VECTYPE, Dim>& rhs) noexcept { /* NOLINT */             \
  for (size_t i = 0; i < Dim; i++) {                                         \
    gsl::at(lhs, i) += gsl::at(rhs, i);                                      \
  }                                                                          \
  return lhs;                                                                \
}

/**
 * Define - and -= operations for std::arrays of VECTYPE's
 */
#define MAKE_EXPRESSION_VECMATH_OP_SUB_ARRAYS_OF_VEC(VECTYPE)                \
template <typename T, size_t Dim>                                            \
std::array<VECTYPE, Dim> operator-( /* NOLINT */                             \
    const std::array<T, Dim>& lhs,                                           \
    const std::array<VECTYPE, Dim>& rhs) noexcept { /* NOLINT */             \
  std::array<VECTYPE, Dim> result; /* NOLINT */                              \
  for (size_t i = 0; i < Dim; i++) {                                         \
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);                  \
  }                                                                          \
  return result;                                                             \
}                                                                            \
template <typename U, size_t Dim>                                            \
std::array<VECTYPE, Dim> operator-( /* NOLINT */                             \
    const std::array<VECTYPE, Dim>& lhs, /* NOLINT */                        \
    const std::array<U, Dim>& rhs) noexcept {                                \
  std::array<VECTYPE, Dim> result; /* NOLINT */                              \
  for (size_t i = 0; i < Dim; i++) {                                         \
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);                  \
  }                                                                          \
  return result;                                                             \
}                                                                            \
template <size_t Dim>                                                        \
std::array<VECTYPE, Dim> operator-( /* NOLINT */                             \
    const std::array<VECTYPE, Dim>& lhs, /* NOLINT */                        \
    const std::array<VECTYPE, Dim>& rhs) noexcept { /* NOLINT */             \
  std::array<VECTYPE, Dim> result; /* NOLINT */                              \
  for (size_t i = 0; i < Dim; i++) {                                         \
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);                  \
  }                                                                          \
  return result;                                                             \
}                                                                            \
template <size_t Dim>                                                        \
std::array<VECTYPE, Dim>& operator-=( /* NOLINT */                           \
    std::array<VECTYPE, Dim>& lhs, /* NOLINT */                              \
    const std::array<VECTYPE, Dim>& rhs) noexcept { /* NOLINT */             \
  for (size_t i = 0; i < Dim; i++) {                                         \
    gsl::at(lhs, i) -= gsl::at(rhs, i);                                      \
  }                                                                          \
  return lhs;                                                                \
}


/**
 * Forbid assignment of blaze::DenseVector<VT,VF>'s to VECTYPE, if the result
 * type VT::ResultType is not VECTYPE
 */
#define MAKE_EXPRESSION_VEC_OP_ASSIGNMENT_RESTRICT_TYPE(VECTYPE)               \
template <typename VT, bool VF> /* NOLINTNEXTLINE */                         \
VECTYPE::VECTYPE(const blaze::DenseVector<VT, VF>& expression) noexcept        \
    : owned_data_((~expression).size()) {                                      \
  static_assert(cpp17::is_same_v<typename VT::ResultType,VECTYPE>, /* NOLINT */\
              "You are attempting to assign the result of an expression that " \
              "is not a " #VECTYPE " to a " #VECTYPE "."); /* NOLINT */        \
  reset_pointer_vector();                                                      \
  ~*this = expression;                                                         \
}                                                                              \
                                                                               \
template <typename VT, bool VF>                                                \
VECTYPE& VECTYPE::operator=( /* NOLINT */                                      \
    const blaze::DenseVector<VT, VF>& expression) noexcept {                   \
  static_assert(cpp17::is_same_v<typename VT::ResultType,VECTYPE>, /* NOLINT */\
              "You are attempting to assign the result of an expression that " \
              "is not a " #VECTYPE " to a " #VECTYPE "."); /* NOLINT */        \
  if (owning_ and (~expression).size() != size()) {                            \
    owned_data_.resize((~expression).size());                                  \
    reset_pointer_vector();                                                    \
  } else if (not owning_) {                                                    \
    ASSERT((~expression).size() == size(), "Must copy into same size, not "    \
                                               << (~expression).size()         \
                                               << " into " << size());         \
  }                                                                            \
  ~*this = expression;                                                         \
  return *this;                                                                \
}


#define MAKE_EXPRESSION_VEC_OP_MAKE_WITH_VALUE(VECTYPE)                     \
namespace MakeWithValueImpls {                                              \
/** \brief Returns a VECTYPE the same size as `input`, with each element */ \
/** equal to `value`. */                                                    \
template <>                                                                 \
SPECTRE_ALWAYS_INLINE VECTYPE /* NOLINT */                                  \
MakeWithValueImpl<VECTYPE,VECTYPE>::apply(const VECTYPE& input, /* NOLINT */\
                                           const double value) {            \
  return VECTYPE(input.size(), value); /* NOLINT */                         \
}                                                                           \
}  /* namespace MakeWithValueImpls*/



/**                 Function definitions               */

/**
 * Construct VECTYPE with value(s)
 */
#define MAKE_EXPRESSION_VEC_DEF_CONSTRUCT_WITH_VALUE(VECTYPE)                \
VECTYPE::VECTYPE(const size_t size, const double value) noexcept /* NOLINT */\
    : owned_data_(size, value) {                                             \
  reset_pointer_vector();                                                    \
}                                                                            \
                                                                             \
VECTYPE::VECTYPE(double* start, size_t size) noexcept /* NOLINT */           \
    : BaseType(start, size), owned_data_(0), owning_(false) {}               \
                                                                             \
template <class T, Requires<cpp17::is_same_v<T, double>>>                    \
VECTYPE::VECTYPE(std::initializer_list<T> list) noexcept /* NOLINT */        \
    : owned_data_(std::move(list)) {                                         \
  reset_pointer_vector();                                                    \
}

/**
 * Construct / Assign VECTYPE with / to VECTYPE reference or rvalue
 */
// clang-tidy: calling a base constructor other than the copy constructor.
//             We reset the base class in reset_pointer_vector after calling its
//             default constructor
#define MAKE_EXPRESSION_VEC_DEF_CONSTRUCT_WITH_VEC(VECTYPE)                  \
VECTYPE::VECTYPE(const VECTYPE& rhs) : BaseType{} { /* NOLINT */             \
  if (rhs.is_owning()) {                                                     \
    owned_data_ = rhs.owned_data_;                                           \
  } else {                                                                   \
    owned_data_.assign(rhs.begin(), rhs.end());                              \
  }                                                                          \
  reset_pointer_vector();                                                    \
}                                                                            \
                                                                             \
VECTYPE& VECTYPE::operator=(const VECTYPE& rhs) { /* NOLINT */               \
  if (this != &rhs) {                                                        \
    if (owning_) {                                                           \
      if (rhs.is_owning()) {                                                 \
        owned_data_ = rhs.owned_data_;                                       \
      } else {                                                               \
        owned_data_.assign(rhs.begin(), rhs.end());                          \
      }                                                                      \
      reset_pointer_vector();                                                \
    } else {                                                                 \
      ASSERT(rhs.size() == size(), "Must copy into same size, not "          \
                                       << rhs.size() << " into " << size()); \
      std::copy(rhs.begin(), rhs.end(), begin());                            \
    }                                                                        \
  }                                                                          \
  return *this;                                                              \
}                                                                            \
                                                                             \
VECTYPE::VECTYPE(VECTYPE&& rhs) noexcept { /* NOLINT */                      \
  owned_data_ = std::move(rhs.owned_data_);                                  \
  ~*this = ~rhs;  /* PointerVector is trivially copyable */                  \
  owning_ = rhs.owning_;                                                     \
                                                                             \
  rhs.owning_ = true;                                                        \
  rhs.reset();                                                               \
}                                                                            \
                                                                             \
VECTYPE& VECTYPE::operator=(VECTYPE&& rhs) noexcept { /* NOLINT */           \
  if (this != &rhs) {                                                        \
    if (owning_) {                                                           \
      owned_data_ = std::move(rhs.owned_data_);                              \
      ~*this = ~rhs;  /* PointerVector is trivially copyable */              \
      owning_ = rhs.owning_;                                                 \
    } else {                                                                 \
      ASSERT(rhs.size() == size(), "Must copy into same size, not "          \
                                       << rhs.size() << " into " << size()); \
      std::copy(rhs.begin(), rhs.end(), begin());                            \
    }                                                                        \
    rhs.owning_ = true;                                                      \
    rhs.reset();                                                             \
  }                                                                          \
  return *this;                                                              \
}


/**
 * Charm++ packing / unpacking of object
 */
#define MAKE_EXPRESSION_VEC_OP_PUP_CHARM(VECTYPE)       \
void VECTYPE::pup(PUP::er& p) noexcept { /* NOLINT */   \
  auto my_size = size();                                \
  p | my_size;                                          \
  if (my_size > 0) {                                    \
    if (p.isUnpacking()) {                              \
      owning_ = true;                                   \
      owned_data_.resize(my_size);                      \
      reset_pointer_vector();                           \
    }                                                   \
    PUParray(p, data(), size());                        \
  }                                                     \
}


/**
 * Define left-shift, equivalence, and inequivalence operations for VECTYPE
 * with itself
 */
#define MAKE_EXPRESSION_VECMATH_OP_DEF_COMP_SELF(VECTYPE)                      \
/** Left-shift operator for VECTYPE */                                         \
std::ostream& operator<<(std::ostream& os, const VECTYPE& d) { /* NOLINT */    \
  sequence_print_helper(os, d.begin(), d.end());                     \
  return os;                                                                   \
}                                                                              \
                                                                               \
/** Equivalence operator for VECTYPE */                                        \
bool operator==(const VECTYPE& lhs, const VECTYPE& rhs) noexcept { /* NOLINT */\
  return lhs.size() == rhs.size() and                                          \
         std::equal(lhs.begin(), lhs.end(), rhs.begin());                      \
}                                                                              \
                                                                               \
/** Inequivalence operator for VECTYPE */                                      \
bool operator!=(const VECTYPE& lhs, const VECTYPE& rhs) noexcept { /* NOLINT */\
  return not(lhs == rhs);                                                      \
}
