/*!
 *  Copyright (c) 2016 by Contributors
 * \file choose_tensor.h
 * \brief support for complex hadamard product
 * \author Xingjian Shi
 */
#ifndef MSHADOW_EXTENSION_CHOOSE_TENSOR_H_
#define MSHADOW_EXTENSION_CHOOSE_TENSOR_H_
#include <algorithm>
#include "../extension.h"

namespace mshadow {
namespace expr {
/*!
 * \brief Choose tensor expression. Get the res = A[I] where A is the source 4D tensor and I is the index tensor
 * \tparam SrcExp type of source tensor expression, shape: (N, C, H, W)
 * \tparam IndExp type of index tensor expression, shape: (M, )
 * \tparam DType the content data type
 */
template<typename SrcExp, typename IndExp, typename DType>
struct ChooseTensorExp:
  public Exp<ChooseTensorExp<SrcExp, IndExp, DType>, DType, type::kChainer> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief index operand */
  const IndExp &ind_;
  /*! \brief source shape */
  Shape<4> src_shape_;
  /*! \brief ind batch number */
  Shape<1> ind_shape_;
  /*! \brief constructor */
  ChooseTensorExp(const SrcExp &src, const IndExp &ind)
      : src_(src), ind_(ind) {
    src_shape_ = ShapeCheck<4, SrcExp>::Check(src_);
    ind_shape_ = ShapeCheck<1, IndExp>::Check(ind_);
  }
};


template<int dim, typename SrcExp, typename IndExp, typename DType>
struct ShapeCheck<dim, ChooseTensorExp<SrcExp, IndExp, DType> > {
  inline static Shape<dim>
    Check(const ChooseTensorExp<SrcExp, IndExp, DType> &t) {
    CHECK(dim == 4)
      << "ChooseTensorExp: Dimension of the dst tensor must be 4.";
    Shape<4> src_shape = ShapeCheck<4, SrcExp>::Check(t.src_);
    Shape<1> ind_shape = ShapeCheck<1, IndExp>::Check(t.ind_);
    Shape<4> ret = src_shape;
    ret[0] = ind_shape[0];
    return ret;
  }
};

/*!
 * \brief Choose tensor expression. Get the res = A[I] where A is the source 4D tensor and I is the index tensor
 * \param src left source, shape: (N, C, H, W)
 * \param ind left source, shape: (M, )
 * \tparam e1 type of source expression
 * \tparam e2 type of roi expression
 */
template<typename SrcExp, typename SDType, typename IndExp, typename IDType, int e1, int e2>
inline ChooseTensorExp<SrcExp, IndExp, SDType>
choose_tensor(const Exp<SrcExp, SDType, e1> &src, const Exp<IndExp, IDType, e2> &ind) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim == 4 && ExpInfo<IndExp>::kDim == 1>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return ChooseTensorExp<SrcExp, IndExp, SDType>(src.self(), ind.self());
}


//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename IndExp, typename DType>
struct Plan<ChooseTensorExp<SrcExp, IndExp, DType>, DType> {
 public:
   explicit Plan(const ChooseTensorExp<SrcExp, IndExp, DType> &e)
     : src_(MakePlan(e.src_)), ind_(MakePlan(e.ind_)), src_batch_(e.src_shape_[0]),
     src_channel_(e.src_shape_[1]), src_height_(e.src_shape_[2]), src_width_(e.src_shape_[3]),
     ind_batch_(e.ind_shape_[0]){}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    DType res = 0;
    const index_t y = i % src_height_;
    i /= src_height_;
    const index_t c = i % src_channel_;
    const index_t b = i / src_channel_;
    const index_t b_src = ind_.Eval(0, b);
    return src_.Eval((b_src * src_channel_ + c) * src_height_ + y, j);
  }

 private:
  expr::Plan<SrcExp, DType> src_;
  expr::Plan<IndExp, DType> ind_;
  const index_t src_batch_;
  const index_t src_channel_;
  const index_t src_height_;
  const index_t src_width_;
  const index_t ind_batch_;
};

template<typename SrcExp, typename IndExp, typename DType>
inline Plan<ChooseTensorExp<SrcExp, IndExp, DType>, DType>
MakePlan(const ChooseTensorExp<SrcExp, IndExp, DType> &exp) {
  return Plan<ChooseTensorExp<SrcExp, IndExp, DType>, DType>(exp);
}


template<typename SrcExp, typename IndExp, typename DType>
struct ExpInfo<ChooseTensorExp<SrcExp, IndExp, DType> > {
  static const int kDim = 4;
  static const int kDevMask = ExpInfo<SrcExp>::kDevMask & ExpInfo<IndExp>::kDevMask;
};

}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_CHOOSE_TENSOR_H_
