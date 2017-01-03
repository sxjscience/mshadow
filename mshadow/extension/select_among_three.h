/*!
 *  Copyright (c) 2016 by Contributors
 * \file select_among_three.h
 * \brief support for selection among three variable
 * \author Xingjian Shi
 */
#ifndef MSHADOW_EXTENSION_SELECT_AMONG_THREE_H_
#define MSHADOW_EXTENSION_SELECT_AMONG_THREE_H_
#include <algorithm>
#include "../extension.h"

namespace mshadow {
namespace expr {
/*!
 * \brief Select the result tensor expression among three input tensors, i.e res = A or B or C. The selection is controlled by the flag.
 * \tparam SrcExp type of source tensor expression, shape: (N, C, H, W)
 * \tparam FlagExp type of the flag tensor expression, shape: (N,), each value is 0 or 1 or 2. 0 --> res = A, 1 --> res = B, 2 --> res = C 
 * \tparam DType the content data type
 */
template<typename Src1Exp, typename Src2Exp, typename Src3Exp, typename FlagExp, typename DType>
struct SelectAmongThreeExp:
  public Exp<SelectAmongThreeExp<Src1Exp, Src2Exp, Src3Exp, FlagExp, DType>, DType, type::kChainer> {
  /*! \brief the first source operand */
  const Src1Exp &src1_;
  /*! \brief the second source operand */
  const Src2Exp &src2_;
  /*! \brief the third source operand */
  const Src3Exp &src3_;
  /*! \brief flag operand */
  const FlagExp &flag_;
  /*! \brief source shape */
  Shape<4> src_shape_;
  /*! \brief ind batch number */
  Shape<1> flag_shape_;
  /*! \brief constructor */
  SelectAmongThreeExp(const Src1Exp &src1, const Src2Exp &src2, const Src3Exp &src3, const FlagExp &flag)
    : src1_(src1), src2_(src2), src3_(src3), flag_(flag) {
    src_shape_ = ShapeCheck<4, Src1Exp>::Check(src1_);
    flag_shape_ = ShapeCheck<1, FlagExp>::Check(flag_);
  }
};


template<int dim, typename Src1Exp, typename Src2Exp, typename Src3Exp, typename FlagExp, typename DType>
struct ShapeCheck<dim, SelectAmongThreeExp<Src1Exp, Src2Exp, Src3Exp, FlagExp, DType> > {
  inline static Shape<dim>
    Check(const SelectAmongThreeExp<Src1Exp, Src2Exp, Src3Exp, FlagExp, DType> &t) {
    CHECK(dim == 4)
      << "SelectAmongThreeExp: Dimension of the dst tensor must be 4.";
    Shape<4> src1_shape = ShapeCheck<4, Src1Exp>::Check(t.src1_);
    Shape<4> src2_shape = ShapeCheck<4, Src2Exp>::Check(t.src2_);
    Shape<4> src3_shape = ShapeCheck<4, Src3Exp>::Check(t.src3_);
    CHECK(src1_shape == src2_shape && src2_shape == src3_shape)
      << "SelectAmongThreeExp: Shape of the three source tensors must be the same.";
    Shape<1> flag_shape = ShapeCheck<1, FlagExp>::Check(t.flag_);
    CHECK_EQ(flag_shape[0], src1_shape[0])
      << "SelectAmongThreeExp: Batch size of the three source tensors and the flag tensor must be the same.";
    Shape<4> ret = src1_shape;
    return ret;
  }
};

/*!
 * \brief Select the result tensor expression among three input tensors, i.e res = A or B or C. The selection is controlled by the flag.
 * \param src1 the first source, shape: (N, C, H, W)
 * \param src2 the second source, shape: (N, C, H, W)
 * \param src3 the third source, shape: (N, C, H, W)
 * \param flag the flag, shape: (N, )
 * \tparam e1 type of source1 expression
 * \tparam e2 type of source2 expression
 * \tparam e3 type of source2 expression
 * \tparam e4 type of source2 expression
 */
template<typename Src1Exp, typename Src2Exp, typename Src3Exp, typename DType,
  typename FlagExp, typename FDType, int e1, int e2, int e3, int e4>
  inline SelectAmongThreeExp<Src1Exp, Src2Exp, Src3Exp, FlagExp, DType>
  select_among_three(const Exp<Src1Exp, DType, e1> &src1, const Exp<Src2Exp, DType, e2> &src2, const Exp<Src3Exp, DType, e3> &src3,
  const Exp<FlagExp, FDType, e4> &flag) {
  TypeCheckPass<ExpInfo<Src1Exp>::kDim == 4 && ExpInfo<Src2Exp>::kDim == 4 && ExpInfo<Src3Exp>::kDim == 4 && ExpInfo<FlagExp>::kDim == 1>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return SelectAmongThreeExp<Src1Exp, Src2Exp, Src3Exp, FlagExp, DType>(src1.self(), src2.self(), src3.self(), flag.self());
}


//----------------------
// Execution plan
//----------------------
template<typename Src1Exp, typename Src2Exp, typename Src3Exp, typename FlagExp, typename DType>
struct Plan<SelectAmongThreeExp<Src1Exp, Src2Exp, Src3Exp, FlagExp, DType>, DType> {
 public:
   explicit Plan(const SelectAmongThreeExp<Src1Exp, Src2Exp, Src3Exp, FlagExp, DType> &e)
     : src1_(MakePlan(e.src1_)), src2_(MakePlan(e.src2_)), src3_(MakePlan(e.src3_)), flag_(MakePlan(e.flag_)),
     src_batch_(e.src_shape_[0]), src_channel_(e.src_shape_[1]), src_height_(e.src_shape_[2]), src_width_(e.src_shape_[3]){}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    DType res = 0;
    const index_t b = (i / src_height_) / src_channel_;
    const index_t l = flag_.Eval(0, b);
    if (l == 0) {
      res = src1_.Eval(i, j);
    } 
    else if (l == 1){
      res = src2_.Eval(i, j);
    } 
    else {
      res = src3_.Eval(i, j);
    }
    return res;
  }

 private:
  expr::Plan<Src1Exp, DType> src1_;
  expr::Plan<Src2Exp, DType> src2_;
  expr::Plan<Src3Exp, DType> src3_;
  expr::Plan<FlagExp, DType> flag_;
  const index_t src_batch_;
  const index_t src_channel_;
  const index_t src_height_;
  const index_t src_width_;
};

template<typename Src1Exp, typename Src2Exp, typename Src3Exp, typename FlagExp, typename DType>
inline Plan<SelectAmongThreeExp<Src1Exp, Src2Exp, Src3Exp, FlagExp, DType>, DType>
MakePlan(const SelectAmongThreeExp<Src1Exp, Src2Exp, Src3Exp, FlagExp, DType> &exp) {
  return Plan<SelectAmongThreeExp<Src1Exp, Src2Exp, Src3Exp, FlagExp, DType>, DType>(exp);
}


template<typename Src1Exp, typename Src2Exp, typename Src3Exp, typename FlagExp, typename DType>
struct ExpInfo<SelectAmongThreeExp<Src1Exp, Src2Exp, Src3Exp, FlagExp, DType> > {
  static const int kDim = 4;
  static const int kDevMask = ExpInfo<Src1Exp>::kDevMask & ExpInfo<Src2Exp>::kDevMask & ExpInfo<Src3Exp>::kDevMask & ExpInfo<FlagExp>::kDevMask;
};

}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SELECT_AMONG_THREE_H_
