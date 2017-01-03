/*!
 *  Copyright (c) 2016 by Contributors
 * \file complex_hadamard_product.h
 * \brief support for complex hadamard product
 * \author Xingjian Shi
 */
#ifndef MSHADOW_EXTENSION_COMPLEX_PRODUCT_H_
#define MSHADOW_EXTENSION_COMPLEX_PRODUCT_H_
#include <algorithm>
#include "../extension.h"

namespace mshadow {
namespace expr {
/*!
 * \brief Complex hadamard product expression. Calculate res = A .* B where A and B are all complex 4D tensors 
 * \tparam SrcExp type of source tensor expression, shape: (N, C, H, W*2)
 * \tparam DType the content data type
 */
template<typename Src1Exp, typename Src2Exp, typename DType>
struct ComplexHadamardProductExp:
  public Exp<ComplexHadamardProductExp<Src1Exp, Src2Exp, DType>, DType, type::kChainer> {
  /*! \brief left source operand */
  const Src1Exp &lsrc_;
  /*! \brief right source operand */
  const Src2Exp &rsrc_;
  /*! \brief constructor */
  ComplexHadamardProductExp(const Src1Exp &lsrc, const Src2Exp &rsrc)
      : lsrc_(lsrc), rsrc_(rsrc) {}
};


template<int dim, typename Src1Exp, typename Src2Exp, typename DType>
struct ShapeCheck<dim, ComplexHadamardProductExp<Src1Exp, Src2Exp, DType> > {
  inline static Shape<dim>
    Check(const ComplexHadamardProductExp<Src1Exp, Src2Exp, DType> &t) {
    CHECK(dim == 4)
      << "ComplexHadamardProductExp: Dimension of the src tensor must be 4.";
    Shape<4> lsrc_shape = ShapeCheck<4, Src1Exp>::Check(t.lsrc_);
    Shape<4> rsrc_shape = ShapeCheck<4, Src2Exp>::Check(t.rsrc_);
    CHECK_EQ(lsrc_shape, rsrc_shape)
      << "ComplexHadamardProductExp: Shape of the two inputs must be equal!";
    CHECK_EQ(lsrc_shape[3]%2, 0)
      << "ComplexHadamardProductExp: Size of the width channel must be even!";
    Shape<4> ret = lsrc_shape;
    return ret;
  }
};

/*!
 * \brief complex_hadamard_product Calculate res = A .* B where A and B are all complex 4D tensors 
 * \param lsrc left source, shape: (batch, channel, height, width*2)
 * \param rsrc left source, shape: (batch, channel, height, width*2)
 * \tparam e1 type of source expression
 * \tparam e2 type of roi expression
 */
template<typename Src1Exp, typename Src2Exp, typename DType, int e1, int e2>
inline ComplexHadamardProductExp<Src1Exp, Src2Exp, DType>
complex_hadamard_product(const Exp<Src1Exp, DType, e1> &lsrc, const Exp<Src2Exp, DType, e2> &rsrc) {
  TypeCheckPass<ExpInfo<Src1Exp>::kDim == 4 && ExpInfo<Src2Exp>::kDim == 4>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return ComplexHadamardProductExp<Src1Exp, Src2Exp, DType>(lsrc.self(), rsrc.self());
}


//----------------------
// Execution plan
//----------------------
template<typename Src1Exp, typename Src2Exp, typename DType>
struct Plan<ComplexHadamardProductExp<Src1Exp, Src2Exp, DType>, DType> {
 public:
   explicit Plan(const ComplexHadamardProductExp<Src1Exp, Src2Exp, DType> &e)
     : lsrc_(MakePlan(e.lsrc_)), rsrc_(MakePlan(e.rsrc_)) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    DType res = 0;
    const index_t base_y = static_cast<index_t>(j / 2) * 2;
    if (j % 2 == 0){
      res = lsrc_.Eval(i, base_y) * rsrc_.Eval(i, base_y) - lsrc_.Eval(i, base_y + 1) * rsrc_.Eval(i, base_y + 1);
    }
    else {
      res = lsrc_.Eval(i, base_y) * rsrc_.Eval(i, base_y + 1) + lsrc_.Eval(i, base_y + 1) * rsrc_.Eval(i, base_y);
    }
    return res;
  }

 private:
  expr::Plan<Src1Exp, DType> lsrc_;
  expr::Plan<Src2Exp, DType> rsrc_;
};

template<typename Src1Exp, typename Src2Exp, typename DType>
inline Plan<ComplexHadamardProductExp<Src1Exp, Src2Exp, DType>, DType>
MakePlan(const ComplexHadamardProductExp<Src1Exp, Src2Exp, DType> &exp) {
  return Plan<ComplexHadamardProductExp<Src1Exp, Src2Exp, DType>, DType>(exp);
}


template<typename Src1Exp, typename Src2Exp, typename DType>
struct ExpInfo<ComplexHadamardProductExp<Src1Exp, Src2Exp, DType> > {
  static const int kDim = 4;
  static const int kDevMask = ExpInfo<Src1Exp>::kDevMask & ExpInfo<Src2Exp>::kDevMask;
};

}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_COMPLEX_HADAMARD_PRODUCT_H_
