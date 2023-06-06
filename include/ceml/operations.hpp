#pragma once
#include "ceml/tensor.hpp"
#include <cstddef>
#include <cstdint>

namespace ceml {

// Element-wise computations

template <class T, class E1, class E2, class F> struct BinaryOp {
  E1 &left;
  E2 &right;
  F f;

  BinaryOp( E1 &left, E2 &right, F func)
      : left(left), right(right), f(func){};

  T operator()(const std::size_t i, const std::size_t j, const std::size_t k ) {
    
    if(m_force_materialise && m_is_materialised){
        return m_result(i,j,k);
    }else if (m_force_materialise) {
        materialise();
        return m_result(i,j,k);
        
    }else{
    
     return f(left, right, i, j, k); 
     
    }
     };
  static constexpr std::size_t size1 = F::get_size1(E1::size1, E2::size1);
  static constexpr std::size_t size2 = F::get_size2(E1::size2, E2::size2);
  static constexpr std::size_t size3 = F::get_size3(E1::size3, E2::size3);
  
  Tensor<T, size1, size2, size3> materialise(){
      if(m_is_materialised){
          return m_result;
      }
        m_result.allocate();
    m_is_materialised = true;
    for(std::size_t i=0; i<size1;i++){
    for(std::size_t j=0; j<size2;j++){
    for(std::size_t k=0; k<size3;k++){

        m_result.assign(i,j,k, f(left, right, i, j, k));
    }
    }
    }
    return m_result;
  }

  private:
    Tensor<T, size1, size2, size3> m_result;
    bool m_is_materialised = false;
    static constexpr bool m_force_materialise = F::must_materialise;
};

template <typename T, class E1, class E2> struct AddOpCPU {
  T operator()(E1& left, E2& right, const std::size_t i, const std::size_t j, const std::size_t k) { 
    
    return left(i,j,k) + right(i,j,k); }

  static constexpr std::size_t get_size1(const std::size_t lhs_size1,
                                         const std::size_t rhs_size1) {
    return lhs_size1;
  }
  static constexpr std::size_t get_size2(const std::size_t lhs_size2,
                                         const std::size_t rhs_size2) {
    return lhs_size2;
  }
  static constexpr std::size_t get_size3(const std::size_t lhs_size3,
                                         const std::size_t rhs_size3) {
    return lhs_size3;
  }
  
  static constexpr bool must_materialise = false;
};

template <typename T> struct MultOpCPU {
  T operator()(T left, T right, const std::size_t i, const std::size_t j, const std::size_t k) { 
    
    return left(i,j,k) + right(i,j,k); }


  static constexpr std::size_t get_size1(const std::size_t lhs_size1,
                                         const std::size_t rhs_size1) {
    return lhs_size1;
  }
  static constexpr std::size_t get_size2(const std::size_t lhs_size2,
                                         const std::size_t rhs_size2) {
    return lhs_size2;
  }
  static constexpr std::size_t get_size3(const std::size_t lhs_size3,
                                         const std::size_t rhs_size3) {
    return lhs_size3;
  }

  static constexpr bool must_materialise = false;
};

template <typename T, class E1, class E2>
BinaryOp<T, E1, E2, AddOpCPU<T, E1, E2>> add(E1 &left, E2 &right) {

  return BinaryOp<T, E1, E2, AddOpCPU<T, E1, E2>>(left, right, AddOpCPU<T, E1, E2>());
}

template <typename T, class E1, class E2>
BinaryOp<T, E1, E2, AddOpCPU<T, E1, E2>> add(E1 &&left, E2 &right) {

  return BinaryOp<T, E1, E2, AddOpCPU<T, E1, E2>>(left, right, AddOpCPU<T, E1, E2>());
}

template <typename T, class E1, class E2>
BinaryOp<T, E1, E2, AddOpCPU<T, E1, E2>> add(E1 &left, E2 &&right) {

  return BinaryOp<T, E1, E2, AddOpCPU<T, E1, E2>>(left, right, AddOpCPU<T, E1, E2>());
}


template <typename T, class E1, class E2>
BinaryOp<T, E1, E2, MultOpCPU<T>> ewiseMult(E1 &left, E2 &right) {

  return BinaryOp<T, E1, E2, MultOpCPU<T>>(left, right, MultOpCPU<T>());
}

// Linear Algebra
// 




// Reductions

template <class T, class E1, class F> struct ApplyDim1{
    E1& left;
    F f;
ApplyDim1(E1& left, F func): left(left), f(func){};

  static constexpr std::size_t size1 = F::get_size(E1::size1) ;
  static constexpr std::size_t size2 = E1::size2;
  static constexpr std::size_t size3 = E1::size3;

} 

} // namespace ceml