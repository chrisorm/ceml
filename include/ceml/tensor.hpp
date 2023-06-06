#pragma once
#include <array>
#include <cstddef>
#include <vector>

namespace ceml{

    
    enum class DeviceType{
        CPU,
        GPU
    };

template<class T, int N1, int N2, int N3>
struct Tensor{
    DeviceType deviceType;
    bool is_param = false;
    
    static constexpr std::size_t n_elems = N1 * N2 * N3;
    static constexpr int size1 = N1;
    static constexpr int size2 = N2;
    static constexpr int size3 = N3;
    std::vector<T>* data;
    Tensor(){};
    
    void allocate(){
        data = new std::vector<T>(n_elems);
    }
    
    void assign(const std::size_t i, const std::size_t j, const std::size_t k, const T val){
       data->at(ijk_i(i,j,k)) = val; 
    }

    static constexpr std::size_t ijk_i(const std::size_t i, const std::size_t j, const std::size_t k) {
        return i + j*size1 + k*size1*size2; // TODO: CHECK THIS?
    }
    
    T operator()(const std::size_t i, const std::size_t j, const std::size_t k) const {
        return data->at(ijk_i(i,j,k));
    }

};


}