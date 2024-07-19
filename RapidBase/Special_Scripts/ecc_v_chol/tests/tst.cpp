#include <iostream>
#include <bitset>


int main(){

    
    int x= 2;
    std::cout << std::bitset<3>(x) << std::endl;
    std::cout << "ffs: " << __builtin_ffs(x) <<" clz: " << __builtin_clz(x) << std::endl;

    x= 3;
    std::cout << std::bitset<3>(x) << std::endl;
    std::cout << "ffs: " << __builtin_ffs(x) <<" clz: " << __builtin_clz(x) << std::endl;

    x= 4;
    std::cout << std::bitset<3>(x) << std::endl;
    std::cout << "ffs: " <<__builtin_ffs(x) <<" clz: " << __builtin_clz(x) << std::endl;
}