#pragma once
#include <boost/core/demangle.hpp>
#include <boost/type_index.hpp>
#include "boost/lexical_cast.hpp"
#include "magic_enum.hpp"

#include <type_traits>
#include <typeinfo>

#include <iostream>
#include <fstream>

#include <array>
#include <string>

#include <fstream>
#include <concepts>
#include <ranges>

namespace boost
{

    template <typename T>
    requires(std::is_enum_v<T>)
        T lexical_cast(const std::string &val_str)
    {
        using namespace std;
        T t;
        auto val = magic_enum::enum_cast<T>(val_str);
        if (val.has_value())
        {
            // cout << boost::typeindex::type_id<decltype( val.value() )>().pretty_name() << endl;
            t = val.value();
        }
        else
        {
            cerr << "error cannot convert " << val_str << " to " << boost::typeindex::type_id<T>().pretty_name() << endl;
            throw bad_lexical_cast();
        }

        return t;
    }
}

namespace std
{
    /*
    template <typename T, auto N>
    // requires(is_enum_v<T>)
    ostream &operator<<(ostream &os, const std::array<T, N> &arr)
    {

        // os << magic_enum::enum_name(enm);
        for (auto i : arr)
        {
            os << i << " ";
        }

        return os;
    }*/
}

template <typename T>
requires(std::is_enum_v<T>)
    std::ostream &
    operator<<(std::ostream &os, const T &enm)
{
    // os << magic_enum::enum_name<enm>();
    os << magic_enum::enum_name(enm);
    return os;
}
