#include <iostream>
#include <string.h>

template<typename T>
class Sum {
public:
    Sum(T v1, T v2) {
        val1 = v1;
        val2 = v2;
    }

    void calculate() {
        result = val1 + val2;
    }

    void print() {
        std::cout << result << "\n";
    }

private:
    T val1;
    T val2;
    T result;
};

template<>
class Sum<std::string> {
public:
    Sum(std::string v1, std::string v2) {
        val1 = v1;
        val2 = v2;
    }

    void calculate() {
        result = val1 + "X" + val2;
    }

    void print() {
        std::cout << result << "\n";
    }

private:
    std::string val1;
    std::string val2;
    std::string result;
};

int main() {
    Sum<int> s = Sum<int>(1, 3);
    s.calculate();
    s.print();

    Sum<std::string> s_str = Sum<std::string>("a", "b");
    s_str.calculate();
    s_str.print();
}