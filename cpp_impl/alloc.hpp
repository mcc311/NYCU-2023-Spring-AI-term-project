#include <iostream>

template<class T, std::size_t SZ>
class ContinuouslyAllocatorWithoutDestruct {
    using Byte = std::uint8_t;
public:
    ContinuouslyAllocatorWithoutDestruct() : alloc_counter{0} {
        // allocate memory without calling the constructor of T
        buffer = new Byte[SZ * sizeof(T)];
    }

    ~ContinuouslyAllocatorWithoutDestruct() {
        delete[] buffer;
    }

    template<class ...Args>
    T* allocate(Args&&... args) {
        if (alloc_counter >= SZ)
            return nullptr;
        // call T's constructor at the memory address buffer + byte_count
        return new (reinterpret_cast<T*>(buffer) + alloc_counter++) T(std::forward<Args>(args)...);
    }

    void reset() {
        alloc_counter = 0;
    }

private:
    Byte* buffer;
    size_t alloc_counter; // the number of the instance of T has already been allocated
};