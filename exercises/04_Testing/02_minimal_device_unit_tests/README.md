# Building instructions

The minimal device unit test uses [Catch2](https://github.com/catchorg/Catch2). The provided CMake configuration uses a system installation when available and otherwise downloads and builds Catch2 automatically.

```bash
cmake -B build
cmake --build build
./build/test
```
