# libtorch_test
## 编译方法

注意修改CMakeLists.txt中的
```CMake
find_package(Torch REQUIRED PATHS "/home/handsome/lib/libtorch2.6.0/libtorch")
```
改为自己的libtorch路径

```shell
cmake -B build
cmake --build build
```

