# llama32.cpp
**llama.cpp** is a minimal, pure-C++ implementation of [Llama 3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
inference on CPU.


## Install and Run llama32.
```
git clone https://github.com/iangitonga/llama32.cpp.git
cd llama32.cpp/
g++ -std=c++17 -O3 -fopenmp llama32.cpp -o llama32
./llama32
```

If you have an Intel CPU, Sandy Bridge or newer, compile with the following line instead to achieve higher performance:

```g++ -std=c++17 -O3 -fopenmp -mavx -mf16c llama32.cpp -o llama32```

Run `./llama32 --help` to see all available options.
