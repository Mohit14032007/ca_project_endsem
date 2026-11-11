# Build & Environment Setup Guide
### Prerequisites
- GCC >= 13.0 (C++17 standard)
- CMake >= 3.28
- Python >= 3.10 with pandas, numpy, matplotlib, pyyaml

### Build Commands
```bash
cd LLMSimulator
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```
