## 使用说明

先编译CTC解码文件decode
再运行predic进行测试
```bash
g++ -o decode.so -std=c++11 -shared -fPIC decode.cpp
python predict.py
```
注：在predict中修改对应路径(pretrained_path)即可，也可以在main中将预测改为批量预测