# HandeyeDemo
### C++编译流程：
1. 修改CMakeLists.txt中的<code>HALCON_INSTALL_DIR</code>,根据你的实际安装路径进行修改;
2. 运行下面命令行创建及编译：
```bash
mkdir build
cd build
cmake ..
make
```

3. 运行编译程序
```bash
cd ../bin

#   生成标定板
./GenCaltabDemo

#   手眼标定
./HandToEyeDemo
```

### halcon使用流程
1. 标定板生成
打开halcon导入，<code>Pattern_Generator.hdev</code>运行即可。

2. 手眼标定
打开halcon导入，<code>nine_point_calibration.hdev</code>运行即可。

### 说明
1. 可以更改程序中的识别图像，修改文件路径和文件名即可。

2. [九点标定参考链接](https://zhuanlan.zhihu.com/p/391938754)

