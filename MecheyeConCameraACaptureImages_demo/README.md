# MecheyeConCameraACaptureImages_demo
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

编译生成的文件在上一级目录的<code>bin</code>文件夹下，运行下面指令启动程序：
```
cd ../bin
./ConCameraACaptureImages_demo
```

### halcon使用流程
打开halcon导入，<code>connect_to_camera_and_capture_images.hdev</code>运行即可。

### 说明
可以更改程序中的识别图像，修改文件路径和文件名即可。


