# SetLaserFrameRange Sample Program

With this sample program, you can set the projection range of the structured light. The entire projector FOV is from 0 to 100.

> Note: This sample program can only be used with Laser, LSR and DEEP series cameras.

If you have any questions or have anything to share, feel free to post on the [Mech-Mind Online Community](https://community.mech-mind.com/). The community also contains a [specific category for development with Mech-Eye SDK](https://community.mech-mind.com/c/mech-eye-sdk-development/19).

## Build the Sample Program

Prerequisites and instructions for building the sample program on Windows and Ubuntu are provided.

### Windows

#### Prerequisites

The following software are required to build this sample program. Please download and install these software.

* [Mech-Eye SDK (latest version)](https://community.mech-mind.com/c/latest-product-downloads/10)
* [Visual Studio (version 2017 or above)](https://visualstudio.microsoft.com/vs/community/)
* [CMake (version 3.2 or above)](https://cmake.org/download/)

#### Instructions

1. Make sure the `SetLaserFrameRange` folder is in a location with read and write permissions.
2. Run Cmake and set the source and build paths:
   
   | Field                       | Path                         |
   | :----                       | :----                        |
   | Where is the source code    | xxx/SetLaserFrameRange       |
   | Where to build the binaries | xxx/SetLaserFrameRange/build |

3. Click the **Configure** button. In the pop-up window, set the generator and platform according to the actual situation, and then click the **Finish** button.
4. When the log displays **Configuring done**, click the **Generate** button. When the log displays **Generating done**, click the **Open Project** button.
5. In Visual Studio, change the Solution Configuration from **Debug** to **Release**.
6. Right-click the sample in **Solution Explorer**, and select **Set as Startup Project**.
7. Click **Local Windows Debugger** to build the solution.
8. Enter the index of the camera to which you want to connect, and press the Enter key. The obtained files are saved to the `build` folder.

### Ubuntu

Ubuntu 18 or above is required.

#### Prerequisites

* Update the software source list.
  
  ```bash
  sudo apt-get update
  ```

* Install required tools.
  
  ```bash
  sudo apt-get install -y build-essential pkg-config cmake
  ```

* Install [Mech-Eye API (latest version)](https://community.mech-mind.com/c/latest-product-downloads/10).

#### Instructions

1. Navigate to the `SetLaserFrameRange` folder. 
   
   ```bash
   cd /opt/mech-mind/mech-eye-sdk/samples/Laser/SetLaserFrameRange/
   ```
   
2. Configure and build the sample program.

   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

3. Run the sample program.

   ```bash
   ./SetLaserFrameRange
   ```
   
4. Enter the index of the camera to which you want to connect, and press the Enter key.