#include<iostream>
#include "HalconCpp.h"

using namespace HalconCpp;
using namespace std;

int main()
{
    HalconCpp::HWindow window;
    try
    {
        // 关闭所有HALCON窗口
        window.CloseWindow();

        // 读取图像
        HObject Image;
        HalconCpp::ReadImage(&Image, "../img/image.png");

        // 获取图像尺寸
        HalconCpp::HTuple Width, Height;
        HalconCpp::GetImageSize(Image, &Width, &Height);

        // 打开一个HALCON窗口并显示图像
        HalconCpp::HWindow WindowHandle;
        window.OpenWindow(0, 0, Width*10, Height*10, 0, "visible", "");
        window.DispObj(Image);
        window.Click(); //  点击才跳转

        // 打开第二个HALCON窗口并显示灰度图像
        HalconCpp::HWindow WindowHandle1;
        HObject GrayImage;
        window.OpenWindow(0, 0, Width*10, Height*10, 0, "visible", "");
        HalconCpp::Rgb1ToGray(Image, &GrayImage);
        window.DispObj(GrayImage);
        window.Click(); //  点击才跳转


        // 对灰度图像进行阈值处理
        HObject Regions;
        HalconCpp::Threshold(GrayImage, &Regions, 0, 186);

        // 打开第三个HALCON窗口并显示处理后的结果
        HalconCpp::HWindow WindowHandle2;
        window.OpenWindow(0, 0, Width*10, Height*10, 0, "visible", "");
        window.DispObj(Regions);
        window.Click(); //  点击才跳转

        // 创建并显示形状模型
        HObject ModelContours;
        HalconCpp::HTuple ModelID;
        HalconCpp::CreateScaledShapeModel(GrayImage, "auto", -0.39, 0.79, "auto", 0.9, 1.1, "auto", "auto", "use_polarity", "auto", "auto", &ModelID);
        HalconCpp::GetShapeModelContours(&ModelContours, ModelID, 1);
        window.DispObj(ModelContours);
        window.Click(); //  点击才跳转
    }
    catch (HException &except)
    {
        cout << "HALCON error: " << except.ErrorMessage() << endl;
    }

    return 0;
}
