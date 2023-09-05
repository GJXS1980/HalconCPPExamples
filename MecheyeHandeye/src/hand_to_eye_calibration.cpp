///////////////////////////////////////////////////////////////////////////////
// File generated by HDevelop for HALCON/C++ Version 23.05.0.0
// Non-ASCII strings in this file are encoded in UTF-8.
// 
// Please note that non-ASCII characters in string constants are exported
// as octal codes in order to guarantee that the strings are correctly
// created on all systems, independent on any compiler settings.
// 
// Source files with different encoding should not be mixed in one project.
///////////////////////////////////////////////////////////////////////////////
#include "halconcpp/HalconCpp.h"

using namespace HalconCpp;

namespace HandEyeCal
{

//Note global variables cannot be shared across exports with different namespace
// Procedure declarations 
// Local procedures 
void calibrate_handeye_mmind (HTuple hv_AcqHandle, HTuple *hv_CalibResult, HTuple *hv_ExtriResult);
void captureTranformedPointCloud (HTuple hv_AcqHandle, HTuple hv_ExtriResult);
void collect_pattern (HTuple hv_AcqHandle, HTuple *hv_CollectResult);
void euler_to_quad (HTuple hv_pose, HTuple hv_EulerType, HTuple hv_FromDegree, HTuple *hv_PoseQuad);
void extrinsicTotransval (HTuple hv_extriResult, HTuple *hv_transformVal);
void quad_to_euler (HTuple hv_PoseQuad, HTuple *hv_Euler);
void setboardType (HTuple hv_input, HTuple *hv_output);

// Procedures 
// Local procedures 
void calibrate_handeye_mmind (HTuple hv_AcqHandle, HTuple *hv_CalibResult, HTuple *hv_ExtriResult)
{

  // Local iconic variables

  // Local control variables
  HTuple  hv_Start, hv_timeout, hv_CalibRet, hv_End;
  HTuple  hv_ErrCode, hv_MSecond, hv_Second, hv_Minute, hv_Hour;
  HTuple  hv_Day, hv_YDay, hv_Month, hv_Year, hv_path, hv_FileHandle;

  (*hv_CalibResult) = HTuple();
  //Start calculating extrinsic parameters.
  SetFramegrabberParam(hv_AcqHandle, "CalibrateHandEye", 0);
  WaitSeconds(5);
  CountSeconds(&hv_Start);
  //Set the timeout period for calculation.
  hv_timeout = 1000;
  while (true)
  {
    WaitSeconds(0.5);
    SetFramegrabberParam(hv_AcqHandle, "UserSetLoad", 0);
    //Check whether the calculation has been completed.
    GetFramegrabberParam(hv_AcqHandle, "CalibStatus", &hv_CalibRet);
    if (0 != (int(hv_CalibRet==HTuple("CALIB_DONE"))))
    {
      break;
    }
    CountSeconds(&hv_End);
    if (0 != (int((hv_End-hv_Start)>hv_timeout)))
    {
      break;
    }
  }
  //Check the execution status of the calculation and store the status code in the "CalibResult" variable.
  GetFramegrabberParam(hv_AcqHandle, "ExtrinErrCode", &hv_ErrCode);
  (*hv_CalibResult) = hv_ErrCode;
  //If calculation succeeded, store the extrinsic parameters in a "ExtrinsicParameters" TXT file.
  if (0 != (int(hv_ErrCode==HTuple("SUCCESS"))))
  {
    GetFramegrabberParam(hv_AcqHandle, "Extrinsic", &(*hv_ExtriResult));
    GetSystemTime(&hv_MSecond, &hv_Second, &hv_Minute, &hv_Hour, &hv_Day, &hv_YDay, 
        &hv_Month, &hv_Year);
    hv_path = ((((((HTuple("Mecheye_EyeToHand")+"-")+(hv_Hour.TupleString("d")))+"h")+(hv_Minute.TupleString("d")))+"min")+"s")+".txt";
    OpenFile(hv_path, "append", &hv_FileHandle);
    FwriteString(hv_FileHandle, "ExtrinsicParameters:");
    FnewLine(hv_FileHandle);
    FwriteString(hv_FileHandle, (*hv_ExtriResult));
    FnewLine(hv_FileHandle);
    CloseFile(hv_FileHandle);
  }
  return;
}

void captureTranformedPointCloud (HTuple hv_AcqHandle, HTuple hv_ExtriResult)
{

  // Local iconic variables
  HObject  ho_Image3d, ho_Region, ho_Contours;

  // Local control variables
  HTuple  hv_TransformVal, hv_Width, hv_Height;
  HTuple  hv_PixeLFormat, hv_ObjectModel3D, hv_NumOfPoints;

  //Set this parameter to "true" to output point clouds in the robot reference frame.
  SetFramegrabberParam(hv_AcqHandle, "Scan3dCoordinateTransformEnable", 0);
  //set_framegrabber_param (AcqHandle, 'Scan3dCoordinateTransformEnable', true)

  //Transfrom the quaternions in the extrinsic parameters to Euler angles (X-Y-Z).
  extrinsicTotransval(hv_ExtriResult, &hv_TransformVal);
  //Set the reference frame transform values to the calculated extrinsic parameters.
  SetFramegrabberParam(hv_AcqHandle, "Scan3dCoordinateTransformSelector", "RotationX");
  SetFramegrabberParam(hv_AcqHandle, "Scan3dTransformValue", HTuple(hv_TransformVal[3]));
  SetFramegrabberParam(hv_AcqHandle, "Scan3dCoordinateTransformSelector", "RotationY");
  SetFramegrabberParam(hv_AcqHandle, "Scan3dTransformValue", HTuple(hv_TransformVal[4]));
  SetFramegrabberParam(hv_AcqHandle, "Scan3dCoordinateTransformSelector", "RotationZ");
  SetFramegrabberParam(hv_AcqHandle, "Scan3dTransformValue", HTuple(hv_TransformVal[5]));
  SetFramegrabberParam(hv_AcqHandle, "Scan3dCoordinateTransformSelector", "TranslationX");
  SetFramegrabberParam(hv_AcqHandle, "Scan3dTransformValue", HTuple(hv_TransformVal[0]));
  SetFramegrabberParam(hv_AcqHandle, "Scan3dCoordinateTransformSelector", "TranslationY");
  SetFramegrabberParam(hv_AcqHandle, "Scan3dTransformValue", HTuple(hv_TransformVal[1]));
  SetFramegrabberParam(hv_AcqHandle, "Scan3dCoordinateTransformSelector", "TranslationZ");
  SetFramegrabberParam(hv_AcqHandle, "Scan3dTransformValue", HTuple(hv_TransformVal[2]));
  //Switch the "DeviceScanType" parameter to "Areascan3D" to obtain the 3D data.
  SetFramegrabberParam(hv_AcqHandle, "DeviceScanType", "Areascan3D");
  //Open the 3D object model generator.
  SetFramegrabberParam(hv_AcqHandle, "create_objectmodel3d", "enable");
  SetFramegrabberParam(hv_AcqHandle, "add_objectmodel3d_overlay_attrib", "enable");

  GetFramegrabberParam(hv_AcqHandle, "Width", &hv_Width);
  GetFramegrabberParam(hv_AcqHandle, "Height", &hv_Height);
  GetFramegrabberParam(hv_AcqHandle, "PixelFormat", &hv_PixeLFormat);

  //Generate the point cloud (stored in the "ObjectModel3D" variable).
  GrabData(&ho_Image3d, &ho_Region, &ho_Contours, hv_AcqHandle, &hv_ObjectModel3D);

  GetObjectModel3dParams(hv_ObjectModel3D, "num_points", &hv_NumOfPoints);

  if (0 != (int(hv_NumOfPoints!=0)))
  {
    //Save the point cloud as a "PointCloud" PLY file.
    WriteObjectModel3d(hv_ObjectModel3D, "ply", "PointCloud.ply", HTuple(), HTuple());
  }
  ClearObjectModel3d(hv_ObjectModel3D);
  return;
}

void collect_pattern (HTuple hv_AcqHandle, HTuple *hv_CollectResult)
{

  // Local iconic variables

  // Local control variables
  HTuple  hv_Start, hv_timeout, hv_CollectRet, hv_End;
  HTuple  hv_ErrCode;

  (*hv_CollectResult) = HTuple();
  //Start performing image capturing at the current calibration pose, detecting features, and adding the feature detection data to the extrinsic parameter calculation.
  SetFramegrabberParam(hv_AcqHandle, "CollectPatternOnce", 0);
  CountSeconds(&hv_Start);
  hv_timeout = 10;
  while (true)
  {
    WaitSeconds(0.5);
    SetFramegrabberParam(hv_AcqHandle, "UserSetLoad", 0);
    //Check whether the above processes have been completed.
    GetFramegrabberParam(hv_AcqHandle, "CollectStatus", &hv_CollectRet);
    if (0 != (int(hv_CollectRet==HTuple("COLLECT_DONE"))))
    {
      break;
    }
    CountSeconds(&hv_End);
    if (0 != (int((hv_End-hv_Start)>hv_timeout)))
    {
      break;
    }
  }
  //Check the execution status of the above processes and store the status code in the "CollectResult" variable.
  GetFramegrabberParam(hv_AcqHandle, "ExtrinErrCode", &hv_ErrCode);
  (*hv_CollectResult) = hv_ErrCode;
  return;
}

void euler_to_quad (HTuple hv_pose, HTuple hv_EulerType, HTuple hv_FromDegree, HTuple *hv_PoseQuad)
{

  // Local iconic variables

  // Local control variables
  HTuple  hv_a, hv_b, hv_c, hv_a1, hv_a2, hv_a3;

  //This procedure contains the conversions from different Euler angle conventions to quaternions.
  //If you do not see the Euler angle convention used by your robot listed here, please refer to the existing code of this procedure and add the conversion.
  (*hv_PoseQuad) = HTuple();
  if (0 != (int(hv_FromDegree==1)))
  {
    TupleRad(HTuple(hv_pose[3]), &hv_a);
    TupleRad(HTuple(hv_pose[4]), &hv_b);
    TupleRad(HTuple(hv_pose[5]), &hv_c);
  }
  hv_a1 = hv_a/2;
  hv_a2 = hv_b/2;
  hv_a3 = hv_c/2;
  (*hv_PoseQuad)[0] = HTuple(hv_pose[0]);
  (*hv_PoseQuad)[1] = HTuple(hv_pose[1]);
  (*hv_PoseQuad)[2] = HTuple(hv_pose[2]);

  //Z-Y'-X''
  if (0 != (int(hv_EulerType==HTuple("rzyx"))))
  {
    (*hv_PoseQuad)[3] = (((hv_a1.TupleSin())*(hv_a2.TupleSin()))*(hv_a3.TupleSin()))+(((hv_a1.TupleCos())*(hv_a2.TupleCos()))*(hv_a3.TupleCos()));
    (*hv_PoseQuad)[4] = (((-(hv_a1.TupleSin()))*(hv_a2.TupleSin()))*(hv_a3.TupleCos()))+(((hv_a3.TupleSin())*(hv_a1.TupleCos()))*(hv_a2.TupleCos()));
    (*hv_PoseQuad)[5] = (((hv_a1.TupleSin())*(hv_a3.TupleSin()))*(hv_a2.TupleCos()))+(((hv_a2.TupleSin())*(hv_a1.TupleCos()))*(hv_a3.TupleCos()));
    (*hv_PoseQuad)[6] = (((hv_a1.TupleSin())*(hv_a2.TupleCos()))*(hv_a3.TupleCos()))-(((hv_a2.TupleSin())*(hv_a3.TupleSin()))*(hv_a1.TupleCos()));
  }

  //Z-Y'-Z''
  if (0 != (int(hv_EulerType==HTuple("rzyz"))))
  {

    (*hv_PoseQuad)[3] = (hv_a2.TupleCos())*((hv_a1+hv_a3).TupleCos());
    (*hv_PoseQuad)[4] = (-(hv_a2.TupleSin()))*((hv_a1-hv_a3).TupleSin());
    (*hv_PoseQuad)[5] = (hv_a2.TupleSin())*((hv_a1-hv_a3).TupleCos());
    (*hv_PoseQuad)[6] = (hv_a2.TupleCos())*((hv_a1+hv_a3).TupleSin());
  }

  //X-Y'-Z''
  if (0 != (int(hv_EulerType==HTuple("rxyz"))))
  {

    (*hv_PoseQuad)[3] = (((-(hv_a1.TupleSin()))*(hv_a2.TupleSin()))*(hv_a3.TupleSin()))+(((hv_a1.TupleCos())*(hv_a2.TupleCos()))*(hv_a3.TupleCos()));
    (*hv_PoseQuad)[4] = (((hv_a1.TupleSin())*(hv_a2.TupleCos()))*(hv_a3.TupleCos()))+(((hv_a2.TupleSin())*(hv_a3.TupleSin()))*(hv_a1.TupleCos()));
    (*hv_PoseQuad)[5] = (((-(hv_a1.TupleSin()))*(hv_a3.TupleSin()))*(hv_a2.TupleCos()))+(((hv_a2.TupleSin())*(hv_a1.TupleCos()))*(hv_a3.TupleCos()));
    (*hv_PoseQuad)[6] = (((hv_a1.TupleSin())*(hv_a2.TupleSin()))*(hv_a3.TupleCos()))+(((hv_a3.TupleSin())*(hv_a1.TupleCos()))*(hv_a2.TupleCos()));
  }

  //Z-X'-Z''
  if (0 != (int(hv_EulerType==HTuple("rzxz"))))
  {

    (*hv_PoseQuad)[3] = (hv_a2.TupleCos())*((hv_a1+hv_a3).TupleCos());
    (*hv_PoseQuad)[4] = (hv_a2.TupleSin())*((hv_a1-hv_a3).TupleCos());
    (*hv_PoseQuad)[5] = (hv_a2.TupleSin())*((hv_a1-hv_a3).TupleSin());
    (*hv_PoseQuad)[6] = (hv_a2.TupleCos())*((hv_a1+hv_a3).TupleSin());
  }

  //X-Y-Z
  if (0 != (int(hv_EulerType==HTuple("sxyz"))))
  {
    hv_a1 = hv_c/2;
    hv_a2 = hv_b/2;
    hv_a3 = hv_a/2;
    (*hv_PoseQuad)[3] = (((hv_a1.TupleSin())*(hv_a2.TupleSin()))*(hv_a3.TupleSin()))+(((hv_a1.TupleCos())*(hv_a2.TupleCos()))*(hv_a3.TupleCos()));
    (*hv_PoseQuad)[4] = (((-(hv_a1.TupleSin()))*(hv_a2.TupleSin()))*(hv_a3.TupleCos()))+(((hv_a3.TupleSin())*(hv_a1.TupleCos()))*(hv_a2.TupleCos()));
    (*hv_PoseQuad)[5] = (((hv_a1.TupleSin())*(hv_a3.TupleSin()))*(hv_a2.TupleCos()))+(((hv_a2.TupleSin())*(hv_a1.TupleCos()))*(hv_a3.TupleCos()));
    (*hv_PoseQuad)[6] = (((hv_a1.TupleSin())*(hv_a2.TupleCos()))*(hv_a3.TupleCos()))-(((hv_a2.TupleSin())*(hv_a3.TupleSin()))*(hv_a1.TupleCos()));
  }

  return;
}

void extrinsicTotransval (HTuple hv_extriResult, HTuple *hv_transformVal)
{

  // Local iconic variables

  // Local control variables
  HTuple  hv_TransformVal, hv_tmp, hv_w, hv_x, hv_y;
  HTuple  hv_z, hv_rada, hv_radb, hv_radc, hv_dega, hv_degb;
  HTuple  hv_degc;

  hv_TransformVal = HTuple();
  TupleSplit(hv_extriResult, HTuple(","), &hv_tmp);
  TupleNumber(hv_tmp, &hv_TransformVal);
  //m to mm
  (*hv_transformVal)[0] = HTuple(hv_TransformVal[0])*1000;
  (*hv_transformVal)[1] = HTuple(hv_TransformVal[1])*1000;
  (*hv_transformVal)[2] = HTuple(hv_TransformVal[2])*1000;
  hv_w = ((const HTuple&)hv_TransformVal)[3];
  hv_x = ((const HTuple&)hv_TransformVal)[4];
  hv_y = ((const HTuple&)hv_TransformVal)[5];
  hv_z = ((const HTuple&)hv_TransformVal)[6];
  hv_rada = (2*((hv_w*hv_x)+(hv_y*hv_z))).TupleAtan2(1-(2*((hv_x*hv_x)+(hv_y*hv_y))));
  hv_radb = (2*((hv_w*hv_y)-(hv_z*hv_x))).TupleAsin();
  hv_radc = (2*((hv_w*hv_z)+(hv_x*hv_y))).TupleAtan2(1-(2*((hv_y*hv_y)+(hv_z*hv_z))));
  TupleDeg(hv_rada, &hv_dega);
  TupleDeg(hv_radb, &hv_degb);
  TupleDeg(hv_radc, &hv_degc);
  (*hv_transformVal)[3] = hv_dega;
  (*hv_transformVal)[4] = hv_degb;
  (*hv_transformVal)[5] = hv_degc;
  return;
}

void quad_to_euler (HTuple hv_PoseQuad, HTuple *hv_Euler)
{

  // Local iconic variables

  // Local control variables
  HTuple  hv_w, hv_x, hv_y, hv_z, hv_r, hv_p, hv_rdeg;
  HTuple  hv_pdeg, hv_ydeg;

  (*hv_Euler) = HTuple();
  hv_w = ((const HTuple&)hv_PoseQuad)[3];
  hv_x = ((const HTuple&)hv_PoseQuad)[4];
  hv_y = ((const HTuple&)hv_PoseQuad)[5];
  hv_z = ((const HTuple&)hv_PoseQuad)[6];
  hv_r = (2*((hv_w*hv_x)+(hv_y*hv_z))).TupleAtan2(1-(2*((hv_x*hv_x)+(hv_y*hv_y))));
  hv_p = (2*((hv_w*hv_y)-(hv_z*hv_x))).TupleAsin();
  hv_y = (2*((hv_w*hv_z)+(hv_x*hv_y))).TupleAtan2(1-(2*((hv_y*hv_y)+(hv_z*hv_z))));
  TupleDeg(hv_r, &hv_rdeg);
  TupleDeg(hv_p, &hv_pdeg);
  TupleDeg(hv_y, &hv_ydeg);
  (*hv_Euler)[0] = HTuple(hv_PoseQuad[0]);
  (*hv_Euler)[1] = HTuple(hv_PoseQuad[1]);
  (*hv_Euler)[2] = HTuple(hv_PoseQuad[2]);
  (*hv_Euler)[3] = hv_rdeg;
  (*hv_Euler)[4] = hv_pdeg;
  (*hv_Euler)[5] = hv_ydeg;
  return;
}

void setboardType (HTuple hv_input, HTuple *hv_output)
{

  TupleRegexpReplace(hv_input, (HTuple("-0*").Append("replace_all")), "_", &(*hv_output));
  return;
}

#ifndef NO_EXPORT_MAIN
// Main procedure 
void action()
{

  // Local iconic variables
  HObject  ho_Image;

  // Local control variables
  HTuple  hv_DeviceInfo, hv_ExtriResult, hv_Info;
  HTuple  hv_DeviceInfos, hv_MechEyeCameras, hv_AcqHandle;
  HTuple  hv_ParameterValues, hv_FirmwareVersion, hv_boardType;
  HTuple  hv_DictJson, hv_ALLKeys, hv_EulerType, hv_FromDegree;
  HTuple  hv_PoseCount, hv_PatternsResult, hv_I, hv_PoseDataKey;
  HTuple  hv_PoseObj, hv_Robot_x, hv_Robot_y, hv_Robot_z;
  HTuple  hv_Robot_r1, hv_Robot_r2, hv_Robot_r3, hv_pose;
  HTuple  hv_PoseQuad, hv_PoseStr, hv_CollectResult, hv_Text;
  HTuple  hv_CalibResult;

  //替换 "MechEye" 为要连接的相机的 "user_name" 或 "unique_name"
  hv_DeviceInfo = "MechEye";

  //new add
  hv_ExtriResult = HTuple();

  //列出可用的相机
  InfoFramegrabber("GigEVision2", "device", &hv_Info, &hv_DeviceInfos);
  TupleRegexpSelect(hv_DeviceInfos, hv_DeviceInfo, &hv_MechEyeCameras);
  // dev_inspect_ctrl(...); only in hdevelop

  //如果没有找到相机则停止程序执行
  if (0 != (HTuple(hv_MechEyeCameras.TupleLength()).TupleNot()))
  {
    // stop(...); only in hdevelop
  }

  //连接相机：如果 "user_Name" 或 "unique_name" 在 tuple_regexp_select 中没有设置，则将连接列表中的第一个相机。
  OpenFramegrabber("GigEVision2", 1, 1, 0, 0, 0, 0, "default", -1, "default", -1, 
      "false", "default", HTuple(hv_MechEyeCameras[0]), 0, -1, &hv_AcqHandle);
  GetFramegrabberParam(hv_AcqHandle, "available_param_names", &hv_ParameterValues);
  GetFramegrabberParam(hv_AcqHandle, "DeviceFirmwareVersion", &hv_FirmwareVersion);

  //如果相机固件版本低于 2.1.0 则停止程序执行
  if (0 != (int(hv_FirmwareVersion<HTuple("2.1.0"))))
  {
    // stop(...); only in hdevelop
  }

  //设置图像采集的超时时间
  SetFramegrabberParam(hv_AcqHandle, "grab_timeout", 10000);

  //将 "DeviceScanType" 参数切换为 "Areascan" 以获取2D图像
  SetFramegrabberParam(hv_AcqHandle, "DeviceScanType", "Areascan");

  //设置相机安装方法：如果相机是安装在手眼间的，将 "EyeInHnad" 替换为 "EyeToHand"
  SetFramegrabberParam(hv_AcqHandle, "CalibrationType", "EyeToHand");

  //设置校准板模型：将 "BDB-5" 替换为正在使用的校准板模型。可能的值包括 BDB-5、BDB-6、BDB-7、OCB-005、OCB-010、OCB-015、OCB-020、CGB-020、CGB-035 和 CGB-050。
  //BDB-5:标定板与相机距离 < 0.6 m;BDB-6:标定板与相机距离 0.6–1.5 m;BDB-7:标定板与相机距离 > 1.5 m
  setboardType("BDB-7", &hv_boardType);
  SetFramegrabberParam(hv_AcqHandle, "BoardType", hv_boardType);
  SetFramegrabberParam(hv_AcqHandle, "Test Collect", 0);

  //从JSON 文件读取数据
  ReadDict("kawasaki_pose.json", HTuple(), HTuple(), &hv_DictJson);
  GetDictParam(hv_DictJson, "keys", HTuple(), &hv_ALLKeys);
  GetDictTuple(hv_DictJson, "EulerType", &hv_EulerType);
  GetDictTuple(hv_DictJson, "FromDegree", &hv_FromDegree);
  GetDictTuple(hv_DictJson, "pose_count", &hv_PoseCount);

  //开始执行手眼标定
  //逐个读取 JSON 文件中的校准位姿
  hv_PatternsResult = HTuple();
  {
  HTuple end_val51 = hv_PoseCount-1;
  HTuple step_val51 = 1;
  for (hv_I=0; hv_I.Continue(end_val51, step_val51); hv_I += step_val51)
  {
    hv_PoseDataKey = "pose_"+(hv_I.TupleString("02d"));
    GetDictTuple(hv_DictJson, hv_PoseDataKey, &hv_PoseObj);
    GetDictTuple(hv_PoseObj, 0, &hv_Robot_x);
    GetDictTuple(hv_PoseObj, 1, &hv_Robot_y);
    GetDictTuple(hv_PoseObj, 2, &hv_Robot_z);
    GetDictTuple(hv_PoseObj, 3, &hv_Robot_r1);
    GetDictTuple(hv_PoseObj, 4, &hv_Robot_r2);
    GetDictTuple(hv_PoseObj, 5, &hv_Robot_r3);
    hv_pose.Clear();
    hv_pose.Append(hv_Robot_x);
    hv_pose.Append(hv_Robot_y);
    hv_pose.Append(hv_Robot_z);
    hv_pose.Append(hv_Robot_r1);
    hv_pose.Append(hv_Robot_r2);
    hv_pose.Append(hv_Robot_r3);

    //将欧拉角转换为四元数。如果您的机器人使用的欧拉角约定不受支持，请打开下面的过程并添加将此欧拉角约定转换为四元数的代码
    euler_to_quad(hv_pose, hv_EulerType, hv_FromDegree, &hv_PoseQuad);
    hv_PoseStr = (((((((((((HTuple(hv_PoseQuad[0])+HTuple(","))+HTuple(hv_PoseQuad[1]))+HTuple(","))+HTuple(hv_PoseQuad[2]))+HTuple(","))+HTuple(hv_PoseQuad[3]))+HTuple(","))+HTuple(hv_PoseQuad[4]))+HTuple(","))+HTuple(hv_PoseQuad[5]))+HTuple(","))+HTuple(hv_PoseQuad[6]);

    //Send the current calibration pose to the camera.
    //发送当前校准位姿给相机
    SetFramegrabberParam(hv_AcqHandle, "PoseData", hv_PoseStr);

    //将该位姿添加到外参参数计算中
    SetFramegrabberParam(hv_AcqHandle, "AddPose", 0);

    //移动机器人到JSON 文件中的下一个校准位姿，然后按F5键继续运行该程序
    // stop(...); only in hdevelop
    //在当前校准位姿进行图像采集，检测特征，并将特征检测数据添加到外参参数计算中
    SetFramegrabberParam(hv_AcqHandle, "Test Collect", 1);

    collect_pattern(hv_AcqHandle, &hv_CollectResult);

    //在 "PatternsResult" 变量中记录每个校准位姿的特征检测状态
    TupleInsert(hv_PatternsResult, 0, hv_CollectResult, &hv_PatternsResult);
    GrabImage(&ho_Image, hv_AcqHandle);
    hv_Text = (((((((("No. of the current calibration pose:"+(hv_I+1))+"\n")+"Total no. of calibration poses:")+hv_PoseCount)+"\n")+"Feature detection status at current pose:")+hv_CollectResult)+"\n")+"Move the robot to the next calibration pose.";
    if (HDevWindowStack::IsOpen())
      DispText(HDevWindowStack::GetActive(),hv_Text, "window", "top", "center", "black", 
          "shadow", 0);
  }
  }

  //已在所有校准位姿进行特征检测数据收集。开始计算外参参数
  calibrate_handeye_mmind(hv_AcqHandle, &hv_CalibResult, &hv_ExtriResult);

  //如果计算成功，外参参数将保存在 "ExtrinsicParameters" TXT 文件中。平移分量的单位为m，旋转分量使用四元数描述。
  //切换输出点云时相机所在的参考坐标系，并获取点云
  //默认情况下，不切换参考坐标系。如果您需要将点云输出在机器人参考坐标系中，请打开下面的过程，并更改 "Scan3dCoordinateTransformEnable" 参数的值
  captureTranformedPointCloud(hv_AcqHandle, hv_ExtriResult);

  //The point cloud is saved as a "PointCloud" PLY file.
  //点云将以 "PointCloud" PLY 文件的形式保存
  CloseFramegrabber(hv_AcqHandle);

}


#ifndef NO_EXPORT_APP_MAIN


} // end namespace

int main(int argc, char *argv[])
{
  int ret = 0;

  try
  {
#if defined(_WIN32)
    SetSystem("use_window_thread", "true");
#endif

    // Default settings used in HDevelop (can be omitted)
    SetSystem("width", 512);
    SetSystem("height", 512);

    HandEyeCal::action();
  }
  catch (HException &exception)
  {
    fprintf(stderr,"  Error #%u in %s: %s\n", exception.ErrorCode(),
            exception.ProcName().TextA(),
            exception.ErrorMessage().TextA());
    ret = 1;
  }
  return ret;
}
namespace HandEyeCal
{


#endif


#endif


} // end namespace

