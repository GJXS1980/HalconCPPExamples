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

namespace GenCaltabNS
{

//Note global variables cannot be shared across exports with different namespace

#ifndef NO_EXPORT_MAIN
// Main procedure 
void action()
{
  //算子gen_caltab( : : XNum,YNum,MarkDist,DiameterRatio,CalTabDescrFile,CalTabPSFile : )
  //XNum 每行黑色标志圆点的数量。
  //YNum 每列黑色标志圆点的数量。
  //MarkDist 两个就近黑色圆点中心之间的距离。（单位为m）
  //DiameterRatio 黑色圆点直径与圆点中心距离的比值。
  //CalTabDescrFile 标定板描述文件的文件路径（.descr）。
  //CalTabPSFile 标定板图像文件的文件路径（.ps）

  GenCaltab(3, 3, 0.02, 0.75, "../circles_pattern/caltab3x3_d15mm.descr", "../circles_pattern/caltab3x3_d15mm.bmp");

  GenCaltab(3, 3, 0.03, 0.666667, "../circles_pattern/caltab3x3_d20mm.descr", "../circles_pattern/caltab3x3_d20mm.bmp");

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

    GenCaltabNS::action();
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
namespace GenCaltabNS
{

#endif


#endif


} // end namespace

