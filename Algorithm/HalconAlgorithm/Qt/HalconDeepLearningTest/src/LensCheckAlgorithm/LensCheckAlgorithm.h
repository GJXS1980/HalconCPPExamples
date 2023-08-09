#pragma once

#include<QObject>
#include <QPixmap>
#include<QMap>
#include<QSharedPointer>
#include "lenscheckalgorithm_global.h"

namespace HalconCpp { 
	class HImage;
	class HTuple;
}

class  LensCheckAlgorithmPrivate;

class LENSCHECKALGORITHM_EXPORT LensCheckAlgorithm:public QObject
{
	Q_OBJECT

public:
	static LensCheckAlgorithm& globalInstance();//ȫ��Ψһ
	~LensCheckAlgorithm();
	void setRingImage(QSharedPointer<HalconCpp::HImage> image);//���û���ͼƬ
	void setBackImage(QSharedPointer<HalconCpp::HImage> image);//���ñ���ͼƬ
	QImage getRingImg();//��û���ͼƬ
	QSharedPointer<HalconCpp::HImage> getHalconRingImg();//��û���ͼƬ
	QImage getBackImg();//��ñ���ͼƬ
	QList<QString> ringCheck();//������<��Ƭ����>
	//����
	void setTestRingImage(const QString& filePath);//���û���ͼƬ·��,��ȡͼƬ��RingImage
private:
	LensCheckAlgorithm(QObject *parent = Q_NULLPTR);

	LensCheckAlgorithmPrivate *data_;
};
