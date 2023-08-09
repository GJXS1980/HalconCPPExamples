#pragma once

#include <QObject>
#include<QList>
#include "HalconAlgorithm.h"
#include "HalconDeepLearning/HalconDeepLearning.h"

class LensCheckAlgorithmPrivate:public HalconDeepLearning
{
	Q_OBJECT

public:
	LensCheckAlgorithmPrivate(QObject *parent = Q_NULLPTR);
	~LensCheckAlgorithmPrivate();
	void setRingImage(QSharedPointer<HalconCpp::HImage> image);//���û���ͼƬ
	void setBackImage(QSharedPointer<HalconCpp::HImage> image);//���ñ���ͼƬ
	QImage getRingImg();//��û���ͼƬ
	QSharedPointer<HalconCpp::HImage> getHalconRingImg();//��û���ͼƬ
	QImage getBackImg();//��ñ���ͼƬ
	QList<QString> ringCheck();//������<��Ƭ����>
	//����
	void setTestRingImage(const QString& filePath);//���û���ͼƬ·��,��ȡͼƬ��RingImage
private:
	QSharedPointer<HalconCpp::HImage> ringImage_;//����ͼƬ
	QSharedPointer<HalconCpp::HImage> backImage_;//����ͼƬ
};