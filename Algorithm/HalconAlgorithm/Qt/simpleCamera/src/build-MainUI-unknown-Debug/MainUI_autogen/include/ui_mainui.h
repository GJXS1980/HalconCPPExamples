/********************************************************************************
** Form generated from reading UI file 'mainui.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINUI_H
#define UI_MAINUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainUI
{
public:
    QWidget *centralwidget;
    QWidget *halconWindow;
    QPushButton *btnStart;
    QPushButton *btnStop;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainUI)
    {
        if (MainUI->objectName().isEmpty())
            MainUI->setObjectName(QString::fromUtf8("MainUI"));
        MainUI->resize(800, 600);
        centralwidget = new QWidget(MainUI);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        halconWindow = new QWidget(centralwidget);
        halconWindow->setObjectName(QString::fromUtf8("halconWindow"));
        halconWindow->setGeometry(QRect(10, 10, 641, 531));
        btnStart = new QPushButton(centralwidget);
        btnStart->setObjectName(QString::fromUtf8("btnStart"));
        btnStart->setGeometry(QRect(690, 490, 75, 23));
        btnStop = new QPushButton(centralwidget);
        btnStop->setObjectName(QString::fromUtf8("btnStop"));
        btnStop->setGeometry(QRect(690, 520, 75, 23));
        MainUI->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainUI);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 800, 23));
        MainUI->setMenuBar(menubar);
        statusbar = new QStatusBar(MainUI);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainUI->setStatusBar(statusbar);

        retranslateUi(MainUI);

        QMetaObject::connectSlotsByName(MainUI);
    } // setupUi

    void retranslateUi(QMainWindow *MainUI)
    {
        MainUI->setWindowTitle(QApplication::translate("MainUI", "MainUI", nullptr));
        btnStart->setText(QApplication::translate("MainUI", "\345\274\200\345\247\213\351\207\207\351\233\206", nullptr));
        btnStop->setText(QApplication::translate("MainUI", "\345\201\234\346\255\242\351\207\207\351\233\206", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainUI: public Ui_MainUI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINUI_H
