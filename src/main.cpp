#include <QApplication>
#include "mainwindow.h"

// 程序入口
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);   // Qt 应用对象（必须有且只能有一个）

    MainWindow w;                   // 创建主窗口
    w.show();                       // 显示窗口

    return app.exec();              // 进入事件循环（程序在这里阻塞运行）
}