#include "mainwindow.h"
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGridLayout>
#include <QDebug>
#include <QTimer>
#include <QPixmap>
#include <QSlider>
#include <QCheckBox>
#include <QTextEdit>
#include <QStatusBar>
#include <QMessageBox>
#include <QPainter>
#include <QDialog>
#include <QLineEdit>
#include <QScrollArea>
#include <QMetaType>
#include  "yolo_datatype.h"

// Q_DECLARE_METATYPE(StreamOption)

class ClassSelectDialog : public QDialog {
public:
    ClassSelectDialog(const QStringList& classes, const QList<int>& currentSelected, QWidget* parent = nullptr)
        : QDialog(parent)
    {
        setWindowTitle("选择检测类别");
        resize(700, 500);
        this->setStyleSheet(
            "QDialog { background-color: #0b0e1a; border: 2px solid #1c2541; }"
            "QCheckBox { color: #cfcfcf; font-size: 14px; }"
            "QCheckBox::indicator { width: 18px; height: 18px; border: 1px solid #5c8df6; }"
            "QCheckBox::indicator:checked { background-color: #00ff00; border: 1px solid #00ff00; }"
            "QLineEdit { background-color: #1c2541; color: white; border: 1px solid #5c8df6; padding: 5px; }"
            "QScrollArea { border: none; background-color: transparent; }"
        );

        QVBoxLayout* mainV = new QVBoxLayout(this);

        QHBoxLayout* topH = new QHBoxLayout();
        searchEdit = new QLineEdit();
        searchEdit->setPlaceholderText("输入关键字筛选...");
        searchEdit->setStyleSheet("background-color: #1c2541; border: 1px solid #2e3a5a; padding: 5px;");

        QPushButton* btnAll = new QPushButton("全选");
        QPushButton* btnNone = new QPushButton("清空");
        QString btnStyle = "QPushButton { background: #1d3161; padding: 5px 15px; border-radius: 3px; } QPushButton:hover { background: #2a478a; }";
        btnAll->setStyleSheet(btnStyle);
        btnNone->setStyleSheet(btnStyle);

        topH->addWidget(searchEdit);
        topH->addWidget(btnAll);
        topH->addWidget(btnNone);
        mainV->addLayout(topH);

        QScrollArea* scroll = new QScrollArea();
        scroll->setWidgetResizable(true);
        scroll->setStyleSheet("QScrollArea { border: none; }");

        QWidget* container = new QWidget();
        grid = new QGridLayout(container);
        grid->setSpacing(10);

        for (int i = 0; i < classes.size(); ++i) {
            QCheckBox* chk = new QCheckBox(classes[i]);
            chk->setProperty("id", i);
            if (currentSelected.contains(i)) chk->setChecked(true);
            checkList.append(chk);
            grid->addWidget(chk, i / 4, i % 4);
        }

        scroll->setWidget(container);
        mainV->addWidget(scroll);

        QHBoxLayout* bottomH = new QHBoxLayout();
        QPushButton* btnOk = new QPushButton("应用配置");
        btnOk->setStyleSheet("background: #0e639c; font-weight: bold; padding: 8px;");
        bottomH->addStretch();
        bottomH->addWidget(btnOk);
        mainV->addLayout(bottomH);

        connect(btnOk, &QPushButton::clicked, this, &QDialog::accept);
        connect(btnAll, &QPushButton::clicked, this, [this](){ for (auto c : checkList) c->setChecked(true); });
        connect(btnNone, &QPushButton::clicked, this, [this](){ for (auto c : checkList) c->setChecked(false); });
        connect(searchEdit, &QLineEdit::textChanged, this, [this](const QString& text){
            for (auto chk : checkList) {
                chk->setHidden(!chk->text().contains(text, Qt::CaseInsensitive));
            }
        });
    }

    QList<int> getSelectedIds() const {
        QList<int> ids;
        for (auto chk : checkList) {
            if (chk->isChecked()) ids.append(chk->property("id").toInt());
        }
        return ids;
    }

private:
    QGridLayout* grid;
    QLineEdit* searchEdit;
    QList<QCheckBox*> checkList;
};

const QStringList MainWindow::cnClasses = {
    "人", "自行车", "汽车", "摩托车", "飞机", "公交车", "火车", "卡车", "船", "红绿灯",
    "消防栓", "停止标志", "停车收费表", "长凳", "鸟", "猫", "狗", "马", "羊", "牛",
    "大象", "熊", "斑马", "长颈鹿", "背包", "雨伞", "手提包", "领带", "手提箱", "飞盘",
    "滑雪板", "单板", "运动球", "风筝", "棒球棒", "棒球套", "滑板", "冲浪板", "网球拍", "瓶子",
    "红酒杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", "苹果", "三明治", "橙子",
    "西兰花", "胡萝卜", "热狗", "比萨", "甜甜圈", "蛋糕", "椅子", "沙发", "盆栽", "床",
    "餐桌", "厕所", "电视", "笔记本电脑", "鼠标", "遥控器", "键盘", "手机", "微波炉", "烤箱",
    "烤面包机", "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷"
};
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    qRegisterMetaType<StreamOption>("StreamOption");

    // 设置窗口基本属性
    setWindowTitle("双摄像头目标检测系统 - 实时监控终端");
    resize(1500, 950);

    // 应用全局深色 QSS 样式
    this->setStyleSheet(
        "QMainWindow { background-color: #1e1e1e; }"
        "QGroupBox { color: #00ff00; border: 1px solid #3d3d3d; margin-top: 15px; font-weight: bold; border-radius: 5px; padding: 10px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        "QLabel { color: #cfcfcf; font-family: 'Microsoft YaHei'; }"
        "QPushButton { background-color: #333; color: #fff; border: 1px solid #555; padding: 6px; border-radius: 4px; }"
        "QPushButton:hover { background-color: #444; border-color: #00ff00; }"
        "QPushButton:pressed { background-color: #222; }"
        "QComboBox { background-color: #252525; color: #fff; border: 1px solid #444; padding: 3px; }"
        "QTextEdit { background-color: #0c0c0c; color: #00dd00; border: 1px solid #333; font-family: 'Consolas'; font-size: 10pt; }"
        "QSlider::handle:horizontal { background: #00ff00; border: 1px solid #00aa00; width: 14px; margin: -5px 0; border-radius: 7px; }"
        "QSlider::groove:horizontal { border: 1px solid #444; height: 4px; background: #222; }");

    setupUi();
    setupLayout();
    initData();
    setupConnections();
}

MainWindow::~MainWindow()
{
   
}

void MainWindow::setupUi()
{
    // 1. 视频显示区
    video1 = new QLabel("CAMERA 01 - WAITING");
    video1->setStyleSheet("background-color: #000; border: 1px solid #00ff00;");
    video1->setAlignment(Qt::AlignCenter);

    video2 = new QLabel("CAMERA 02 - WAITING");
    video2->setStyleSheet("background-color: #000; border: 1px solid #00ff00;");
    video2->setAlignment(Qt::AlignCenter);
    // 主视频显示区
    videoMain = new QLabel("MAIN MONITOR - ACTIVE");
    videoMain->setStyleSheet("background-color: #000; border: 2px solid #00ff00; color: #00ff00; font-weight: bold;");
    videoMain->setAlignment(Qt::AlignCenter);
    // 2. 日志输出区
    logArea = new QTextEdit();
    logArea->setReadOnly(true);
    logArea->setPlaceholderText("系统初始化中...");

    // 3. 右侧控制组件
    cbSource1_1 = new QComboBox();
    cbSource1_2 = new QComboBox();
    btnStart1 = new QPushButton("启动");
    btnPause1 = new QPushButton("暂停");
    btnStart1->setStyleSheet("background-color: #0e639c; font-weight: bold;");

    cbSource2_1 = new QComboBox();
    cbSource2_2 = new QComboBox();
    btnStart2 = new QPushButton("启动");
    btnPause2 = new QPushButton("暂停");
    btnStart2->setStyleSheet("background-color: #0e639c; font-weight: bold;");

    btnStart3 = new QPushButton("启动");
    btnPause3 = new QPushButton("暂停");
    btnStart3->setStyleSheet("background-color: #0e639c; font-weight: bold;");

    // 4. 参数调节组件
    sliderConf = new QSlider(Qt::Horizontal);
    sliderConf->setRange(0, 100);
    sliderConf->setValue(55);

    sliderIOU = new QSlider(Qt::Horizontal);
    sliderIOU->setRange(0, 100);
    sliderIOU->setValue(25);

    // chkGPU = new QCheckBox("使用 TensorRT/GPU 加速");
    // chkGPU->setChecked(true);
    // chkGPU->setStyleSheet("color: #00ff00; font-size: 11px;");

    // 5. 状态栏
    infoLabel = new QLabel("系统就绪 | 未检测到异常");
}



void MainWindow::setupLayout()
{
    QWidget *central = new QWidget(this);
    setCentralWidget(central);
    // 全局暗黑赛博风格背景
    central->setStyleSheet("background-color: #05070d; color: #d0d0d0;");
    
    QHBoxLayout *mainLayout = new QHBoxLayout(central);
    mainLayout->setContentsMargins(0, 0, 15, 15); // 左侧无边距，让导航栏贴边
    mainLayout->setSpacing(15);

    // ==========================================
    // 第一部分：左侧手风琴折叠控制栏
    // ==========================================
    QWidget *sidebarWidget = new QWidget();
    sidebarWidget->setFixedWidth(280); 
    sidebarWidget->setStyleSheet("background-color: #0b0e1a; border-right: 1px solid #1c2541;");
    QVBoxLayout *sidebarV = new QVBoxLayout(sidebarWidget);
    sidebarV->setContentsMargins(15, 20, 15, 20);
    sidebarV->setSpacing(5);

    // 1. 系统 Logo & 标题
    QLabel *lblSysTitle = new QLabel("多源融合跟踪系统\n  Multi-source Tracking");
    lblSysTitle->setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold; border: none; margin-bottom: 15px;");
    sidebarV->addWidget(lblSysTitle);

    // --- 定义折叠面板的通用样式 ---
    QString navStyle = "QPushButton { background: transparent; color: #a0aabf; font-size: 14px; text-align: left; padding: 12px 15px; border-radius: 6px; border: none; font-weight: bold; } "
                       "QPushButton:hover { background: #151a30; color: #fff; } ";
    QString activeNavStyle = "QPushButton { background: #1d3161; color: #5c8df6; font-size: 14px; text-align: left; padding: 12px 15px; border-radius: 6px; border: none; font-weight: bold; }";
    QString innerBtnStyle = "QPushButton { background: #1c2541; color: #d0d0d0; border-radius: 4px; padding: 10px; border: none; margin-left: 20px; font-weight: bold; } "
                            "QPushButton:hover { background: #2a478a; color: #00ff00; }";

    // ----------------------------------------------------
    // [折叠组 1] 启动控制
    // ----------------------------------------------------
    btnNavStart = new QPushButton("▶ 启动控制");
    btnNavStart->setStyleSheet(navStyle);
    panelStart = new QWidget();
    QVBoxLayout *vStart = new QVBoxLayout(panelStart);
    vStart->setContentsMargins(0, 0, 0, 10); vStart->setSpacing(8);
    btnStart1->setText("🚀 启动 CAM 01"); btnStart1->setStyleSheet(innerBtnStyle);
    btnStart2->setText("🚀 启动 CAM 02"); btnStart2->setStyleSheet(innerBtnStyle);
    btnStart3->setText("🚀 启动 CAM 03"); btnStart3->setStyleSheet(innerBtnStyle);
    vStart->addWidget(btnStart1); vStart->addWidget(btnStart2); vStart->addWidget(btnStart3);
    panelStart->hide(); 
    sidebarV->addWidget(btnNavStart); sidebarV->addWidget(panelStart);

    connect(btnNavStart, &QPushButton::clicked, this, [=](){
        bool v = panelStart->isVisible(); panelStart->setVisible(!v);
        btnNavStart->setText(v ? "▶ 启动控制 " : "▶ 启动控制 ");
        btnNavStart->setStyleSheet(v ? navStyle : activeNavStyle);
    });

    // ----------------------------------------------------
    // [折叠组 2] 暂停控制
    // ----------------------------------------------------
    btnNavPause = new QPushButton("⏸ 暂停控制 ");
    btnNavPause->setStyleSheet(navStyle);
    panelPause = new QWidget();
    QVBoxLayout *vPause = new QVBoxLayout(panelPause);
    vPause->setContentsMargins(0, 0, 0, 10); vPause->setSpacing(8);
    btnPause1->setText("⏸ 暂停 CAM 01"); btnPause1->setStyleSheet(innerBtnStyle);
    btnPause2->setText("⏸ 暂停 CAM 02"); btnPause2->setStyleSheet(innerBtnStyle);
    btnPause3->setText("⏸ 暂停 CAM 03"); btnPause3->setStyleSheet(innerBtnStyle);
    vPause->addWidget(btnPause1); vPause->addWidget(btnPause2); vPause->addWidget(btnPause3);
    panelPause->hide();
    sidebarV->addWidget(btnNavPause); sidebarV->addWidget(panelPause);

    connect(btnNavPause, &QPushButton::clicked, this, [=](){
        bool v = panelPause->isVisible(); panelPause->setVisible(!v);
        btnNavPause->setText(v ? "⏸ 暂停控制 " : "⏸ 暂停控制 ");
        btnNavPause->setStyleSheet(v ? navStyle : activeNavStyle);
    });

    // ----------------------------------------------------
    // [折叠组 5] 系统设置 (流选择与参数)
    // ----------------------------------------------------
    btnNavConfig = new QPushButton("⚙️ 系统设置 ▼");
    btnNavConfig->setStyleSheet(navStyle);
    panelConfig = new QWidget();
    QVBoxLayout *vConfig = new QVBoxLayout(panelConfig);
    vConfig->setContentsMargins(15, 0, 5, 10);

    QString groupStyle = "QGroupBox { border: 1px solid #1c2541; border-radius: 5px; margin-top: 10px; color: #5c8df6; font-size: 12px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; }";
    
    QGroupBox *sourceGroup = new QGroupBox("数据源选择");
    sourceGroup->setStyleSheet(groupStyle);
    QVBoxLayout *srcV = new QVBoxLayout(sourceGroup);
    srcV->addWidget(new QLabel("CAM 01:")); srcV->addWidget(cbSource1_1); srcV->addWidget(cbSource1_2);
    srcV->addWidget(new QLabel("CAM 02:")); srcV->addWidget(cbSource2_1); srcV->addWidget(cbSource2_2);
    vConfig->addWidget(sourceGroup);

   // --- 算法参数部分修复 ---
    QGroupBox *paramGroup = new QGroupBox("算法参数");
    paramGroup->setStyleSheet(groupStyle);
    
    // 1. 关键修复：必须先创建 QFormLayout 并设置给 paramGroup
    QFormLayout *form = new QFormLayout(paramGroup); 
    form->setContentsMargins(10, 15, 10, 10);
    form->setSpacing(10);

    // 2. 类别选择按钮
    btnOpenClassDialog = new QPushButton("选择类别");
    btnOpenClassDialog->setStyleSheet(innerBtnStyle); 
    form->addRow("检测目标:", btnOpenClassDialog);

    // 3. 状态显示（确保 lblSelectedStatus 已在 mainwindow.h 中声明为类成员）
    lblSelectedStatus = new QLabel("当前选择: 0 类");
    lblSelectedStatus->setStyleSheet("color: #888; font-size: 11px;");
    form->addRow("", lblSelectedStatus);

    // 4. 滑块控制
    form->addRow("置信度:", sliderConf);
    form->addRow("IOU:", sliderIOU);
    // form->addRow("", chkGPU);
    vConfig->addWidget(paramGroup);

    panelConfig->hide();
    sidebarV->addWidget(btnNavConfig); sidebarV->addWidget(panelConfig);

    connect(btnNavConfig, &QPushButton::clicked, this, [=](){
        bool v = panelConfig->isVisible(); panelConfig->setVisible(!v);
        btnNavConfig->setText(v ? "⚙️ 系统设置 ▼" : "⚙️ 系统设置 ▲");
        btnNavConfig->setStyleSheet(v ? navStyle : activeNavStyle);
    });

    sidebarV->addStretch(); // 底部推起

    // 底部系统状态栏 (保留图片风格，设为可控变量)
    QLabel *lblStatusTitle = new QLabel("系统运行状态");
    lblStatusTitle->setStyleSheet("color: #5c8df6; font-weight: bold; border: none; margin-bottom: 5px;");
    sidebarV->addWidget(lblStatusTitle);
    
    // QString statusStyle = "color: #888; font-size: 12px; border: none;";
    // lblCpu = new QLabel("⚙ CPU使用率:              32%"); lblCpu->setStyleSheet(statusStyle);
    // lblMem = new QLabel("💾 内存使用率:              45%"); lblMem->setStyleSheet(statusStyle);
    // lblGpu = new QLabel("🎮 GPU使用率:              28%"); lblGpu->setStyleSheet(statusStyle);
    // sidebarV->addWidget(lblCpu);
    // sidebarV->addWidget(lblMem);
    // sidebarV->addWidget(lblGpu);

    mainLayout->addWidget(sidebarWidget);

    // ==========================================
    // 第二部分：右侧核心内容区
    // ==========================================
    QHBoxLayout *contentLayout = new QHBoxLayout();
    contentLayout->setContentsMargins(0, 15, 0, 0);
    contentLayout->setSpacing(15);

    // --- 2.1 中间区：主视频 + 底部信息框 ---
    QVBoxLayout *centerLayout = new QVBoxLayout();
    
    QLabel *lblMainTitle = new QLabel("❖ 融合跟踪结果 (主视图)");
    lblMainTitle->setStyleSheet("color: #5c8df6; font-size: 14px; font-weight: bold;");
    videoMain->setStyleSheet("background-color: #000; border: 1px solid #2e3a5a; border-radius: 8px;");
    centerLayout->addWidget(lblMainTitle);
    centerLayout->addWidget(videoMain, 8); // 视频占高度 8

    // 底部信息面板
    QWidget *bottomInfoWidget = new QWidget();
    bottomInfoWidget->setStyleSheet("background-color: #0d1224; border: 1px solid #1c2541; border-radius: 8px;");
    QHBoxLayout *bottomInfoH = new QHBoxLayout(bottomInfoWidget);
    
    QVBoxLayout *trackInfoV = new QVBoxLayout();
    QLabel *lblTrackTitle = new QLabel("🎯 跟踪信息"); lblTrackTitle->setStyleSheet("color: #5c8df6; font-weight:bold; border:none;");
    trackInfoV->addWidget(lblTrackTitle);
    // 🚀 新增：使用 QTextEdit 作为动态列表
    txtTrackInfo = new QTextEdit();
    txtTrackInfo->setReadOnly(true); // 设为只读
    // 样式设置为透明背景、无边框，看起来就像一个高级的动态 Label，且自带滚动条
    txtTrackInfo->setStyleSheet("background-color: transparent; border: none; color: #cfcfcf; font-family: 'Consolas'; font-size: 13px;");

    trackInfoV->addWidget(txtTrackInfo);
    bottomInfoH->addLayout(trackInfoV);
  

    QVBoxLayout *sysInfoV = new QVBoxLayout();
    QLabel *lblSysInfoTitle = new QLabel("📉 系统信息"); lblSysInfoTitle->setStyleSheet("color: #5c8df6; font-weight:bold; border:none;");
    // sysInfoV->addWidget(lblSysInfoTitle);
    
    // // 初始化系统性能数据为变量
    // lblSysFps = new QLabel("融合帧率: 25.7 FPS");
    // lblSysDelay = new QLabel("处理延迟: 38 ms");
    // sysInfoV->addWidget(lblSysFps);
    // sysInfoV->addWidget(lblSysDelay);
    // bottomInfoH->addLayout(sysInfoV);
    
    centerLayout->addWidget(bottomInfoWidget, 2);

    // --- 2.2 右侧边区：双视频 + 底部日志 ---
    QVBoxLayout *rightLayout = new QVBoxLayout();
    
    QLabel *lblSub1Title = new QLabel(" 可见光 检测结果");
    lblSub1Title->setStyleSheet("color: #5c8df6; font-size: 13px; font-weight: bold;");
    video1->setStyleSheet("background-color: #000; border: 1px solid #2e3a5a; border-radius: 8px;");
    
    QLabel *lblSub2Title = new QLabel("红外 检测结果");
    lblSub2Title->setStyleSheet("color: #5c8df6; font-size: 13px; font-weight: bold;");
    video2->setStyleSheet("background-color: #000; border: 1px solid #2e3a5a; border-radius: 8px;");

    // 日志区放在最下方
    QGroupBox *logGroup = new QGroupBox("实时监控日志");
    logGroup->setStyleSheet("QGroupBox { border: 1px solid #2e3a5a; border-radius: 5px; color: #5c8df6; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; }");
    QVBoxLayout *logV = new QVBoxLayout(logGroup);
    logV->setContentsMargins(5, 15, 5, 5);
    logArea->setStyleSheet("background-color: #05070d; border: none; color: #00ff00; font-family: Consolas;");
    logV->addWidget(logArea);

    rightLayout->addWidget(lblSub1Title);
    rightLayout->addWidget(video1, 3);
    rightLayout->addSpacing(10);
    rightLayout->addWidget(lblSub2Title);
    rightLayout->addWidget(video2, 3);
    rightLayout->addSpacing(10);
    rightLayout->addWidget(logGroup, 2);

    contentLayout->addLayout(centerLayout, 7);
    contentLayout->addLayout(rightLayout, 3);

    mainLayout->addLayout(contentLayout);

    // ==========================================
    // 第三部分：全局善后
    // ==========================================
    
    // 状态栏
    statusBar()->addPermanentWidget(infoLabel);
    statusBar()->setStyleSheet("background-color: #0b0e1a; color: #888; border-top: 1px solid #1c2541;");

    videoMain->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    video1->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    video2->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
}
void MainWindow::setupConnections()
{
    connect(btnStart1, &QPushButton::clicked, this, &MainWindow::onBtnStartClicked_cam1);
    connect(btnStart2, &QPushButton::clicked, this, &MainWindow::onBtnStartClicked_cam2);
    connect(btnStart3, &QPushButton::clicked, this, &MainWindow::onBtnStartClicked_cam3);
    connect(btnPause1, &QPushButton::clicked, this, &MainWindow::onBtnPauseClicked_cam1);
    connect(btnPause2, &QPushButton::clicked, this, &MainWindow::onBtnPauseClicked_cam2);
    connect(btnPause3, &QPushButton::clicked, this, &MainWindow::onBtnPauseClicked_cam3);
    connect(cbSource1_1, &QComboBox::currentTextChanged,
            this, &MainWindow::onTypeChanged);
    connect(cbSource2_1, &QComboBox::currentTextChanged,
            this, &MainWindow::onTypeChanged);
    connect(cbSource1_2, &QComboBox::currentTextChanged,
            this, &MainWindow::onOptionChanged);
    connect(cbSource2_2, &QComboBox::currentTextChanged,
            this, &MainWindow::onOptionChanged);
    // 在 setupConnections 中
    connect(btnOpenClassDialog, &QPushButton::clicked, this, [=](){
        ClassSelectDialog dlg(MainWindow::cnClasses, m_selectedClassIds, this);
        if (dlg.exec() == QDialog::Accepted) {
            m_selectedClassIds = dlg.getSelectedIds();
            
            // 更新界面显示文字
            if (lblSelectedStatus) {
                lblSelectedStatus->setText(QString("已选择: %1 类").arg(m_selectedClassIds.size()));
            }
            
            syncConfigToAI(); 
            
            logArea->append(QString("<font color='#00ff00'>[System] 目标筛选已更新，当前监测 %1 类</font>")
                            .arg(m_selectedClassIds.size()));
        }
    });
    connect(m_timer_cam1, &QTimer::timeout, this, &MainWindow::updateDisplay_cam1);
    connect(m_timer_cam2, &QTimer::timeout, this, &MainWindow::updateDisplay_cam2);
    connect(m_timer_cam3, &QTimer::timeout, this, &MainWindow::updateDisplay_cam3);

    connect(sliderConf, &QSlider::valueChanged, this, &MainWindow::onConfSliderChanged);
    connect(sliderIOU, &QSlider::valueChanged, this, &MainWindow::onIOUSliderChanged);

}
void MainWindow::initStreamTypeComboBoxes(
    const std::vector<StreamGroup> &streamGroups,
    QComboBox *cb1_1, QComboBox *cb1_2,
    QComboBox *cb2_1, QComboBox *cb2_2)
{
    cb1_1->clear();
    cb1_2->clear();



    // ========= 1️⃣ 填充类型 =========
    for (const auto &g : streamGroups)
    {
        QString name = QString(rs2_stream_to_string(g.streamType));
        cb1_1->addItem(name);
        cb2_1->addItem(name);
    }

    // ========= 2️⃣ 根据当前选择填充第二个 (注意这里按值捕获 streamGroups) =========
    auto fillOptions = [streamGroups](QComboBox *cbType, QComboBox *cbOpt)
    {
        QString type = cbType->currentText();
        cbOpt->clear();

        for (const auto &g : streamGroups)
        {
            if (QString(rs2_stream_to_string(g.streamType)) == type)
            {
                for (const auto &opt : g.options)
                {
                    QString text = QString("%1x%2 @%3 (%4)")
                                       .arg(opt.width)
                                       .arg(opt.height)
                                       .arg(opt.fps)
                                       .arg(QString(rs2_format_to_string(opt.format)));

                    // 将配置结构体作为 UserData 存入
                    cbOpt->addItem(text, QVariant::fromValue(opt));
                }
                break;
            }
        }
    };

    // ========= 3️⃣ 初始化填充并同步 currentGroup1 =========
    if (cb1_1->count() > 0)
    {
        cb1_1->setCurrentIndex(0);
        fillOptions(cb1_1, cb1_2);

        // 从默认选中的项中初始化 currentGroup1
        if (cb1_2->count() > 0)
        {
            cb1_2->setCurrentIndex(0); // 确保有默认选中项
            QString type1 = cb1_1->currentText();

            for (const auto &g : streamGroups)
            {
                if (QString(rs2_stream_to_string(g.streamType)) == type1)
                {
                    currentGroup1.streamType = g.streamType;
                    currentGroup1.options.clear();
                    // 直接从 UserData 中提取 StreamOption
                    currentGroup1.options.push_back(cb1_2->currentData().value<StreamOption>());
                    break;
                }
            }
        }
    }

    // ========= 4️⃣ 初始化填充并同步 currentGroup2 =========
    if (cb2_1->count() > 0)
    {
        cb2_1->setCurrentIndex(1);
        fillOptions(cb2_1, cb2_2);

        // 从默认选中的项中初始化 currentGroup2
        if (cb2_2->count() > 0)
        {
            cb2_2->setCurrentIndex(0);// 确保有默认选中项
            QString type2 = cb2_1->currentText();

            for (const auto &g : streamGroups)
            {
                if (QString(rs2_stream_to_string(g.streamType)) == type2)
                {
                    currentGroup2.streamType = g.streamType;
                    currentGroup2.options.clear();
                    // 直接从 UserData 中提取 StreamOption
                    currentGroup2.options.push_back(cb2_2->currentData().value<StreamOption>());
                    break;
                }
            }
        }
    }
}
void MainWindow::initData()
{  
    try{
        yolov8Config initial_cfg;
        initial_cfg.probabilityThreshold = sliderConf->value() / 100.0f;
        initial_cfg.nmsThreshold = sliderIOU->value() / 100.0f;
        // === 新增：默认全选 80 个类别 ===
        if (m_selectedClassIds.isEmpty()) {
            for (int i = 0; i < 80; ++i) {
                m_selectedClassIds.append(i);
            }
        }
        m_input_queue = std::make_unique<queque_mutex<input_alo>>();
        m_output_queue = std::make_unique<queque_mutex<output_alo>>();
        this->rsController_cam = std::make_unique<RealSenseController>();
        if(this->rsController_cam){
            this->rsController_cam->init(this->m_input_queue.get());
            this->rsController_cam->get_supported_streams(streamGroups);
        }else{
            throw std::runtime_error("RealSenseController init failed");
        }
        this->ai=std::make_unique<Algorithm>();
        if(this->ai){
            this->ai->init(this->m_input_queue.get(), this->m_output_queue.get(),initial_cfg);
        }else{
            throw std::runtime_error("Algorithm init failed");
        }
        m_timer_cam1 = new QTimer(this);
        m_timer_cam2 = new QTimer(this);
        m_timer_cam3 = new QTimer(this);
        this->initStreamTypeComboBoxes(
            streamGroups,
            cbSource1_1, cbSource1_2,
            cbSource2_1, cbSource2_2);
    }catch(const std::exception& e){
       // 弹出对话框提示用户
        QMessageBox::critical(this, "初始化失败", 
            QString("程序无法启动，原因：\n%1").arg(e.what()));
        
        // 打印到日志区方便排查
        logArea->append(QString("<font color='red'>[Fatal] %1</font>").arg(e.what()));

        // 如果初始化失败，通常需要禁用开始按钮，防止用户强行操作导致崩溃
        btnStart1->setEnabled(false);
        btnStart2->setEnabled(false);
        btnStart3->setEnabled(false);
        btnPause1->setEnabled(false);
        btnPause2->setEnabled(false);
        btnPause3->setEnabled(false);
        return; // 退出函数，不再执行后续逻辑
    }
    }




void MainWindow::onTypeChanged()
{
    QComboBox *senderCombo = qobject_cast<QComboBox *>(sender());
    if (!senderCombo)
        return;

    // =========================
    // 🟢 Cam1 (cbSource1_1 改变 -> 更新 cbSource1_2)
    // =========================
    if (senderCombo == cbSource1_1)
    {
        QString type1 = cbSource1_1->currentText();
        cbSource1_2->clear(); // 清空旧数据

        for (const auto &g : streamGroups)
        {
            if (QString(rs2_stream_to_string(g.streamType)) == type1)
            {
                // 1. 保存当前选择的流类型
                currentGroup1.streamType = g.streamType;
                currentGroup1.options.clear();

                // 2. 将该类型下的所有配置项填充到第二个下拉框
                for (const auto &opt : g.options)
                {
                    QString text = QString("%1x%2 @%3 (%4)")
                                       .arg(opt.width)
                                       .arg(opt.height)
                                       .arg(opt.fps)
                                       .arg(QString(rs2_format_to_string(opt.format)));

                    // 注意：这里将 opt 结构体作为 UserData 存入 ComboBox
                    cbSource1_2->addItem(text, QVariant::fromValue(opt));
                }
                break;
            }
        }
    }
    // =========================
    // 🔵 Cam2 (cbSource2_1 改变 -> 更新 cbSource2_2)
    // =========================
    else if (senderCombo == cbSource2_1)
    {
        QString type2 = cbSource2_1->currentText();
        cbSource2_2->clear();

        for (const auto &g : streamGroups)
        {
            if (QString(rs2_stream_to_string(g.streamType)) == type2)
            {
                currentGroup2.streamType = g.streamType;
                currentGroup2.options.clear();

                for (const auto &opt : g.options)
                {
                    QString text = QString("%1x%2 @%3 (%4)")
                                       .arg(opt.width)
                                       .arg(opt.height)
                                       .arg(opt.fps)
                                       .arg(QString(rs2_format_to_string(opt.format)));

                    cbSource2_2->addItem(text, QVariant::fromValue(opt));
                }
                break;
            }
        }
    }
}
void MainWindow::onOptionChanged()
{
    QComboBox *senderCombo = qobject_cast<QComboBox *>(sender());
    if (!senderCombo)
        return;

    // cbSource 改变时（比如被 clear），currentIndex 会变成 -1，需要拦截
    if (senderCombo->currentIndex() < 0)
        return;

    // =========================
    // 🟢 Cam1 Option (cbSource1_2 改变 -> 保存到 currentGroup1)
    // =========================
    if (senderCombo == cbSource1_2)
    {
        // 从 ComboBox 的 UserData 中提取 StreamOption 结构体
        StreamOption selectedOpt = cbSource1_2->currentData().value<StreamOption>();

        // 永远只保留当前选中的这一个
        currentGroup1.options.clear();
        currentGroup1.options.push_back(selectedOpt);

        // 可选：打印调试信息验证是否成功
        qDebug() << "Cam1 Selected:" << selectedOpt.width << "x" << selectedOpt.height;
    }
    // =========================
    // 🔵 Cam2 Option (cbSource2_2 改变 -> 保存到 currentGroup2)
    // =========================
    else if (senderCombo == cbSource2_2)
    {
        StreamOption selectedOpt = cbSource2_2->currentData().value<StreamOption>();

        currentGroup2.options.clear();
        currentGroup2.options.push_back(selectedOpt);
    }
}
void MainWindow::onBtnStartClicked_cam1()
{
    // 1️⃣ 检查并启动硬件线程 (无论点哪个，没开就开)
    if (!rsController_cam->is_active())
    {
        if (!rsController_cam->start(currentGroup1, currentGroup2))
        {
            logArea->append("<font color='red'>[System] 硬件启动失败！</font>");
            return;
        }
        if(!ai->is_active()){
            if (!ai->start())
        {
            logArea->append("<font color='red'>[System] 算法启动失败！</font>");
            return;
        }
        }
    }

    // 2️⃣ 逻辑层：标记 Cam1 开始显示
    cam1Running = true;

    // 3️⃣ UI 反馈：锁定自己的配置，切换按钮状态
    cbSource1_1->setEnabled(false);
    cbSource1_2->setEnabled(false);
    btnStart1->setEnabled(false);
    btnPause1->setEnabled(true);

    // 4️⃣ 启动消费定时器 (去 updateDisplay_cam1 拿数据)
    m_timer_cam1->start(15);
    logArea->append("<font color='#00ff00'>[Cam1] 预览已启动</font>");
}

void MainWindow::onBtnStartClicked_cam2()
{
    // 1️⃣ 检查并启动硬件线程 (如果 Cam1 已经开了，这里直接跳过硬件操作)
    if (!rsController_cam->is_active())
    {
        if (!rsController_cam->start(currentGroup1, currentGroup2))
        {
            logArea->append("<font color='red'>[System] 硬件启动失败！</font>");
            return;
        }
         if(!ai->is_active()){
            if (!ai->start()){
            logArea->append("<font color='red'>[System] 算法启动失败！</font>");
            return;
        }
        }
    }

    // 2️⃣ 逻辑层
    cam2Running = true;

    // 3️⃣ UI 反馈
    cbSource2_1->setEnabled(false);
    cbSource2_2->setEnabled(false);
    btnStart2->setEnabled(false);
    btnPause2->setEnabled(true);

    // 4️⃣ 启动消费定时器
    m_timer_cam2->start(15);
    logArea->append("<font color='#00ff00'>[Cam2] 预览已启动</font>");
}
void MainWindow::onBtnStartClicked_cam3()
{
    // 1️⃣ 检查并启动硬件线程 (如果 Cam1 已经开了，这里直接跳过硬件操作)
    if (!rsController_cam->is_active())
    {
        if (!rsController_cam->start(currentGroup1, currentGroup2))
        {
            logArea->append("<font color='red'>[System] 硬件启动失败！</font>");
            return;
        }
         if(!ai->is_active()){
            if (!ai->start()){
            logArea->append("<font color='red'>[System] 算法启动失败！</font>");
            return;
        }
        }
    }

    // 2️⃣ 逻辑层
    cam3Running = true;

    // 3️⃣ UI 反馈

    btnStart3->setEnabled(false);
    btnPause3->setEnabled(true);

    // 4️⃣ 启动消费定时器
    m_timer_cam3->start(15);
    logArea->append("<font color='#00ff00'>[Cam3] 预览已启动</font>");
}

void MainWindow::onBtnPauseClicked_cam1()
{
    if (cam1Running) {
        cam1Running = false; // 标记 cam1 停止
        btnPause1->setEnabled(false);   // 恢复开始按钮
        btnStart1->setEnabled(true);  // 禁用暂停按钮
        
        logArea->append("<font color='orange'>[Camera 1] 暂停信号已发送，等待系统释放...</font>");
        
        // 关键：检查是否可以关闭底层硬件
        checkAndStopHardware();
    }
}

void MainWindow::onBtnPauseClicked_cam2()
{
    if (cam2Running) {
        cam2Running = false; // 标记 cam2 停止
        btnPause2->setEnabled(false);
        btnStart2->setEnabled(true);
        
        logArea->append("<font color='orange'>[Camera 2] 暂停信号已发送，等待系统释放...</font>");
        
        // 关键：检查是否可以关闭底层硬件
        checkAndStopHardware();
    }
}
void MainWindow::onBtnPauseClicked_cam3()
{
    if (cam3Running) {
        cam3Running = false; // 标记 cam3 停止
        btnPause3->setEnabled(false);
        btnStart3->setEnabled(true);
        
        logArea->append("<font color='orange'>[Camera 3] 暂停信号已发送，等待系统释放...</font>");
        
        // 关键：检查是否可以关闭底层硬件
        checkAndStopHardware();
    }
}
// 🚀 将 OpenCV 的 Mat 转换为 Qt 的 QImage
QImage MainWindow::cvMatToQImage(const cv::Mat &mat) {
    switch (mat.type()) {
        // 8-bit, 3 channel (彩色图)
            case CV_8UC3: {
         
            return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_RGB888).copy();
        }
        // 8-bit, 1 channel (红外图/灰度图)
        case CV_8UC1: {
            return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8);
        }
        // 16-bit, 1 channel (深度图)
        case CV_16UC1: {
            // 深度图直接显示会全黑（因为值很大），这里先转为 8 位显示
            cv::Mat depth8;
            mat.convertTo(depth8, CV_8U, 0.05); // 缩放因子可以根据实际距离调整
            return QImage(depth8.data, depth8.cols, depth8.rows, static_cast<int>(depth8.step), QImage::Format_Grayscale8).copy();
        }
        default:
            return QImage();
    }
}
void MainWindow::updateDisplay_cam1()
{
    if (!cam1Running) return;
// printf("updateDisplay_cam1\n"); 
    output_alo current_pack;
    int last_seq = 0;

    // 直接从 AI 输出队列获取打包好的数据 (包含图+框+深度)
    if (ai && ai->get_latest_output(current_pack, last_seq))
    {
        // 1. 获取目标图像 (Cam1 通常显示 Color)
        cv::Mat mat_to_show = current_pack.imgs.color;
        int batch_idx = 0; // 彩色图对应 batch 0

        if (mat_to_show.empty()) return;

        // 2. 将 Mat 转为 QImage
        QImage img = cvMatToQImage(mat_to_show);

        // 3. 画框逻辑 (使用当前包内自带的 detection 结果)
        drawYoloResults(img, current_pack, batch_idx);

        // 4. 显示
        video1->setPixmap(QPixmap::fromImage(img).scaled(
            video1->contentsRect().size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

void MainWindow::updateDisplay_cam2()
{
    if (!cam2Running) return;

    output_alo current_pack;
    int last_seq = 0;

    if (ai && ai->get_latest_output(current_pack, last_seq))
    {
        printf("updateDisplay_cam2\n");
        // 1. 获取红外图 (Cam2 显示 Infrared)
        cv::Mat ir_mat = current_pack.imgs.infrared;
        int batch_idx = 1; // 红外图对应 batch 1

        if (ir_mat.empty()) return;

        // 🚀 核心逻辑：将红外图转回单通道进行显示 (如果之前为了推理转成了 3 通道)
        cv::Mat show_mat;
        if (ir_mat.channels() == 3) {
            cv::cvtColor(ir_mat, show_mat, cv::COLOR_BGR2GRAY);
        } else {
            show_mat = ir_mat;
        }

        // 2. 转为 QImage (单通道对应 Format_Grayscale8)
        QImage img(show_mat.data, show_mat.cols, show_mat.rows, show_mat.step, QImage::Format_Grayscale8);
        
        // 为了能在灰度图上画彩色框，我们需要把 QImage 转回 RGB 格式画图
        QImage drawImg = img.convertToFormat(QImage::Format_RGB888);

        // 3. 画框逻辑
        drawYoloResults(drawImg, current_pack, batch_idx);

        // 4. 显示
        video2->setPixmap(QPixmap::fromImage(drawImg).scaled(
            video2->contentsRect().size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}
void MainWindow::updateDisplay_cam3()
{
    if (!cam3Running) return;
// printf("updateDisplay_cam3\n"); 
    output_alo current_pack;
    int last_seq = 0;

    // 直接从 AI 输出队列获取打包好的数据 (包含图+框+深度)
    if (ai && ai->get_latest_output(current_pack, last_seq))
    {
        // --- 1. 立即清空文本框 ---
        txtTrackInfo->clear();
        cv::Mat mat_to_show = current_pack.imgs.color;
        int batch_idx = 2; // 主画面使用 sort 输出批次索引 2

        if (mat_to_show.empty()) return;

        // 2. 将 Mat 转为 QImage
        QImage img = cvMatToQImage(mat_to_show);
        printf("进入画图\n");
        // 3. 画框逻辑 (使用当前包内自带的 detection 结果)
        drawYoloResults(img, current_pack, batch_idx);

        // 4. 显示
        videoMain->setPixmap(QPixmap::fromImage(img).scaled(
            videoMain->contentsRect().size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        int num_tracks = current_pack.dections_sums[2];
        
        if (num_tracks <= 0) {
            txtTrackInfo->setHtml("<font color='#888'>等待追踪目标...</font>");
            // 这里记得也要更新视频显示，否则画面会卡在最后一帧有框的状态
            return;
        }

        QString trackDisplayStr;

        for (int i = 0; i < num_tracks; ++i) {
            yolo_dection det = current_pack.sort_output[i];

            // 过滤无效数据：如果 ID 为 0 或置信度太低，说明是数组残留
            if (det.id <= 0 || det.conf < 0.1) continue; 
            if (!m_selectedClassIds.contains(static_cast<int>(det.cls))) {
                            continue; 
                        }
            int clsIdx = static_cast<int>(det.cls);
            QString clsName = (clsIdx >= 0 && clsIdx < cnClasses.size()) ? cnClasses[clsIdx] : "未知";

            trackDisplayStr += QString(
                "<div style='margin:2px; padding:2px; background:#151a30;'>"
                "<b style='color:#5c8df6;'>ID: %1</b> | "
                "<b style='color:#00ff00;'>%2</b> | "
                "<span style='color:#aaa;'>%3</span>"
                "</div>")
                .arg(static_cast<int>(det.id))
                .arg(clsName)
                .arg(det.conf, 0, 'f', 2);
        }
        
        // --- 2. 重新塞入内容 ---
        txtTrackInfo->setHtml(trackDisplayStr);
        
    }
}

void MainWindow::drawYoloResults(QImage& img, const output_alo& pack, int batch_idx)
{
    if (pack.dections_sums.empty() || batch_idx >= pack.dections_sums.size()) {
        printf("drawYoloResults: batch_idx %d out of range\n", batch_idx);
        return;
    }

    QPainter painter(&img);
    painter.setRenderHint(QPainter::Antialiasing);
    QPen pen(Qt::green, 3);
    painter.setPen(pen);

    // 计算该 batch 的偏移
    yolo_dection det;
    int offset = 0;
    if(batch_idx == 2){
        offset = 0;
    }else{

    for (int b = 0; b < batch_idx; ++b) {
        offset += pack.dections_sums[b];
    }
    }


    int num_boxes = pack.dections_sums[batch_idx];

    for (int i = 0; i < num_boxes; ++i) {
        if (batch_idx == 2) {
            det = pack.sort_output[offset + i];
        } else {
            det = pack.yolo_output[offset + i];
        }
        // printf("检测到目标: ID=%d, 类别=%d, 置信度=%.2f\n", static_cast<int>(det.id), static_cast<int>(det.cls), det.conf);
        if (!m_selectedClassIds.contains(static_cast<int>(det.cls))) {
            continue; // 如果没勾选这个类别，直接跳过后续的画框和标字操作！
        }

        int x = static_cast<int>(det.x - det.w / 2.0f);
        int y = static_cast<int>(det.y - det.h / 2.0f);

        float dist = 0.0f;
        if (!pack.imgs.depth.empty()) {
            unsigned short d_raw = pack.imgs.depth.at<unsigned short>(static_cast<int>(det.y), static_cast<int>(det.x));
            dist = d_raw * 0.001f;
        }

        painter.drawRect(x, y, static_cast<int>(det.w), static_cast<int>(det.h));

        QString label = QString("ID:%1 %2m").arg(det.id).arg(dist, 0, 'f', 2);
        painter.fillRect(x, y - 22, 180, 22, QColor(0, 255, 0, 160));
        painter.setPen(Qt::black);
        painter.drawText(x + 5, y - 6, label);
        painter.setPen(pen);
        // printf("检测到目标: ID=%d, 类别=%d, 置信度=%.2f\n", static_cast<int>(det.id), static_cast<int>(det.cls), det.conf);
    }
    painter.end();
}

void MainWindow::checkAndStopHardware()
{
    if (!cam1Running && !cam2Running && !cam3Running)
    {
        // 1. 先停算法（消费者）
        if (this->ai && this->ai->is_active())
        {
            if (this->ai->stop()) { // 停止算法的 while 循环并 join 线程
                logArea->append("<font color='blue'>[System] AI 算法线程已安全停止。</font>");
            } else {
                logArea->append("<font color='red'>[System] AI 算法线程停止失败。</font>");
            }
        }

        // 2. 后停硬件（生产者）
        if (rsController_cam && rsController_cam->is_active())
        {
            rsController_cam->stop();
            logArea->append("<font color='gray'>[System] 所有预览已关闭，硬件线程已安全停止。</font>");
        }
    }
}

// MainWindow.cpp
void MainWindow::syncConfigToAI() {
    // 构造一个新的临时配置，或者只修改动态参数
    yolov8Config dynamic_cfg = ai->get_latest_config(); // 获取当前配置
    dynamic_cfg.probabilityThreshold = sliderConf->value() / 100.0f;
    dynamic_cfg.nmsThreshold = sliderIOU->value() / 100.0f;
    dynamic_cfg.targetClassId.clear();
    std::vector<int> targetIds;
        for(int id : m_selectedClassIds) {
            targetIds.push_back(id);
        }
    dynamic_cfg.targetClassId = targetIds;
    // 一次性同步给算法引擎
    ai->updateDynamicParams(dynamic_cfg); 
}

// 槽函数就很简单了
void MainWindow::onConfSliderChanged(int value) { 
    logArea->append(QString("<font color='#55aaff'>[参数] Confidence 阈值调整为: %1%</font>").arg(value)); 
    syncConfigToAI(); 
}
void MainWindow::onIOUSliderChanged(int value) { 
    logArea->append(QString("<font color='#55aaff'>[参数] iou阈值调整为: %1%</font>").arg(value));     
    syncConfigToAI(); 

}



