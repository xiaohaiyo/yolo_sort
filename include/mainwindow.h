#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QTimer>
#include <QList>
#include <memory>
#include <vector>
#include "RealSenseController.h"
#include "data.h"
#include "yolo_datatype.h"

namespace rs2 { class frame; class colorizer; }

class QLabel;
class QPushButton;
class QComboBox;
class QGroupBox;
class QSlider;
class QCheckBox;
class QTextEdit;
class QWidget;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private:
    void setupUi();
    void setupLayout();
    void setupConnections();
    void initData();

    void onTypeChanged();
    void onOptionChanged();
    void updateDisplay_cam1();
    void updateDisplay_cam2();
    void updateDisplay_cam3();
    void initStreamTypeComboBoxes(const std::vector<StreamGroup>&,
                                 QComboBox*, QComboBox*, QComboBox*, QComboBox*);
    void checkAndStopHardware();

    void onConfSliderChanged(int value);
    void onIOUSliderChanged(int value);
    void syncConfigToAI();

    QImage cvMatToQImage(const cv::Mat &mat);
    void drawYoloResults(QImage& img, const output_alo& pack, int batch_idx);

private:
    QLabel *video1;
    QLabel *video2;
    QLabel *videoMain;
    QTextEdit *logArea;

    QComboBox *cbSource1_1;
    QComboBox *cbSource1_2;
    QPushButton *btnStart1;
    QPushButton *btnPause1;

    QComboBox *cbSource2_1;
    QComboBox *cbSource2_2;
    QPushButton *btnStart2;
    QPushButton *btnPause2;

    QPushButton *btnStart3;
    QPushButton *btnPause3;

    QPushButton *btnOpenClassDialog;
    QLabel *lblSelectedStatus;
    QList<int> m_selectedClassIds;

    QSlider *sliderConf;
    QSlider *sliderIOU;

    QLabel *infoLabel;

    QPushButton *btnNavStart;
    QPushButton *btnNavPause;
    QPushButton *btnNavConfig;

    QWidget *panelStart;
    QWidget *panelPause;
    QWidget *panelConfig;

    QTextEdit *txtTrackInfo;

    std::vector<StreamGroup> streamGroups;
    StreamGroup currentGroup1;
    StreamGroup currentGroup2;

    bool cam1Running = false;
    bool cam2Running = false;
    bool cam3Running = false;

    QTimer *m_timer_cam1;
    QTimer *m_timer_cam2;
    QTimer *m_timer_cam3;

    std::unique_ptr<Algorithm> ai;
    std::unique_ptr<RealSenseController> rsController_cam;
    std::unique_ptr<queque_mutex<output_alo>> m_output_queue;
    std::unique_ptr<queque_mutex<input_alo>> m_input_queue;

    static const QStringList cnClasses;

private slots:
    void onBtnStartClicked_cam1();
    void onBtnStartClicked_cam2();
    void onBtnStartClicked_cam3();
    void onBtnPauseClicked_cam1();
    void onBtnPauseClicked_cam2();
    void onBtnPauseClicked_cam3();
};

#endif // MAINWINDOW_H
