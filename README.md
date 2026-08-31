# LocateCam 随处找

手机本地离线的开放词汇实时目标检测 App。摄像头实时画面 + 中文输入要找的物体 + 实时画框 + 延迟/帧率显示。

- 模型：YOLO-World v2 (yolov8s-worldv2) int8 量化，约 400 个中英文常用词烘焙为类别
- 推理：ONNX Runtime Android（NNAPI 加速，CPU 回退）
- 全程离线，无任何联网权限

词库见 tools/vocab_zh.json，修改后自动触发云端重新编译出 APK（GitHub Actions）。
