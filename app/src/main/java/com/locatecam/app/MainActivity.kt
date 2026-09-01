package com.locatecam.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var hudText: TextView
    private lateinit var inputEdit: EditText

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    @Volatile private var engine: DetectionEngine? = null
    @Volatile private var tracker: TrackerEngine? = null
    @Volatile private var vocab: Vocab? = null
    private var fpsEma = 0f
    private var lastFrameAt = 0L
    private var lastProcessAt = 0L
    private val rgb = IntArray(YuvToRgb.OUT * YuvToRgb.OUT)
    private var fullFrame = IntArray(480 * 640)
    private var fullW = 480
    private var fullH = 640

    @Volatile private var tracking = false
    @Volatile private var lockedLabel = ""
    private var lowStreak = 0

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            start()
        } else {
            Toast.makeText(this, "需要相机权限才能使用", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        hudText = findViewById(R.id.hudText)
        inputEdit = findViewById(R.id.inputEdit)
        findViewById<Button>(R.id.btnApply).setOnClickListener { applyInput() }
        inputEdit.setText("手机, 人, 杯子")

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            start()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun start() {
        hudText.text = "模型加载中，请稍候…"
        Thread {
            try {
                val v = Vocab(this)
                vocab = v
                val e = DetectionEngine(this)
                e.selectedIndices = findIndices(listOf("cell phone", "person", "cup", "book"))
                e.warmup()
                engine = e
                var t: TrackerEngine? = null
                try {
                    t = TrackerEngine(this)
                } catch (te: Throwable) {
                    Log.w(TAG, "tracker unavailable: ${te.message}")
                }
                tracker = t
                runOnUiThread {
                    hudText.text = if (t != null) "就绪（检测+跟踪），开始识别：手机 / 人 / 杯子 / 书" else "就绪（仅检测）"
                    startCamera()
                }
            } catch (t: Throwable) {
                Log.e(TAG, "init failed", t)
                runOnUiThread { hudText.text = "初始化失败：${t.javaClass.simpleName}: ${t.message}" }
            }
        }.start()
    }

    private fun findIndices(terms: List<String>): List<Int> {
        val v = vocab ?: return emptyList()
        return terms.mapNotNull { t ->
            val i = v.indexOf(t)
            if (i >= 0) i else null
        }.distinct()
    }

    private fun applyInput() {
        val e = engine ?: return
        val text = inputEdit.text.toString()
        val terms = text.split(",", "，", "、", " ", "/").map { it.trim() }.filter { it.isNotEmpty() }
        if (terms.isEmpty()) {
            Toast.makeText(this, "请输入要找的物体，如：手机, 钥匙, 猫", Toast.LENGTH_SHORT).show()
            return
        }
        val v = vocab ?: return
        val matched = ArrayList<Int>()
        val missing = ArrayList<String>()
        for (t in terms) {
            val i = v.indexOf(t)
            if (i >= 0) matched.add(i) else missing.add(t)
        }
        if (matched.isEmpty()) {
            Toast.makeText(
                this,
                "词库中找不到：${missing.joinToString("、")}\n可试试：手机 人 猫 狗 钥匙 遥控器",
                Toast.LENGTH_LONG
            ).show()
            return
        }
        e.selectedIndices = matched.distinct()
        val msg = if (missing.isEmpty()) {
            "开始识别：${matched.joinToString("、") { v.display(it) }}"
        } else {
            "开始识别：${matched.joinToString("、") { v.display(it) }}\n未收录已忽略：${missing.joinToString("、")}"
        }
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(cameraExecutor, ::onFrame)
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } catch (t: Throwable) {
                Log.e(TAG, "camera failed", t)
                hudText.text = "相机启动失败：${t.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onFrame(image: ImageProxy) {
        val e = engine
        if (e == null) {
            image.close()
            return
        }
        val t = tracker
        val interval = if (tracking && t != null) TRACK_THROTTLE_MS else THROTTLE_MS
        val nowMs = System.currentTimeMillis()
        if (nowMs - lastProcessAt < interval || !busy.compareAndSet(false, true)) {
            image.close()
            return
        }
        lastProcessAt = nowMs
        try {
            if (tracking && t != null) {
                val info = YuvToRgb.frameInfo(image)
                if (fullW != info.w || fullH != info.h) {
                    fullW = info.w
                    fullH = info.h
                    fullFrame = IntArray(fullW * fullH)
                }
                try {
                    YuvToRgb.convertFull(image, fullFrame)
                } finally {
                    image.close()
                }
                val res = t.track(fullFrame, fullW, fullH)
                if (res.box == null || res.score < 0.25f) {
                    lowStreak++
                } else {
                    lowStreak = 0
                }
                val lost = lowStreak >= 5
                runOnUiThread {
                    if (lost) {
                        tracking = false
                        overlayView.clearTarget()
                        Toast.makeText(this@MainActivity, "目标丢失，重新搜索", Toast.LENGTH_SHORT).show()
                    } else {
                        overlayView.setTarget(res.box, lockedLabel)
                    }
                    hudText.text = String.format(
                        Locale.CHINA,
                        "锁定 %s | 跟踪 %d ms | 置信 %.2f%s",
                        lockedLabel, res.ms, res.score,
                        if (lowStreak > 0) " (弱)" else ""
                    )
                }
            } else {
                var closed = false
                try {
                    val srcInfo: YuvToRgb.FrameInfo
                    try {
                        srcInfo = YuvToRgb.convert(image, rgb)
                    } catch (ce: Throwable) {
                        image.close()
                        closed = true
                        throw ce
                    }
                    val (dets, timing) = e.detect(rgb, srcInfo)
                    val totalMs = (System.currentTimeMillis() - nowMs)
                    val now = System.currentTimeMillis()
                    if (lastFrameAt > 0) {
                        val gap = (now - lastFrameAt).coerceAtLeast(1)
                        val fps = 1000f / gap
                        fpsEma = if (fpsEma == 0f) fps else fpsEma * 0.9f + fps * 0.1f
                    }
                    lastFrameAt = now
                    val v = vocab
                    val best = dets
                        .filter { it.score >= 0.45f && it.box.width() * it.box.height() >= 0.02f * srcInfo.w * srcInfo.h }
                        .maxByOrNull { it.score }
                    var locked = false
                    if (best != null && t != null) {
                        if (fullW != srcInfo.w || fullH != srcInfo.h) {
                            fullW = srcInfo.w
                            fullH = srcInfo.h
                            fullFrame = IntArray(fullW * fullH)
                        }
                        YuvToRgb.convertFull(image, fullFrame)
                        image.close()
                        closed = true
                        t.init(fullFrame, fullW, fullH, best.box)
                        lockedLabel = v?.display(best.labelIndex) ?: "?"
                        tracking = true
                        lowStreak = 0
                        locked = true
                    }
                    runOnUiThread {
                        if (locked) {
                            Toast.makeText(this@MainActivity, "已锁定：$lockedLabel", Toast.LENGTH_SHORT).show()
                        } else {
                            hudText.text = String.format(
                                Locale.CHINA,
                                "引擎 %s | 预处理 %d ms | 推理 %d ms | 后处理 %d ms\n端到端 %d ms | %.1f FPS",
                                e.engineMode, timing.preMs, timing.inferMs, timing.postMs, totalMs, fpsEma
                            )
                            overlayView.update(dets, srcInfo.w, srcInfo.h) { i -> v?.display(i) ?: "?" }
                        }
                    }
                } finally {
                    if (!closed) image.close()
                }
            }
        } catch (t2: Throwable) {
            Log.e(TAG, "frame error", t2)
        } finally {
            busy.set(false)
        }
    }
        val nowMs = System.currentTimeMillis()
        if (nowMs - lastProcessAt < THROTTLE_MS || !busy.compareAndSet(false, true)) {
            image.close()
            return
        }
        lastProcessAt = nowMs
        try {
            val t0 = System.nanoTime()
            val srcInfo: YuvToRgb.FrameInfo
            try {
                srcInfo = YuvToRgb.convert(image, rgb)
            } finally {
                image.close()
            }
            val (dets, timing) = e.detect(rgb, srcInfo)
            val totalMs = (System.nanoTime() - t0) / 1_000_000
            val now = System.currentTimeMillis()
            if (lastFrameAt > 0) {
                val interval = (now - lastFrameAt).coerceAtLeast(1)
                val fps = 1000f / interval
                fpsEma = if (fpsEma == 0f) fps else fpsEma * 0.9f + fps * 0.1f
            }
            lastFrameAt = now
            val v = vocab
            runOnUiThread {
                hudText.text = String.format(
                    Locale.CHINA,
                    "引擎 %s | 预处理 %d ms | 推理 %d ms | 后处理 %d ms\n端到端 %d ms | %.1f FPS",
                    e.engineMode, timing.preMs, timing.inferMs, timing.postMs, totalMs, fpsEma
                )
                overlayView.update(dets, srcInfo.w, srcInfo.h) { i -> v?.display(i) ?: "?" }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "frame error", t)
        } finally {
            busy.set(false)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        engine?.close()
        tracker?.close()
    }

    companion object {
        private const val TAG = "LocateCam"
        private const val THROTTLE_MS = 1000L
        private const val TRACK_THROTTLE_MS = 400L
    }
}
