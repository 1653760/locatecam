package com.locatecam.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.RectF
import android.os.Bundle
import android.os.SystemClock
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
    private lateinit var btnMic: Button

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
    @Volatile private var lockedEn = ""
    private var lowStreak = 0
    private var trackFailCount = 0

    // depth / direction
    @Volatile private var depth: DepthEngine? = null
    private var depthTick = 0
    @Volatile private var lastDistM = -1f

    // periodic class re-validation while tracking
    private var validateTick = 0
    private var classMissCount = 0

    // ---- voice (offline Vosk, hands-free continuous listening) ----
    private var voiceEngine: VoiceEngine? = null
    @Volatile private var voiceEnabled = true
    @Volatile private var lastVoiceTriggerAt = 0L

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

    private val micPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startVoiceLoop()
        } else {
            voiceEnabled = false
            btnMic.text = "🎤关"
            Toast.makeText(this, "麦克风未授权，语音功能关闭（文字输入仍可用）", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        hudText = findViewById(R.id.hudText)
        inputEdit = findViewById(R.id.inputEdit)
        btnMic = findViewById(R.id.btnMic)
        findViewById<Button>(R.id.btnApply).setOnClickListener { applyInput() }
        btnMic.setOnClickListener { toggleVoice() }
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
                var trackerErr = "未知"
                try {
                    t = TrackerEngine(this)
                } catch (te: Throwable) {
                    Log.w(TAG, "tracker unavailable: ${te.message}")
                    trackerErr = "${te.javaClass.simpleName}: ${te.message}"
                }
                tracker = t
                var d: DepthEngine? = null
                try {
                    d = DepthEngine(this)
                } catch (de: Throwable) {
                    Log.w(TAG, "depth unavailable: ${de.message}")
                }
                depth = d
                runOnUiThread {
                    hudText.text = if (t != null) "就绪（检测+跟踪${if (d != null) "+测距" else ""}就绪），说“我要找XX”或输入目标"
                    else "跟踪器加载失败（$trackerErr）"
                    startCamera()
                    maybeStartVoice()
                }
            } catch (t: Throwable) {
                Log.e(TAG, "init failed", t)
                runOnUiThread { hudText.text = "初始化失败：${t.javaClass.simpleName}: ${t.message}" }
            }
        }.start()
    }

    // ---------------- voice ----------------

    private fun toggleVoice() {
        if (voiceEnabled) {
            voiceEnabled = false
            btnMic.text = "🎤关"
            voiceEngine?.stop()
            Toast.makeText(this, "语音监听已关闭", Toast.LENGTH_SHORT).show()
        } else {
            voiceEnabled = true
            btnMic.text = "🎤开"
            startVoiceLoop()
            Toast.makeText(this, "语音监听已开启，说“我要找XX”", Toast.LENGTH_SHORT).show()
        }
    }

    private fun maybeStartVoice() {
        if (!voiceEnabled) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startVoiceLoop()
        } else {
            micPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun startVoiceLoop() {
        voiceEngine?.let { it.start(); return }
        voiceEngine = VoiceEngine(
            this,
            onText = { text, _ -> handleVoiceText(text) },
            onError = { msg ->
                runOnUiThread {
                    voiceEnabled = false
                    btnMic.text = "🎤关"
                    Toast.makeText(this, "语音不可用：$msg", Toast.LENGTH_LONG).show()
                }
            }
        ).also { it.start() }
    }

    private val wakePrefixes = arrayOf(
        "我要寻找一下", "我要寻找", "帮我寻找一下", "帮我寻找", "我想寻找一下", "我想寻找",
        "我要找一下", "我想找一下", "帮我找一下", "帮我找", "我要找到", "我想找到",
        "我要找", "我想找", "找一下", "寻找", "帮我", "我要", "我想", "找"
    )

    private val trailingFillers = arrayOf(
        "在哪里", "在哪儿", "在哪", "在哪里呢", "谢谢", "好吗", "行吗", "一下", "呢", "啊", "吧"
    )

    /** Parse "我要找手机和钥匙" -> (["手机","钥匙"], true). Returns hasWake=false if no 找 in text. */
    private fun parseVoiceCommand(raw: String): Pair<List<String>, Boolean> {
        var s = raw.trim()
            .replace(" ", "")
            .replace("。", "")
            .replace("，", "")
            .replace(",", "")
            .replace("、", "")
            .replace("！", "")
            .replace("？", "")
        if (s.isEmpty() || !s.contains("找")) return emptyList<String>() to false
        var changed = true
        while (changed) {
            changed = false
            for (p in wakePrefixes) {
                if (s.startsWith(p) && s.length > p.length) {
                    s = s.substring(p.length); changed = true; break
                }
            }
        }
        for (f in trailingFillers) {
            if (s.endsWith(f) && s.length > f.length) {
                s = s.substring(0, s.length - f.length); break
            }
        }
        // split multi-target: 手机和钥匙 / 手机钥匙 / 手机跟钥匙
        val terms = s.split("和", "跟", "与")
            .map { it.trim() }
            .filter { it.isNotEmpty() && it != "找" && it != "的" }
        return terms to true
    }

    private fun handleVoiceText(raw: String) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastVoiceTriggerAt < 2500) return
        val (terms, hasWake) = parseVoiceCommand(raw)
        if (!hasWake || terms.isEmpty()) return
        val v = vocab ?: return
        val e = engine ?: return
        val matched = ArrayList<Int>()
        val missing = ArrayList<String>()
        for (t in terms) {
            val i = v.indexOf(t)
            if (i >= 0) matched.add(i) else missing.add(t)
        }
        if (matched.isEmpty()) {
            if (missing.isNotEmpty()) {
                lastVoiceTriggerAt = now
                runOnUiThread {
                    Toast.makeText(this, "没听懂“${raw.trim()}”\n可试试：手机 人 猫 狗 钥匙 遥控器", Toast.LENGTH_LONG).show()
                }
            }
            return
        }
        lastVoiceTriggerAt = now
        runOnUiThread {
            e.selectedIndices = matched.distinct()
            tracking = false
            lowStreak = 0
            lastDistM = -1f
            lockedEn = ""
            classMissCount = 0
            validateTick = 0
            overlayView.clearTarget()
            inputEdit.setText(matched.joinToString(",") { v.display(it) })
            val msg = "语音指令：开始寻找 ${matched.joinToString("、") { v.display(it) }}" +
                    if (missing.isEmpty()) "" else "\n未收录已忽略：${missing.joinToString("、")}"
            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
        }
    }

    // ---------------- manual input ----------------

    /** Horizontal position of the box center: 左侧 / 前方 / 右侧. */
    private fun directionOf(box: RectF, frameW: Int): String {
        val cx = box.centerX() / frameW.coerceAtLeast(1)
        return when {
            cx < 0.40f -> "左侧"
            cx <= 0.60f -> "前方"
            else -> "右侧"
        }
    }

    private fun targetInfo(box: RectF, frameW: Int): String {
        val dir = directionOf(box, frameW)
        return if (lastDistM > 0f) String.format(Locale.CHINA, "%.1f米 · %s", lastDistM, dir)
        else "$dir · 测距中"
    }

    private fun refreshDistance(box: RectF) {
        val d = depth
        val dModel = if (d != null) {
            val t0 = System.currentTimeMillis()
            val v = d.estimate(fullFrame, fullW, fullH, box)
            lastDepthMs = System.currentTimeMillis() - t0
            v
        } else -1f
        // Geometric estimate: pinhole size prior. Reliable when the box is large
        // (near field), where the metric depth model is out of distribution.
        val boxWFrac = box.width() / fullW.coerceAtLeast(1)
        val prior = SIZE_PRIOR[lockedEn]
        val dGeo = if (prior != null && boxWFrac > 0.02f) {
            prior / (2f * HFOV_TAN_HALF * boxWFrac)
        } else -1f
        // blend weight: 0 below 22% box width (model), 1 above 52% (geometric)
        val wGeo = ((boxWFrac - 0.22f) / 0.30f).coerceIn(0f, 1f)
        lastDistM = when {
            dGeo > 0f && dModel > 0f -> wGeo * dGeo + (1f - wGeo) * dModel
            dGeo > 0f -> dGeo
            else -> dModel
        }
    }

    @Volatile private var lastDepthMs = 0L

    private fun iou(a: RectF, b: RectF): Float {
        val l = maxOf(a.left, b.left)
        val t = maxOf(a.top, b.top)
        val r = minOf(a.right, b.right)
        val bo = minOf(a.bottom, b.bottom)
        val inter = (r - l).coerceAtLeast(0f) * (bo - t).coerceAtLeast(0f)
        val uni = a.width() * a.height() + b.width() * b.height() - inter
        return if (uni > 0f) inter / uni else 0f
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
        // reset tracking state so the old target's red box disappears immediately
        tracking = false
        lowStreak = 0
        lastDistM = -1f
        lockedEn = ""
        classMissCount = 0
        validateTick = 0
        overlayView.clearTarget()
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
                validateTick++
                val needValidation = validateTick % 4 == 0
                try {
                    if (needValidation) {
                        try {
                            YuvToRgb.convert(image, rgb)
                        } catch (ce: Throwable) {
                            Log.w(TAG, "validation convert failed: ${ce.message}")
                        }
                    }
                    YuvToRgb.convertFull(image, fullFrame)
                } finally {
                    image.close()
                }
                var res: TrackerEngine.TrackResult? = null
                var trackError: Throwable? = null
                try {
                    res = t.track(fullFrame, fullW, fullH)
                } catch (te: Throwable) {
                    Log.e(TAG, "track error", te)
                    trackError = te
                    trackFailCount++
                }
                if (trackError != null) {
                    if (trackFailCount >= 3) {
                        tracking = false
                        tracker = null
                        runOnUiThread {
                            overlayView.clearTarget()
                            Toast.makeText(this@MainActivity, "跟踪引擎异常(${trackError.javaClass.simpleName})，已切回仅检测模式", Toast.LENGTH_LONG).show()
                        }
                    }
                    return
                }
                trackFailCount = 0
                val r = res!!
                if (r.box == null || r.score < 0.25f) {
                    lowStreak++
                } else {
                    lowStreak = 0
                }
                // periodic class re-validation: is the tracked box still the target class?
                var classFail = false
                if (lowStreak < 5 && needValidation && r.box != null) {
                    try {
                        val (vdets, _) = e.detect(rgb, info)
                        val sel = e.selectedIndices
                        val ok = vdets.any { dt ->
                            dt.score >= 0.35f && sel.contains(dt.labelIndex) && iou(dt.box, r.box) >= 0.25f
                        }
                        if (ok) classMissCount = 0 else classMissCount++
                        classFail = classMissCount >= 2
                        if (classFail) Log.i(TAG, "class validation failed ${classMissCount}x, unlocking")
                    } catch (de: Throwable) {
                        Log.w(TAG, "validation detect failed: ${de.message}")
                    }
                }
                val lost = lowStreak >= 5 || classFail
                // refresh distance every 2nd track tick (~0.8s)
                if (!lost && r.box != null) {
                    depthTick++
                    if (depthTick % 2 == 0) refreshDistance(r.box)
                }
                val infoText = if (!lost && r.box != null) targetInfo(r.box, fullW) else ""
                runOnUiThread {
                    if (lost) {
                        tracking = false
                        lastDistM = -1f
                        classMissCount = 0
                        overlayView.clearTarget()
                        val msg = if (classFail) "目标类别校验失败，重新搜索" else "目标丢失，重新搜索"
                        Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show()
                    } else {
                        overlayView.setTarget(r.box, lockedLabel, infoText)
                    }
                    hudText.text = String.format(
                        Locale.CHINA,
                        "跟踪 | %s | 跟踪 %d ms | 深度 %d ms | 置信 %.2f%s",
                        if (infoText.isNotEmpty()) infoText else lockedLabel, r.ms, lastDepthMs, r.score,
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
                        .filter { it.score >= 0.40f && it.box.width() * it.box.height() >= 0.015f * srcInfo.w * srcInfo.h }
                        .maxByOrNull { it.score }
                    var locked = false
                    var lockInfo = ""
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
                        lockedEn = v?.entries?.getOrNull(best.labelIndex)?.en?.lowercase() ?: ""
                        tracking = true
                        lowStreak = 0
                        depthTick = 1
                        validateTick = 0
                        classMissCount = 0
                        locked = true
                    }
                    if (locked) {
                        // immediate first distance reading at lock time
                        refreshDistance(best!!.box)
                        lockInfo = targetInfo(best.box, fullW)
                    }
                    runOnUiThread {
                        if (locked) {
                            overlayView.setTarget(best!!.box, lockedLabel, lockInfo)
                            Toast.makeText(this@MainActivity, "已锁定：$lockedLabel（$lockInfo），开始跟踪", Toast.LENGTH_LONG).show()
                        } else {
                            hudText.text = String.format(
                                Locale.CHINA,
                                "搜索模式%s | 引擎 %s | 推理 %d ms | 端到端 %d ms | %.1f FPS",
                                if (t != null) "" else "（无跟踪器）",
                                e.engineMode, timing.inferMs, totalMs, fpsEma
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

    override fun onDestroy() {
        super.onDestroy()
        voiceEnabled = false
        voiceEngine?.close()
        voiceEngine = null
        cameraExecutor.shutdown()
        engine?.close()
        tracker?.close()
        depth?.close()
        depth = null
    }

    companion object {
        private const val TAG = "LocateCam"
        private const val THROTTLE_MS = 1000L
        private const val TRACK_THROTTLE_MS = 400L

        // Assumed camera horizontal FOV ~64° -> tan(half) ~ 0.625
        private const val HFOV_TAN_HALF = 0.625f

        // Typical real-world widths (meters) of findable objects, for near-field
        // geometric distance: d = realWidth / (2*tan(FOV/2) * boxWidthFraction)
        private val SIZE_PRIOR = mapOf(
            "person" to 0.50f, "cell phone" to 0.07f, "laptop" to 0.34f, "cup" to 0.09f,
            "book" to 0.17f, "bottle" to 0.07f, "chair" to 0.45f, "keyboard" to 0.42f,
            "computer keyboard" to 0.42f, "mouse" to 0.06f, "computer mouse" to 0.06f,
            "tv" to 0.70f, "monitor" to 0.70f, "remote" to 0.055f, "remote control" to 0.055f,
            "backpack" to 0.32f, "handbag" to 0.30f, "suitcase" to 0.38f, "dog" to 0.32f,
            "cat" to 0.28f, "teddy bear" to 0.30f, "clock" to 0.25f, "scissors" to 0.10f,
            "bowl" to 0.15f, "wine glass" to 0.07f, "umbrella" to 0.35f, "vase" to 0.12f,
            "toothbrush" to 0.02f
        )
    }
}
