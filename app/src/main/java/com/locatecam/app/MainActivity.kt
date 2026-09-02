package com.locatecam.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
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
    private val mainHandler = Handler(Looper.getMainLooper())
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
    private var trackFailCount = 0

    // ---- voice (hands-free continuous listening) ----
    private var speechRecognizer: SpeechRecognizer? = null
    @Volatile private var voiceEnabled = true
    @Volatile private var lastVoiceTriggerAt = 0L
    private var voiceAvailable = true

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
                runOnUiThread {
                    hudText.text = if (t != null) "就绪（检测+跟踪就绪），说“我要找XX”或输入：手机 / 人 / 杯子 / 书"
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
            speechRecognizer?.stopListening()
            Toast.makeText(this, "语音监听已关闭", Toast.LENGTH_SHORT).show()
        } else {
            if (!voiceAvailable) {
                Toast.makeText(this, "本机不支持语音识别服务", Toast.LENGTH_LONG).show()
                return
            }
            voiceEnabled = true
            btnMic.text = "🎤开"
            maybeStartVoice()
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

    @Synchronized
    private fun startVoiceLoop() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            voiceAvailable = false
            voiceEnabled = false
            btnMic.text = "🎤关"
            Toast.makeText(this, "本机无语音识别服务，语音功能不可用", Toast.LENGTH_LONG).show()
            return
        }
        if (speechRecognizer != null) return
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onEvent(eventType: Int, params: Bundle?) {}

                override fun onPartialResults(partialResults: Bundle?) {
                    val text = partialResults
                        ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        ?.firstOrNull() ?: return
                    if (text.isNotEmpty()) handleVoiceText(text)
                }

                override fun onResults(results: Bundle?) {
                    val text = results
                        ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        ?.firstOrNull()
                    if (!text.isNullOrEmpty()) handleVoiceText(text)
                    restartListening()
                }

                override fun onError(error: Int) {
                    restartListening()
                }
            })
        }
        startListeningNow()
    }

    private fun startListeningNow() {
        val sr = speechRecognizer ?: return
        if (!voiceEnabled) return
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "zh-CN")
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, "zh-CN")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        }
        try {
            sr.startListening(intent)
        } catch (t: Throwable) {
            Log.w(TAG, "startListening failed: ${t.message}")
            mainHandler.postDelayed({ startListeningNow() }, 1000)
        }
    }

    /** SpeechRecognizer sessions end after each utterance; restart to keep listening forever. */
    private fun restartListening() {
        if (!voiceEnabled) return
        mainHandler.postDelayed({
            val sr = speechRecognizer ?: return@postDelayed
            if (voiceEnabled) {
                try { sr.cancel() } catch (_: Throwable) {}
                startListeningNow()
            }
        }, 300)
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
            overlayView.clearTarget()
            inputEdit.setText(matched.joinToString(",") { v.display(it) })
            val msg = "语音指令：开始寻找 ${matched.joinToString("、") { v.display(it) }}" +
                    if (missing.isEmpty()) "" else "\n未收录已忽略：${missing.joinToString("、")}"
            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
        }
    }

    // ---------------- manual input ----------------

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
                val lost = lowStreak >= 5
                runOnUiThread {
                    if (lost) {
                        tracking = false
                        overlayView.clearTarget()
                        Toast.makeText(this@MainActivity, "目标丢失，重新搜索", Toast.LENGTH_SHORT).show()
                    } else {
                        overlayView.setTarget(r.box, lockedLabel)
                    }
                    hudText.text = String.format(
                        Locale.CHINA,
                        "跟踪模式 | 锁定 %s | 跟踪 %d ms | 置信 %.2f%s",
                        lockedLabel, r.ms, r.score,
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
                            overlayView.setTarget(best!!.box, lockedLabel)
                            Toast.makeText(this@MainActivity, "已锁定：$lockedLabel，开始跟踪", Toast.LENGTH_LONG).show()
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
        mainHandler.removeCallbacksAndMessages(null)
        try { speechRecognizer?.destroy() } catch (_: Throwable) {}
        speechRecognizer = null
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
