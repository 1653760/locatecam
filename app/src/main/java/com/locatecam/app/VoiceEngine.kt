package com.locatecam.app

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import java.io.InputStream
import java.util.zip.ZipInputStream

/**
 * Offline continuous speech recognition using Vosk + device microphone.
 * Emits recognized text via callback: (text, isFinal). Partial results are
 * emitted continuously while speaking; final result when the utterance ends.
 */
class VoiceEngine(
    private val context: Context,
    private val onText: (String, Boolean) -> Unit,
    private val onError: (String) -> Unit
) {

    @Volatile private var running = false
    private var thread: Thread? = null
    private var model: Model? = null

    fun start() {
        if (running) return
        running = true
        thread = Thread { runLoop() }.apply { name = "vosk-audio"; start() }
    }

    fun stop() {
        running = false
        thread?.interrupt()
        thread = null
    }

    fun close() {
        stop()
        try { model?.close() } catch (_: Throwable) {}
        model = null
    }

    /** Extract bundled model-cn.zip from assets into filesDir (one-time). */
    private fun ensureModelDir(): File? {
        val dir = File(context.filesDir, "vosk-model")
        val marker = File(dir, ".ok")
        if (marker.exists() && dir.isDirectory) return dir
        try {
            dir.deleteRecursively()
            context.assets.open(MODEL_ZIP).use { input ->
                unzipInto(input, dir)
            }
            marker.createNewFile()
            Log.i(TAG, "vosk model extracted to $dir")
            return dir
        } catch (t: Throwable) {
            Log.e(TAG, "model extract failed: ${t.message}")
            return null
        }
    }

    private fun unzipInto(input: InputStream, target: File) {
        target.mkdirs()
        ZipInputStream(input).use { zis ->
            while (true) {
                val e = zis.nextEntry ?: break
                val f = File(target, e.name)
                if (e.isDirectory) {
                    f.mkdirs()
                } else {
                    f.parentFile?.mkdirs()
                    f.outputStream().use { zis.copyTo(it) }
                }
                zis.closeEntry()
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun runLoop() {
        val m = try {
            val dir = ensureModelDir() ?: run {
                onError("模型解压失败"); return
            }
            Model(dir.absolutePath)
        } catch (t: Throwable) {
            Log.e(TAG, "vosk model load failed", t)
            onError("语音模型加载失败: ${t.message}")
            return
        }
        model = m

        val sampleRate = 16000
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBuf <= 0) { onError("音频初始化失败"); return }
        val record = try {
            AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                maxOf(minBuf * 2, 16384)
            )
        } catch (t: Throwable) {
            Log.e(TAG, "AudioRecord failed", t)
            onError("麦克风初始化失败: ${t.message}")
            return
        }
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            onError("麦克风初始化失败")
            return
        }

        var recognizer: Recognizer? = null
        try {
            recognizer = Recognizer(m, sampleRate.toFloat())
            record.startRecording()
            val shorts = ShortArray(1024)
            val bytes = ByteArray(1024 * 2)
            while (running) {
                val n = record.read(shorts, 0, shorts.size)
                if (n <= 0) continue
                var p = 0
                for (i in 0 until n) {
                    val v = shorts[i].toInt()
                    bytes[p++] = (v and 0xFF).toByte()
                    bytes[p++] = (v shr 8).toByte()
                }
                val isFinal = recognizer.acceptWaveForm(bytes, p)
                val json = if (isFinal) recognizer.result else recognizer.partialResult
                try {
                    val text = JSONObject(json).optString("text").trim()
                    if (text.isNotEmpty()) onText(text, isFinal)
                } catch (_: Throwable) {
                }
            }
        } catch (t: Throwable) {
            if (running) {
                Log.e(TAG, "vosk loop error", t)
                onError("语音识别异常: ${t.message}")
            }
        } finally {
            try { recognizer?.close() } catch (_: Throwable) {}
            try { record.stop() } catch (_: Throwable) {}
            record.release()
        }
    }

    companion object {
        private const val TAG = "VoiceEngine"
        private const val MODEL_ZIP = "model-cn.zip"
    }
}
