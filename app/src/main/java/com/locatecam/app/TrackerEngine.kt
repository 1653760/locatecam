package com.locatecam.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.RectF
import android.util.Log
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.sqrt

class TrackerEngine(context: Context) {

    data class TrackResult(val box: RectF?, val score: Float, val ms: Long)

    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var zName = "z"
    private var xName = "x"

    private val templateSize = 192
    private val searchSize = 384
    private val featSz = 24

    private val zBuf = FloatArray(3 * templateSize * templateSize)
    private val xBuf = FloatArray(3 * searchSize * searchSize)
    private val hann = FloatArray(featSz * featSz)
    private val rowSink = ArrayList<FloatArray>(64)
    private val flatTmp = FloatArray(featSz * featSz)

    private val means = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val stds = floatArrayOf(0.229f, 0.224f, 0.225f)

    private var boxX = 0f
    private var boxY = 0f
    private var boxW = 0f
    private var boxH = 0f
    private var frameW = 10000f
    private var frameH = 10000f

    init {
        for (i in 0 until featSz) {
            val hi = 0.5f * (1f - cos(2.0 * Math.PI * (i + 1) / (featSz + 1)).toFloat())
            for (j in 0 until featSz) {
                val hj = 0.5f * (1f - cos(2.0 * Math.PI * (j + 1) / (featSz + 1)).toFloat())
                hann[i * featSz + j] = hi * hj
            }
        }
        val bytes = context.assets.open("tracker.onnx").readBytes()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, opts)
        for ((k, v) in session!!.inputInfo) {
            val s = v.info
            if (s.contains(templateSize.toString())) zName = k.key
            if (s.contains(searchSize.toString())) xName = k.key
        }
        Log.i(TAG, "tracker loaded, z=$zName x=$xName")
    }

    fun setFrameBounds(w: Int, h: Int) {
        frameW = w.toFloat()
        frameH = h.toFloat()
    }

    fun init(frame: IntArray, w: Int, h: Int, box: RectF) {
        frameW = w.toFloat()
        frameH = h.toFloat()
        boxX = box.left
        boxY = box.top
        boxW = box.width()
        boxH = box.height()
        sampleTarget(frame, w, h, 2f, templateSize, zBuf)
    }

    fun track(frame: IntArray, w: Int, h: Int): TrackResult {
        val t0 = System.nanoTime()
        val rf = sampleTarget(frame, w, h, 4f, searchSize, xBuf)
        val session = this.session ?: return TrackResult(null, 0f, 0)
        val zShape = longArrayOf(1, 3, templateSize.toLong(), templateSize.toLong())
        val xShape = longArrayOf(1, 3, searchSize.toLong(), searchSize.toLong())
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(zBuf), zShape).use { zz ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(xBuf), xShape).use { xx ->
                session.run(mapOf(zName to zz, xName to xx)).use { out ->
                    val rect = decode(out[0].value, out[1].value, out[2].value, rf)
                    val t2 = System.nanoTime()
                    TrackResult(rect.first, rect.second, (t2 - t0) / 1_000_000)
                }
            }
        }
    }

    private fun sampleTarget(frame: IntArray, W: Int, H: Int, factor: Float, outSz: Int, out: FloatArray): Float {
        val cropSz = ceil(sqrt(boxW * boxH) * factor).toInt().coerceAtLeast(16)
        val cx = boxX + 0.5f * boxW
        val cy = boxY + 0.5f * boxH
        val x0 = Math.round(cx - cropSz * 0.5f)
        val y0 = Math.round(cy - cropSz * 0.5f)
        val n = outSz * outSz
        for (oy in 0 until outSz) {
            val sy = y0 + oy * (cropSz - 1) / (outSz - 1)
            val row = oy * outSz
            val inRow = sy in 0 until H
            for (ox in 0 until outSz) {
                val sx = x0 + ox * (cropSz - 1) / (outSz - 1)
                val p = if (inRow && sx in 0 until W) frame[sy * W + sx] else 0
                val r = ((p shr 16) and 0xFF) / 255f
                val g = ((p shr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f
                val idx = row + ox
                out[idx] = (r - means[0]) / stds[0]
                out[n + idx] = (g - means[1]) / stds[1]
                out[2 * n + idx] = (b - means[2]) / stds[2]
            }
        }
        return outSz.toFloat() / cropSz
    }

    private fun collectRows(node: Any, sink: MutableList<FloatArray>) {
        if (node is FloatArray) {
            sink.add(node)
        } else if (node is Array<*>) {
            for (child in node) {
                if (child != null) collectRows(child, sink)
            }
        }
    }

    private fun at(rows: List<FloatArray>, channel: Int, y: Int, x: Int): Float =
        rows[channel * featSz + y][x]

    private fun decode(scoreV: Any, sizeV: Any, offsetV: Any, rf: Float): Pair<RectF, Float> {
        rowSink.clear()
        collectRows(scoreV, rowSink)
        val scoreRows = ArrayList(rowSink)

        var best = -1e9f
        var bestIdx = 0
        for (i in 0 until featSz) {
            val row = scoreRows[i]
            for (j in 0 until featSz) {
                val s = hann[i * featSz + j] * row[j]
                if (s > best) {
                    best = s
                    bestIdx = i * featSz + j
                }
            }
        }
        val idxY = bestIdx / featSz
        val idxX = bestIdx % featSz

        rowSink.clear()
        collectRows(sizeV, rowSink)
        val sizeRows = ArrayList(rowSink)
        rowSink.clear()
        collectRows(offsetV, rowSink)
        val offsetRows = ArrayList(rowSink)

        val wN = at(sizeRows, 0, idxY, idxX)
        val hN = at(sizeRows, 1, idxY, idxX)
        val offX = at(offsetRows, 0, idxY, idxX)
        val offY = at(offsetRows, 1, idxY, idxX)

        val cx = (idxX + offX) / featSz * searchSize
        val cy = (idxY + offY) / featSz * searchSize
        val wpx = wN * searchSize
        val hpx = hN * searchSize

        val half = 0.5f * searchSize / rf
        val cxReal = cx / rf + (boxX + 0.5f * boxW) - half
        val cyReal = cy / rf + (boxY + 0.5f * boxH) - half
        val wReal = (wpx / rf).coerceAtLeast(8f)
        val hReal = (hpx / rf).coerceAtLeast(8f)

        val x1 = (cxReal - 0.5f * wReal).coerceIn(0f, (frameW - 10f).coerceAtLeast(0f))
        val y1 = (cyReal - 0.5f * hReal).coerceIn(0f, (frameH - 10f).coerceAtLeast(0f))
        val x2 = (x1 + wReal).coerceAtMost(frameW)
        val y2 = (y1 + hReal).coerceAtMost(frameH)

        val rect = RectF(x1, y1, x2, y2)
        boxX = x1
        boxY = y1
        boxW = x2 - x1
        boxH = y2 - y1
        return rect to best
    }

    fun close() {
        session?.close()
    }

    companion object {
        private const val TAG = "TrackerEngine"
    }
}
